import os
import re
import io
import gc
import csv
import sys
import math
import json
import time
import base64
import socket
import argparse
import threading
import webbrowser
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    PANDAS_SKLEARN_AVAILABLE = True
except ImportError:
    PANDAS_SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    for _pkg in ("wordnet", "omw-1.4", "stopwords"):
        try:
            nltk.data.find(f"corpora/{_pkg}")
        except LookupError:
            try:
                nltk.download(_pkg, quiet=True)
            except Exception:
                pass
    NLTK_LEMMATIZER = WordNetLemmatizer()
    _ = NLTK_LEMMATIZER.lemmatize("rooms")
    def lemmatize_word(w: str) -> str:
        try:
            return NLTK_LEMMATIZER.lemmatize(w)
        except Exception:
            return w[:-1] if (len(w) > 3 and w.endswith("s") and not w.endswith("ss")) else w
except Exception:
    def lemmatize_word(w: str) -> str:
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

try:
    from nltk.corpus import wordnet as wn
    _ = wn.synsets("room")
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False

CANONICAL_ASPECTS = ["Service", "Facility", "Experience", "Amenity", "Loyalty", "Branding"]
ID_TO_SENTIMENT = {0: "Not Mentioned", 1: "Negative", 2: "Neutral", 3: "Positive"}
SENTIMENT_TO_ID = {"Not Mentioned": 0, "Negative": 1, "Neutral": 2, "Positive": 3}

CONTRACTIONS_MAP = {
    "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will", "'m": " am"
}

RAW_STOPWORDS = {
    "a", "an", "the", "and", "or", "if", "in", "on", "at", "to", "from", "by", "for",
    "with", "about", "as", "into", "like", "through", "after", "over", "between", "out",
    "of", "during", "before", "under", "around", "among", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "he", "him", "his", "she",
    "her", "hers", "it", "its", "they", "them", "their", "theirs", "this", "that", "these",
    "those", "am", "so", "than", "too", "very", "just", "where", "when", "who", "whom",
    "which", "what", "there", "here", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "then", "once"
}

NEGATION_AND_CONTRAST_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nothing", "nowhere", "hardly",
    "barely", "scarcely", "without", "but", "however", "although", "though", "yet",
    "despite", "unfortunately", "except", "fault"
}

SAFE_STOPWORDS = RAW_STOPWORDS - NEGATION_AND_CONTRAST_WORDS
STOPWORDS_PAPER4 = SAFE_STOPWORDS - {"very", "too"}

OPINION_LEXICON_POS = {
    "good", "great", "excellent", "amazing", "wonderful", "friendly", "helpful", "clean",
    "spacious", "comfortable", "delicious", "lovely", "perfect", "best", "nice", "fantastic",
    "superb", "polite", "attentive", "convenient", "beautiful", "cozy", "relaxing", "awesome",
    "recommend", "recommended", "happy", "pleased", "impressed", "welcoming", "professional",
    "quiet", "modern", "fresh", "tasty", "worth", "luxury", "luxurious", "outstanding", "warm",
    "praised", "rave", "extensive", "must", "spotless", "exceptional", "courteous", "efficient"
}

OPINION_LEXICON_NEG = {
    "bad", "poor", "terrible", "horrible", "awful", "slow", "rude", "dirty", "noisy", "small",
    "old", "outdated", "broken", "cold", "hard", "uncomfortable", "overpriced", "expensive",
    "worst", "disappointed", "disappointing", "unfriendly", "unhelpful", "smell", "smelly",
    "stain", "stained", "cockroach", "bug", "mold", "issue", "problem", "complaint", "wait",
    "waiting", "delayed", "mediocre", "boring", "crowded", "weak", "fault", "damaged", "filthy"
}

HEDGING_MODIFIERS = {
    "moderate", "okay", "ok", "decent", "fair", "acceptable", "average", "slightly",
    "bit", "ordinary", "normal", "passable", "tolerable", "adequate", "fine", "basic"
}

INTENSIFIERS = {
    "very", "extremely", "really", "super", "highly", "totally", "absolutely", "completely",
    "quite", "pretty", "so", "too", "exceptionally", "incredibly", "amazingly", "remarkably",
    "consistently", "truly", "especially", "particularly"
}

ASPECT_KEYWORDS = {
    "Service": {
        "staff", "service", "services", "member", "members", "reception", "receptionist",
        "front", "desk", "manager", "waiter", "waitress", "concierge", "bellboy", "helpful",
        "friendly", "polite", "check", "checkin", "checkout", "attendant", "housekeeping",
        "host", "crew", "hospitality", "team", "assistance", "welcoming", "professional", "attentive"
    },
    "Facility": {
        "facility", "facilities", "maintenance", "traffic", "location", "building", "property",
        "room", "rooms", "common", "area", "areas", "bed", "beds", "bathroom", "shower",
        "toilet", "air", "ac", "conditioning", "clean", "cleanliness", "noise", "noisy",
        "soundproof", "view", "balcony", "floor", "elevator", "lift", "pillow", "mattress",
        "furniture", "decor", "water", "carpet", "lobby", "parking", "surrounding", "hallway"
    },
    "Experience": {
        "stay", "stayed", "experience", "holiday", "vacation", "trip", "night", "nights",
        "overall", "atmosphere", "vibe", "relax", "relaxing", "enjoy", "enjoyed", "memorable",
        "comfort", "comfortable", "quiet", "peaceful", "feeling", "impression", "journey",
        "moment", "couples", "family", "solo", "honeymoon", "getaway", "place"
    },
    "Amenity": {
        "amenity", "amenities", "kitchen", "food", "foods", "meal", "meals", "dish", "dishes",
        "taste", "tasty", "delicious", "flavor", "dining", "breakfast", "buffet", "pool",
        "swimming", "spa", "gym", "fitness", "wifi", "internet", "restaurant", "bar", "drink",
        "drinks", "coffee", "tea", "lounge", "shuttle", "massage", "rooftop", "menu", "cuisine",
        "activity", "activities"
    },
    "Loyalty": {
        "back", "return", "returned", "returning", "again", "times", "visit", "visits",
        "visited", "revisit", "recommend", "recommended", "definitely", "choose", "future",
        "next", "repeat", "loyal", "regular", "choice", "staying", "book", "booking", "must"
    },
    "Branding": {
        "star", "stars", "luxury", "luxurious", "brand", "chain", "standard", "reputation",
        "class", "five", "four", "management", "value", "price", "pricing", "worth",
        "prestigious", "accor", "sofitel", "pullman", "novotel", "marriott", "hilton",
        "intercontinental", "sheraton", "flagship", "resort"
    }
}

ASPECT_ID_MAP = {asp: idx + 1 for idx, asp in enumerate(CANONICAL_ASPECTS)}
ASPECT_ID_MAP["None"] = 0

FEATURE_COLS_20 = [
    "feat_aspect_count", "feat_dominant_aspect", "feat_net_sentiment_score", "feat_sentiment_conflict",
    "feat_evidence_density", "feat_avg_rationale_len", "feat_has_cross_overlap", "feat_service_sentiment",
    "feat_facility_sentiment", "feat_experience_sentiment", "feat_amenity_sentiment", "feat_loyalty_sentiment",
    "feat_branding_sentiment", "feat_aspect_coverage_ratio", "feat_neutral_hedging_density", "feat_consensus_strength",
    "feat_lexical_polarity_intensity", "feat_nltk_clean_token_ratio", "feat_nltk_opinion_negation_density",
    "feat_ebr_aspect_prototype_sim"
]

AMBIGUOUS_HOTEL_TERMS = {
    "sharp", "cold", "warm", "hard", "soft", "small", "high", "low",
    "light", "heavy", "old", "dark", "basic", "quiet", "plain", "rich", "fair", "fine"
}

MODEL_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
MAX_CACHED_MODELS = 2
LOADED_MODEL_CACHE: Dict[str, Any] = {}
CURRENT_MODEL_DIR: Optional[Path] = None
CURRENT_SELECTED_MODEL_FILE: Optional[Path] = None
SELECTED_MODEL_FILES: Dict[str, Path] = {}

# In-memory accumulated benchmark table across the session (keyed by display_id e.g. "model lần 1/modelv1.pt")
BENCHMARK_HISTORY: Dict[str, Dict[str, Any]] = {}
BENCHMARK_STATE: Dict[str, Any] = {
    "is_running": False,
    "progress": 0.0,
    "message": "Sẵn sàng",
    "model_id": "",
    "result": None,
    "error": None
}
BENCHMARK_LOCK = threading.Lock()
MODEL_EXEC_LOCK = threading.Lock()


def _release_device_memory():
    gc.collect()
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

# Cached test split dataframe & pre-tokenized structures to speed up repeated benchmark runs
_CACHED_TEST_DF: Optional[Any] = None
_CACHED_BERT_TOK: Optional[Any] = None
_CACHED_TEST_CRAM_TENSORS: Optional[Dict[str, Any]] = None
_CACHED_TEST_P1_TENSORS: Optional[Dict[str, Any]] = None
_CACHED_TEST_P9_TENSORS: Optional[Dict[str, Any]] = None
_LEMMA_CACHE: Dict[str, str] = {}


def cached_lemmatize(w: str) -> str:
    res = _LEMMA_CACHE.get(w)
    if res is None:
        res = lemmatize_word(w)
        _LEMMA_CACHE[w] = res
    return res


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_model_dir() -> Path:
    cwd_model = Path.cwd() / "model"
    script_model = Path(__file__).resolve().parent / "model"
    parent_model = Path(__file__).resolve().parent.parent / "model"
    for candidate in (cwd_model, script_model, parent_model):
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return cwd_model.resolve()


def get_model_dir() -> Path:
    global CURRENT_MODEL_DIR
    if CURRENT_MODEL_DIR is None:
        CURRENT_MODEL_DIR = get_default_model_dir()
    return CURRENT_MODEL_DIR


def set_model_dir(dir_path_str: str) -> Tuple[bool, str, Path]:
    global CURRENT_MODEL_DIR
    cleaned = str(dir_path_str or "").strip().strip('"').strip("'")
    if not cleaned:
        return False, "Vui lòng chọn thư mục chứa model.", get_model_dir()
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if p.is_file():
        set_selected_model_file(str(p))
        p = p.parent
    if not p.exists() or not p.is_dir():
        return False, f"Thư mục không tồn tại: {p}", get_model_dir()
    CURRENT_MODEL_DIR = p
    return True, "OK", CURRENT_MODEL_DIR


def compute_model_display_id(file_path: Path, base_dir: Optional[Path] = None) -> str:
    """
    Return the model filename (e.g. 'modelv1.pt', 'modelmethod1.pt').
    """
    return file_path.resolve().name


def get_model_file_info(file_path: Path) -> Dict[str, Any]:
    resolved = file_path.resolve()
    abs_s = resolved.as_posix()
    display_id = compute_model_display_id(resolved)
    stat = resolved.stat()
    size_mb = stat.st_size / (1024 * 1024)
    size_label = f"{size_mb:.2f} MB" if size_mb >= 0.1 else f"{max(1, int(stat.st_size / 1024))} KB"
    return {
        "filename": abs_s,
        "name": resolved.name,
        "display_id": display_id,
        "abs_path": abs_s,
        "size_label": size_label
    }


def set_selected_model_file(file_path_str: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    global CURRENT_SELECTED_MODEL_FILE, CURRENT_MODEL_DIR
    cleaned = str(file_path_str or "").strip().strip('"').strip("'")
    if not cleaned:
        return False, "Vui lòng chọn file model.", None
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_file():
        return False, f"File model không tồn tại: {p}", None
    if p.suffix.lower() not in MODEL_EXTENSIONS:
        return (
            False,
            f"File '{p.name}' không đúng định dạng model ({', '.join(sorted(MODEL_EXTENSIONS))}).",
            None
        )
    CURRENT_SELECTED_MODEL_FILE = p
    CURRENT_MODEL_DIR = p.parent
    info = get_model_file_info(p)
    SELECTED_MODEL_FILES[info["abs_path"]] = p
    SELECTED_MODEL_FILES[info["display_id"]] = p
    return True, "OK", info


def choose_model_file_dialog() -> Optional[str]:
    initial_dir = str(get_model_dir()) if get_model_dir().exists() else str(Path.cwd())
    if sys.platform == "darwin":
        try:
            import subprocess
            escaped_dir = initial_dir.replace('"', '\\"')
            script = (
                f'try\n'
                f'  set defaultDir to POSIX file "{escaped_dir}" as alias\n'
                f'  set chosenFile to choose file with prompt "Chọn file model (.pt, .pth, .bin, .safetensors, .ckpt):" default location defaultDir\n'
                f'on error\n'
                f'  set chosenFile to choose file with prompt "Chọn file model (.pt, .pth, .bin, .safetensors, .ckpt):"\n'
                f'end try\n'
                f'POSIX path of chosenFile'
            )
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
            return None
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Chọn file model",
            filetypes=[("Model files", "*.pt *.pth *.bin *.safetensors *.ckpt"), ("All files", "*.*")]
        )
        root.destroy()
        if selected:
            return str(selected)
    except Exception:
        pass
    return None


def choose_save_file_dialog(default_name: str = "benchmark_chart.png") -> Optional[str]:
    initial_dir = str(get_repo_root())
    if sys.platform == "darwin":
        try:
            import subprocess
            escaped_dir = initial_dir.replace('"', '\\"')
            escaped_name = default_name.replace('"', '\\"')
            script = (
                f'try\n'
                f'  set defaultDir to POSIX file "{escaped_dir}" as alias\n'
                f'  set chosenFile to choose file name with prompt "Chọn vị trí lưu hình ảnh (.png):" default name "{escaped_name}" default location defaultDir\n'
                f'on error\n'
                f'  set chosenFile to choose file name with prompt "Chọn vị trí lưu hình ảnh (.png):" default name "{escaped_name}"\n'
                f'end try\n'
                f'POSIX path of chosenFile'
            )
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
            if proc.returncode == 0 and proc.stdout.strip():
                out_path = proc.stdout.strip()
                if not out_path.lower().endswith(".png"):
                    out_path += ".png"
                return out_path
            return None
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=".png",
            title="Chọn vị trí lưu hình ảnh",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")]
        )
        root.destroy()
        if selected:
            out_path = str(selected)
            if not out_path.lower().endswith(".png"):
                out_path += ".png"
            return out_path
    except Exception:
        pass
    return None


def choose_save_directory_dialog() -> Optional[str]:
    initial_dir = str(get_repo_root())
    if sys.platform == "darwin":
        try:
            import subprocess
            escaped_dir = initial_dir.replace('"', '\\"')
            script = (
                f'try\n'
                f'  set defaultDir to POSIX file "{escaped_dir}" as alias\n'
                f'  set chosenFolder to choose folder with prompt "Chọn thư mục lưu hình ảnh Visualization:" default location defaultDir\n'
                f'on error\n'
                f'  set chosenFolder to choose folder with prompt "Chọn thư mục lưu hình ảnh Visualization:"\n'
                f'end try\n'
                f'POSIX path of chosenFolder'
            )
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
            return None
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=initial_dir, title="Chọn thư mục lưu hình ảnh Visualization")
        root.destroy()
        if selected:
            return str(selected)
    except Exception:
        pass
    return None


def save_data_url_png(data_url: str, target_path: Path) -> None:
    if "," in data_url:
        b64_part = data_url.split(",", 1)[1]
    else:
        b64_part = data_url
    raw_bytes = base64.b64decode(b64_part)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(raw_bytes)


def resolve_model_file(model_key: str) -> Optional[Path]:
    cleaned = str(model_key or "").strip()
    if not cleaned:
        if CURRENT_SELECTED_MODEL_FILE is not None and CURRENT_SELECTED_MODEL_FILE.is_file():
            return CURRENT_SELECTED_MODEL_FILE
        return None

    if cleaned in SELECTED_MODEL_FILES:
        cand = SELECTED_MODEL_FILES[cleaned]
        if cand.is_file():
            return cand

    repo_root = get_repo_root().resolve()
    model_dir = get_model_dir().resolve()
    for base in (repo_root, model_dir):
        try:
            direct = (base / cleaned).resolve()
            is_inside = direct.is_relative_to(base) if hasattr(direct, "is_relative_to") else str(direct).startswith(str(base))
            if is_inside and direct.is_file() and direct.suffix.lower() in MODEL_EXTENSIONS:
                return direct
        except Exception:
            pass
    return None


def find_parquet_file() -> Optional[Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [
        repo_root / "data" / "df" / "hotel_reviews.parquet",
        Path.cwd() / "data" / "df" / "hotel_reviews.parquet",
        script_dir / "data" / "df" / "hotel_reviews.parquet",
        repo_root / "hotel_reviews.parquet",
        Path.cwd() / "hotel_reviews.parquet",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def parse_annotations_safe(anns: Any) -> List[Dict[str, Any]]:
    if anns is None:
        return []
    if isinstance(anns, np.ndarray):
        anns = anns.tolist()
    elif isinstance(anns, str):
        try:
            anns = json.loads(anns)
        except Exception:
            return []
    if not isinstance(anns, list):
        return []
    return [dict(a) for a in anns if isinstance(a, dict)]


def load_test_dataframe():
    global _CACHED_TEST_DF
    if _CACHED_TEST_DF is not None:
        return _CACHED_TEST_DF
    if not PANDAS_SKLEARN_AVAILABLE:
        raise RuntimeError("Thiếu thư viện pandas hoặc scikit-learn để đọc tập test.")
    pq_path = find_parquet_file()
    if not pq_path:
        raise FileNotFoundError("Không tìm thấy file dữ liệu data/df/hotel_reviews.parquet.")
    df = pd.read_parquet(pq_path)
    if "split" in df.columns:
        df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    else:
        # Fallback 80/10/10 test slice (last 10%)
        n_test = max(1, int(round(len(df) * 0.10)))
        df_test = df.iloc[-n_test:].copy().reset_index(drop=True)
    _CACHED_TEST_DF = df_test
    return _CACHED_TEST_DF


def nltk_negation_aware_preprocess(text: str) -> Tuple[List[str], List[str]]:
    lower_txt = unicodedata.normalize("NFC", str(text)).lower()
    for k, v in CONTRACTIONS_MAP.items():
        lower_txt = re.sub(r"\b" + re.escape(k) + r"\b", v, lower_txt)
        lower_txt = lower_txt.replace(k, v)
    lower_txt = re.sub(r"@\w+|https?://\S+|www\.\S+", " ", lower_txt)
    raw_tokens = re.findall(r"[a-z]+", lower_txt)
    clean_tokens = [cached_lemmatize(t) for t in raw_tokens if t not in SAFE_STOPWORDS and len(t) > 1]
    return raw_tokens, clean_tokens


def paper1_clean_text(text: str) -> str:
    txt = unicodedata.normalize("NFC", str(text)).lower()
    for k, v in CONTRACTIONS_MAP.items():
        txt = re.sub(r"\b" + re.escape(k) + r"\b", v, txt)
        txt = txt.replace(k, v)
    txt = re.sub(r"@\w+|https?://\S+|www\.\S+", " ", txt)
    tokens = re.findall(r"[a-z]+", txt)
    filtered = [t for t in tokens if t not in SAFE_STOPWORDS and len(t) > 1]
    return " ".join(filtered) if filtered else txt


def paper4_preprocess_tokens(text: str) -> List[str]:
    norm = unicodedata.normalize("NFC", str(text)).lower()
    for k, v in CONTRACTIONS_MAP.items():
        norm = re.sub(r"\b" + re.escape(k) + r"\b", v, norm)
        norm = norm.replace(k, v)
    unigrams = [cached_lemmatize(t) for t in re.findall(r"[a-z]+", norm) if t not in STOPWORDS_PAPER4 and len(t) > 1]
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def enrich_text_with_wordnet_wsd(text: str) -> str:
    raw = unicodedata.normalize("NFC", str(text).strip())
    if not WORDNET_AVAILABLE:
        return raw
    words = re.findall(r"[a-zA-Z]+", raw)
    context_set = {w.lower() for w in words if len(w) > 2}
    sense_glosses = []
    for w in words:
        wl = w.lower()
        if wl in AMBIGUOUS_HOTEL_TERMS:
            try:
                synsets = wn.synsets(wl)
            except Exception:
                synsets = []
            if synsets:
                best_syn = synsets[0]
                best_sim = -1.0
                for syn in synsets[:4]:
                    gloss_words = set(re.findall(r"[a-z]+", (syn.definition() + " " + " ".join(syn.examples())).lower()))
                    if gloss_words:
                        inter = len(context_set & gloss_words)
                        cos_sim = inter / math.sqrt(max(len(context_set), 1) * len(gloss_words))
                        if cos_sim > best_sim:
                            best_sim = cos_sim
                            best_syn = syn
                sense_glosses.append(f"{wl}: {best_syn.definition()}")
                if len(sense_glosses) >= 2:
                    break
    if sense_glosses:
        return raw + " [SEP] " + " ; ".join(sense_glosses)
    return raw


def compute_span_calibrated_sentiment(span_text: str, default_sent_id: int) -> int:
    raw_toks, _ = nltk_negation_aware_preprocess(span_text)
    pos_cnt = sum(1 for t in raw_toks if t in OPINION_LEXICON_POS)
    neg_cnt = sum(1 for t in raw_toks if t in OPINION_LEXICON_NEG)
    hedge_cnt = sum(1 for t in raw_toks if t in HEDGING_MODIFIERS and t not in {"never"})

    for idx_t, tok in enumerate(raw_toks[:-1]):
        if tok in {"not", "never", "no", "hardly"} and raw_toks[idx_t + 1] != "forget":
            nxt = raw_toks[idx_t + 1]
            if nxt in OPINION_LEXICON_POS:
                pos_cnt = max(0, pos_cnt - 1)
                neg_cnt += 1
            elif nxt in OPINION_LEXICON_NEG:
                neg_cnt = max(0, neg_cnt - 1)
                pos_cnt += 1

    base_sent = default_sent_id if default_sent_id in (1, 2, 3) else 2

    if neg_cnt > pos_cnt and neg_cnt >= 1:
        return 1
    elif pos_cnt > neg_cnt and pos_cnt >= 1:
        if hedge_cnt > 0 and pos_cnt == 1:
            return 2
        return 3
    elif pos_cnt == neg_cnt and pos_cnt > 0:
        return 2
    elif hedge_cnt > 0:
        return 2
    else:
        return base_sent


def extract_20_features_for_new_review(review_text: str) -> np.ndarray:
    raw_text = unicodedata.normalize("NFC", str(review_text).strip())
    char_len = max(len(raw_text), 1)
    raw_tokens, clean_tokens = nltk_negation_aware_preprocess(raw_text)
    total_raw_toks = max(len(raw_tokens), 1)

    clean_token_ratio = float(min(len(clean_tokens) / total_raw_toks, 1.0))
    opinion_neg_cnt = sum(
        1 for t in raw_tokens
        if (t in OPINION_LEXICON_POS or t in OPINION_LEXICON_NEG or t in NEGATION_AND_CONTRAST_WORDS or t in INTENSIFIERS)
    )
    opinion_neg_density = float(min(opinion_neg_cnt / total_raw_toks, 1.0))

    neutral_hedge_set = HEDGING_MODIFIERS | {"but", "however", "although", "though", "yet", "despite"}
    neutral_hedging_density = float(min(sum(1 for t in raw_tokens if t in neutral_hedge_set) / total_raw_toks, 1.0))

    pos_cnt = sum(1 for t in raw_tokens if t in OPINION_LEXICON_POS)
    neg_cnt = sum(1 for t in raw_tokens if t in OPINION_LEXICON_NEG)
    intens_cnt = sum(1 for t in raw_tokens if t in INTENSIFIERS)
    for idx_t, tok in enumerate(raw_tokens[:-1]):
        if tok in {"not", "never", "no", "hardly"} and raw_tokens[idx_t + 1] != "forget":
            nxt = raw_tokens[idx_t + 1]
            if nxt in OPINION_LEXICON_POS:
                pos_cnt = max(0, pos_cnt - 1)
                neg_cnt += 1
            elif nxt in OPINION_LEXICON_NEG:
                neg_cnt = max(0, neg_cnt - 1)
                pos_cnt += 1
    base_pol = (pos_cnt - neg_cnt) / max(pos_cnt + neg_cnt, 1) if (pos_cnt + neg_cnt) > 0 else 0.0
    intens_boost = min(0.35, 0.12 * intens_cnt) * (1.0 if base_pol >= 0 else -1.0) if base_pol != 0 else 0.0
    lexical_polarity_intensity = float(np.clip(base_pol + intens_boost, -1.0, 1.0))

    tok_counts = Counter(clean_tokens)
    aspect_sims = []
    for asp in CANONICAL_ASPECTS:
        kw_set = ASPECT_KEYWORDS[asp]
        overlap_w = sum(cnt for tok, cnt in tok_counts.items() if tok in kw_set)
        norm_sim = overlap_w / math.sqrt(len(clean_tokens) * len(kw_set)) if clean_tokens else 0.0
        aspect_sims.append(min(1.0, norm_sim * 3.5))
    top_sims = sorted(aspect_sims, reverse=True)[:3]
    ebr_proto_sim = float(sum(top_sims) / len(top_sims)) if top_sims else 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", raw_text) if s.strip()]
    aspect_scores = {}
    matched_sentences_per_aspect = defaultdict(list)
    overlap_flag = 0

    for sent in sentences:
        s_raw, s_clean = nltk_negation_aware_preprocess(sent)
        s_all = set(s_raw) | set(s_clean)
        matched_asps = [asp for asp in CANONICAL_ASPECTS if any(t in ASPECT_KEYWORDS[asp] for t in s_all)]
        if len(matched_asps) >= 2:
            overlap_flag = 1
        pos_hits = sum(1 for t in s_raw if t in OPINION_LEXICON_POS)
        neg_hits = sum(1 for t in s_raw if t in OPINION_LEXICON_NEG)
        hedge_hits = sum(1 for t in s_raw if t in HEDGING_MODIFIERS)
        for idx_t, tok in enumerate(s_raw[:-1]):
            if tok in {"not", "never", "no", "hardly"} and s_raw[idx_t + 1] != "forget":
                nxt = s_raw[idx_t + 1]
                if nxt in OPINION_LEXICON_POS:
                    pos_hits = max(0, pos_hits - 1)
                    neg_hits += 1
                elif nxt in OPINION_LEXICON_NEG:
                    neg_hits = max(0, neg_hits - 1)
                    pos_hits += 1
        sent_pol = (pos_hits - neg_hits - 0.25 * hedge_hits) / max(pos_hits + neg_hits + hedge_hits, 1) if (pos_hits + neg_hits + hedge_hits) > 0 else 0.25
        for asp in matched_asps:
            matched_sentences_per_aspect[asp].append((sent, sent_pol))

    for asp in CANONICAL_ASPECTS:
        items = matched_sentences_per_aspect.get(asp, [])
        aspect_scores[asp] = float(np.clip(np.mean([sc for _, sc in items]), -1.0, 1.0)) if items else 0.0

    active_aspects = [asp for asp in CANONICAL_ASPECTS if matched_sentences_per_aspect.get(asp)]
    asp_count = len(active_aspects)
    dominant_aspect = ASPECT_ID_MAP.get(active_aspects[0], 0) if active_aspects else 0
    active_vals = [aspect_scores[a] for a in active_aspects] if active_aspects else [0.0]
    net_sentiment = float(np.mean(active_vals))
    sent_conflict = 1 if (any(v > 0.1 for v in active_vals) and any(v < -0.1 for v in active_vals)) else 0
    all_matched_lens = [len(s) for items in matched_sentences_per_aspect.values() for s, _ in items]
    evidence_density = float(min(sum(all_matched_lens) / char_len, 1.0)) if all_matched_lens else clean_token_ratio
    avg_rat_len = float(np.mean(all_matched_lens)) if all_matched_lens else float(char_len / max(asp_count, 1))
    coverage_ratio = float(min(asp_count / 6.0, 1.0))
    consensus_strength = float(abs(net_sentiment))

    return np.array([
        asp_count, dominant_aspect, net_sentiment, sent_conflict,
        evidence_density, avg_rat_len, overlap_flag,
        aspect_scores["Service"], aspect_scores["Facility"], aspect_scores["Experience"],
        aspect_scores["Amenity"], aspect_scores["Loyalty"], aspect_scores["Branding"],
        coverage_ratio, neutral_hedging_density, consensus_strength,
        lexical_polarity_intensity, clean_token_ratio, opinion_neg_density, ebr_proto_sim
    ], dtype=np.float32)


def extract_18_features_for_new_review(review_text: str) -> np.ndarray:
    feat20 = extract_20_features_for_new_review(review_text)
    return feat20[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19]]


if TORCH_AVAILABLE:
    def decode_corn_predictions(corn_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(corn_logits)
        cum_probs = torch.cumprod(probs, dim=-1)
        return (cum_probs >= 0.5).sum(dim=-1)

    class CRAM_ABSA_MultiAspectModel(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", hidden_dim=768, num_features=20, proj_dim=256, attn_heads=8, num_aspects=6, num_sent_classes=4, corn_out_dim=2, max_comment_len=48, use_cross_attn=True):
            super().__init__()
            self.num_aspects = num_aspects
            self.num_sent_classes = num_sent_classes
            self.corn_out_dim = corn_out_dim
            self.hidden_dim = hidden_dim
            self.max_comment_len = max_comment_len
            self.use_cross_attn = use_cross_attn
            self.canonical_aspects = list(CANONICAL_ASPECTS)

            self.bert = AutoModel.from_pretrained(backbone_name)
            self.vocab_size = self.bert.config.vocab_size

            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(proj_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.residual_gate = nn.Parameter(torch.tensor(0.28))
            self.ebr_aspect_prototypes = nn.Parameter(torch.randn(num_aspects, hidden_dim) * 0.02)
            if use_cross_attn:
                self.aspect_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, dropout=0.1, batch_first=True)
            self.rationale_proj = nn.Linear(hidden_dim, hidden_dim)
            self.rationale_bias = nn.Parameter(torch.zeros(num_aspects, 1))

            self.aspect_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                    nn.Dropout(0.20),
                    nn.Linear(proj_dim, num_sent_classes)
                )
                for _ in range(num_aspects)
            ])

            self.corn_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(proj_dim, corn_out_dim)
            )

            self.commentary_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
            self.commentary_out = nn.Linear(hidden_dim, self.vocab_size, bias=False)
            self.commentary_out.weight = self.bert.embeddings.word_embeddings.weight

        def forward(self, input_ids, attention_mask, dense_features):
            bsz = input_ids.size(0)
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_hidden = bert_out.last_hidden_state
            cls_hidden = seq_hidden[:, 0, :]

            feat_emb = self.feature_proj(dense_features)
            conditioned_seq = seq_hidden + self.residual_gate * feat_emb.unsqueeze(1)
            conditioned_cls = cls_hidden + feat_emb

            token_proj = self.rationale_proj(conditioned_seq)
            proto_norm = F.normalize(self.ebr_aspect_prototypes, p=2, dim=-1)
            token_norm = F.normalize(token_proj, p=2, dim=-1)
            bilinear_scores = torch.einsum("bld,ad->bal", token_proj, self.ebr_aspect_prototypes) / math.sqrt(self.hidden_dim)
            cosine_scores = torch.einsum("bld,ad->bal", token_norm, proto_norm)

            if self.use_cross_attn and hasattr(self, "aspect_cross_attn"):
                queries = self.ebr_aspect_prototypes.unsqueeze(0).expand(bsz, -1, -1)
                cross_ctx, cross_attn_weights = self.aspect_cross_attn(
                    query=queries,
                    key=conditioned_seq,
                    value=conditioned_seq,
                    key_padding_mask=(attention_mask == 0),
                    need_weights=True
                )
                valid_len = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(1).float().clamp(min=1.0)
                rel_attn = (cross_attn_weights * valid_len).clamp(min=1e-4)
                raw_rat_logits = bilinear_scores + cosine_scores + 0.5 * torch.log(rel_attn) + self.rationale_bias
            else:
                cross_ctx = None
                raw_rat_logits = bilinear_scores + cosine_scores + self.rationale_bias

            pad_mask = attention_mask.unsqueeze(1).float()
            raw_rat_probs = torch.sigmoid(raw_rat_logits) * pad_mask

            denom = raw_rat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            rat_pooled = torch.einsum("bal,bld->bad", raw_rat_probs, conditioned_seq) / denom
            masked_pooled = 0.55 * rat_pooled + 0.45 * cross_ctx if cross_ctx is not None else rat_pooled

            aspect_logits_list = []
            for a_idx in range(self.num_aspects):
                asp_ctx = torch.cat([masked_pooled[:, a_idx, :], feat_emb], dim=-1)
                aspect_logits_list.append(self.aspect_classifiers[a_idx](asp_ctx))

            aspect_logits = torch.stack(aspect_logits_list, dim=1)
            asp_present_logit = (aspect_logits[:, :, 1:].logsumexp(dim=-1, keepdim=True) - aspect_logits[:, :, 0:1]).detach()
            rationale_logits = raw_rat_logits + 0.35 * asp_present_logit.clamp(min=-6.0, max=4.0)
            rationale_probs = torch.sigmoid(rationale_logits) * pad_mask

            global_aspect_ctx = masked_pooled.mean(dim=1)
            corn_input = torch.cat([conditioned_cls, global_aspect_ctx], dim=-1)
            corn_logits = self.corn_head(corn_input)

            return {
                "aspect_logits": aspect_logits,
                "rationale_logits": rationale_logits,
                "rationale_probs": rationale_probs,
                "corn_logits": corn_logits,
                "aspect_embeddings": masked_pooled,
                "seq_hidden": conditioned_seq
            }

        @torch.no_grad()
        def generate_aspect_commentary(self, aspect_idx: int, outputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, raw_review_text: Optional[str] = None) -> str:
            aspect_name = self.canonical_aspects[aspect_idx]
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len]
            scores = rat_prob[:valid_len].detach().cpu().numpy()

            h_state = outputs["aspect_embeddings"][:, aspect_idx, :].unsqueeze(0).contiguous()
            cur_tok = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long, device=input_ids.device)
            gen_ids = []

            allowed_tokens = set(tok_ids.tolist()) | {tokenizer.sep_token_id, tokenizer.pad_token_id}
            vocab_mask = torch.full((self.vocab_size,), -1e4, device=input_ids.device)
            for tid in allowed_tokens:
                vocab_mask[tid] = 0.0

            for _ in range(self.max_comment_len - 1):
                emb = self.bert.embeddings.word_embeddings(cur_tok)
                dec_out, h_state = self.commentary_gru(emb, h_state)
                step_logits = self.commentary_out(dec_out[:, -1, :]).squeeze(0) + vocab_mask
                for prev_id in gen_ids[-4:]:
                    step_logits[prev_id] -= 3.0
                next_id = int(torch.argmax(step_logits).item())
                if next_id in (tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id):
                    break
                gen_ids.append(next_id)
                cur_tok = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)

            gru_tok_set = set(gen_ids)
            if raw_review_text is None or not str(raw_review_text).strip():
                raw_review_text = tokenizer.decode(tok_ids[1:max(1, valid_len - 1)], skip_special_tokens=True).strip()

            raw_text = unicodedata.normalize("NFC", str(raw_review_text).strip())
            enc_off = tokenizer(
                raw_text,
                max_length=input_ids.size(1),
                truncation=True,
                return_offsets_mapping=True
            )
            offsets = enc_off["offset_mapping"]

            sent_spans = []
            for m in re.finditer(r"[^.!?]+[.!?]*", raw_text):
                s_str = m.group(0).strip()
                if len(s_str) >= 3:
                    sent_spans.append((m.start(), m.end(), s_str))
            if not sent_spans:
                return raw_text

            asp_kw_set = ASPECT_KEYWORDS.get(aspect_name, set())
            scored_units = []
            for s_idx, (s_char, e_char, s_str) in enumerate(sent_spans):
                tok_indices = [
                    t_i for t_i, (ts, te) in enumerate(offsets)
                    if te > ts and ts < e_char and te > s_char and t_i < valid_len
                ]
                rat_sc = float(np.mean([scores[t_i] for t_i in tok_indices])) if tok_indices else 0.0
                gru_ov = sum(1 for t_i in tok_indices if int(tok_ids[t_i].item()) in gru_tok_set) / max(len(tok_indices), 1)
                s_raw_toks, s_clean_toks = nltk_negation_aware_preprocess(s_str)
                s_all_toks = set(s_raw_toks) | set(s_clean_toks)
                kw_hits = sum(1 for w in s_all_toks if w in asp_kw_set)
                if aspect_name == "Loyalty" and re.search(r"\b(many times|return visit|came back|come back|been here|stay again|recommend|book early|must go)\b", s_str.lower()):
                    kw_hits += 2
                ebr_sc = min(1.0, kw_hits * 0.45)
                total_sc = 0.40 * rat_sc + 0.15 * gru_ov + 0.45 * ebr_sc
                scored_units.append((total_sc, kw_hits, s_idx, s_str))

            scored_units.sort(key=lambda x: (x[1] > 0, x[0]), reverse=True)
            best_sc, best_kw, best_idx, best_sent = scored_units[0]
            selected = [(best_idx, best_sent)]

            for sc, kw_h, s_i, s_str in scored_units[1:]:
                if kw_h > 0 and sc >= best_sc * 0.65 and abs(s_i - best_idx) == 1 and len(selected) < 2:
                    selected.append((s_i, s_str))

            selected.sort(key=lambda x: x[0])
            extracted_evidence = " ".join(s for _, s in selected).strip()
            return extracted_evidence if extracted_evidence else best_sent


    class Paper1BERTModel(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", hidden_size_bert=768, hidden_size_classifier=50, num_aspects=6, num_sent_classes=4):
            super().__init__()
            self.num_aspects = num_aspects
            self.bert = AutoModel.from_pretrained(backbone_name)
            self.aspect_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size_bert, hidden_size_classifier), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size_classifier, num_sent_classes))
                for _ in range(num_aspects)
            ])
            self.token_scorer = nn.Linear(hidden_size_bert, num_aspects)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_h = out.last_hidden_state
            cls_h = seq_h[:, 0, :]
            tok_scores = torch.sigmoid(self.token_scorer(seq_h)).transpose(1, 2) * attention_mask.unsqueeze(1).float()
            return {
                "aspect_logits": torch.stack([head(cls_h) for head in self.aspect_heads], dim=1),
                "rationale_probs": tok_scores
            }

        @torch.no_grad()
        def extract_evidence_span(self, aspect_idx: int, review_text: str, outputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, bert_tok) -> str:
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len].cpu().tolist()
            scores = rat_prob[:valid_len].cpu().numpy()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text.strip()) if s.strip()]
            if len(sentences) > 1:
                scored_sents = []
                for s in sentences:
                    s_ids = set(bert_tok.encode(paper1_clean_text(s), add_special_tokens=False))
                    hit_scores = [scores[i] for i in range(1, valid_len - 1) if tok_ids[i] in s_ids]
                    scored_sents.append((float(np.mean(hit_scores)) if hit_scores else 0.0, s))
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                return scored_sents[0][1]
            return review_text.strip()


    class Paper4MultiRegionCNN(nn.Module):
        def __init__(self, vocab_size=6001, w2v_dim=100, region_sizes=(2, 3, 4), filters_per_region=2, num_aspects=6, num_sent_classes=4):
            super().__init__()
            self.num_aspects = num_aspects
            self.embedding = nn.Embedding(vocab_size, w2v_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=w2v_dim, out_channels=filters_per_region, kernel_size=k)
                for k in region_sizes
            ])
            total_filters = len(region_sizes) * filters_per_region
            self.aspect_fc = nn.ModuleList([nn.Linear(total_filters, num_sent_classes) for _ in range(num_aspects)])

            self.register_buffer("svm_weights", torch.zeros(num_aspects, num_sent_classes, vocab_size))
            self.register_buffer("svm_biases", torch.zeros(num_aspects, num_sent_classes))

        def forward_cnn(self, token_ids: torch.Tensor):
            emb = self.embedding(token_ids).transpose(1, 2)
            pooled = [F.max_pool1d(F.relu(conv(emb)), kernel_size=conv(emb).size(-1)).squeeze(-1) for conv in self.convs]
            feat = torch.cat(pooled, dim=-1)
            asp_logits = torch.stack([fc(feat) for fc in self.aspect_fc], dim=1)
            return asp_logits


    class GraphConvLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
            return self.dropout(F.relu(torch.bmm(norm_adj, self.linear(x))))


    class Paper9WSD_BiLSTM_GCN(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", bert_dim=768, b_units=128, gcn_dim=128, dropout_rate=0.5, num_aspects=6, num_sent_classes=4):
            super().__init__()
            self.num_aspects = num_aspects
            self.bert = AutoModel.from_pretrained(backbone_name)
            self.bilstm = nn.LSTM(input_size=bert_dim, hidden_size=b_units, num_layers=1, batch_first=True, bidirectional=True)
            self.bilstm_dropout = nn.Dropout(dropout_rate)
            self.attn_queries = nn.Parameter(torch.randn(num_aspects, b_units * 2) * 0.02)
            self.gcn1 = GraphConvLayer(bert_dim, gcn_dim, dropout_rate)
            self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim, dropout_rate)

            combined_dim = b_units * 2 + gcn_dim
            self.fusion_gate = nn.Parameter(torch.tensor(0.5))
            self.aspect_classifiers = nn.ModuleList([nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(combined_dim, num_sent_classes)) for _ in range(num_aspects)])

        def forward(self, input_ids, attention_mask):
            bsz, seq_len = attention_mask.size()
            if input_ids.is_cuda:
                self.bilstm.flatten_parameters()
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_emb = bert_out.last_hidden_state
            lstm_out = self.bilstm_dropout(self.bilstm(seq_emb)[0])

            idx = torch.arange(seq_len, device=attention_mask.device)
            local_adj = ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= 2).float().unsqueeze(0).expand(bsz, -1, -1)
            adj = local_adj * (attention_mask.unsqueeze(1).float() * attention_mask.unsqueeze(2).float())
            adj[:, 0, :] = attention_mask.float()
            adj[:, :, 0] = attention_mask.float()
            inv_sqrt_deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0).pow(-0.5)
            norm_adj = inv_sqrt_deg * adj * inv_sqrt_deg.transpose(1, 2)

            gcn_h = self.gcn2(self.gcn1(seq_emb, norm_adj), norm_adj)
            pad_mask = (attention_mask == 0).unsqueeze(1)
            attn_scores = (torch.einsum("bld,ad->bal", lstm_out, self.attn_queries) / math.sqrt(lstm_out.size(-1))).masked_fill(pad_mask, -1e4)
            attn_weights = F.softmax(attn_scores, dim=-1)
            rationale_probs = torch.sigmoid(attn_scores) * attention_mask.unsqueeze(1).float()

            lstm_Side = torch.einsum("bal,bld->bad", attn_weights, lstm_out)
            gcn_Side = torch.einsum("bal,bld->bad", attn_weights, gcn_h)
            w = torch.sigmoid(self.fusion_gate)
            combined = torch.cat([w * lstm_Side, (1.0 - w) * gcn_Side], dim=-1)

            asp_logits = torch.stack([self.aspect_classifiers[a](combined[:, a, :]) for a in range(self.num_aspects)], dim=1)
            return {
                "aspect_logits": asp_logits,
                "rationale_probs": rationale_probs
            }

        @torch.no_grad()
        def extract_evidence_span(self, aspect_idx: int, review_text: str, outputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, bert_tok) -> str:
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len].cpu().tolist()
            scores = rat_prob[:valid_len].cpu().numpy()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text.strip()) if s.strip()]
            if len(sentences) > 1:
                scored_sents = []
                for s in sentences:
                    s_ids = set(bert_tok.encode(s, add_special_tokens=False))
                    hit_scores = [scores[i] for i in range(1, valid_len - 1) if tok_ids[i] in s_ids]
                    scored_sents.append((float(np.mean(hit_scores)) if hit_scores else 0.0, s))
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                return scored_sents[0][1]
            return review_text.strip()


def get_shared_bert_tokenizer():
    global _CACHED_BERT_TOK
    if _CACHED_BERT_TOK is None:
        _CACHED_BERT_TOK = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _CACHED_BERT_TOK


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    cache_key = str(model_path.resolve())
    if cache_key in LOADED_MODEL_CACHE:
        return LOADED_MODEL_CACHE[cache_key]

    while len(LOADED_MODEL_CACHE) >= MAX_CACHED_MODELS:
        oldest_key = next(iter(LOADED_MODEL_CACHE))
        evicted = LOADED_MODEL_CACHE.pop(oldest_key, None)
        if evicted and "model" in evicted:
            del evicted["model"]
        _release_device_memory()

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"File '{model_path.name}' không chứa model_state_dict hợp lệ.")

    raw_sd = dict(ckpt["model_state_dict"])
    state_keys = set(raw_sd.keys())
    arch = ckpt.get("architecture", "")
    if not arch:
        if "gcn1.linear.weight" in state_keys:
            arch = "paper9_wsd_bilstm_gcn"
        elif "svm_weights" in state_keys:
            arch = "paper4_bm25_tfigm_svm_cnn"
        elif "token_scorer.weight" in state_keys:
            arch = "paper1_bert"
        else:
            arch = "cram_absa"

    tok = get_shared_bert_tokenizer() if arch != "paper4_bm25_tfigm_svm_cnn" else None
    if arch == "paper1_bert":
        net = Paper1BERTModel()
        # Remap legacy Paper 1 keys (aspect_hidden_layers.{i}.0.* + aspect_heads.{i}.weight/bias)
        for i in range(6):
            hk_w = f"aspect_hidden_layers.{i}.0.weight"
            hk_b = f"aspect_hidden_layers.{i}.0.bias"
            ak_w = f"aspect_heads.{i}.weight"
            ak_b = f"aspect_heads.{i}.bias"
            if hk_w in raw_sd and f"aspect_heads.{i}.0.weight" not in raw_sd:
                raw_sd[f"aspect_heads.{i}.0.weight"] = raw_sd[hk_w]
            if hk_b in raw_sd and f"aspect_heads.{i}.0.bias" not in raw_sd:
                raw_sd[f"aspect_heads.{i}.0.bias"] = raw_sd[hk_b]
            if ak_w in raw_sd and f"aspect_heads.{i}.3.weight" not in raw_sd:
                raw_sd[f"aspect_heads.{i}.3.weight"] = raw_sd[ak_w]
            if ak_b in raw_sd and f"aspect_heads.{i}.3.bias" not in raw_sd:
                raw_sd[f"aspect_heads.{i}.3.bias"] = raw_sd[ak_b]
    elif arch == "paper4_bm25_tfigm_svm_cnn":
        v_size = raw_sd["embedding.weight"].shape[0]
        net = Paper4MultiRegionCNN(vocab_size=v_size)
    elif arch == "paper9_wsd_bilstm_gcn":
        net = Paper9WSD_BiLSTM_GCN()
    else:
        has_cross = any(k.startswith("aspect_cross_attn.") for k in state_keys)
        in_feat = raw_sd["feature_proj.0.weight"].shape[1] if "feature_proj.0.weight" in raw_sd else 20
        corn_dim = raw_sd["corn_head.4.weight"].shape[0] if "corn_head.4.weight" in raw_sd else 2
        net = CRAM_ABSA_MultiAspectModel(num_features=in_feat, corn_out_dim=corn_dim, use_cross_attn=has_cross)
        arch = "cram_absa"

    model_sd = net.state_dict()
    compat_sd = {
        k: v for k, v in raw_sd.items()
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape
    }
    net.load_state_dict(compat_sd, strict=False)
    net = net.to(DEVICE)
    net.eval()

    bundle = {
        "arch": arch,
        "ckpt": ckpt,
        "model": net,
        "tokenizer": tok,
        "feat_mean": ckpt.get("feat_mean", None),
        "feat_std": ckpt.get("feat_std", None),
        "epoch": ckpt.get("epoch", "-")
    }
    LOADED_MODEL_CACHE[cache_key] = bundle
    return bundle


def prepare_cram_test_tensors(df_test, tokenizer, max_len: int = 256) -> Dict[str, Any]:
    global _CACHED_TEST_CRAM_TENSORS
    if _CACHED_TEST_CRAM_TENSORS is not None:
        return _CACHED_TEST_CRAM_TENSORS

    n_rows = len(df_test)
    raw_texts = [unicodedata.normalize("NFC", str(t)) for t in df_test["review_text"]]
    has_sent_col = "sentiment_id" in df_test.columns
    sent_col_vals = df_test["sentiment_id"].values if has_sent_col else np.full(n_rows, 2)
    anns_col = df_test["aspect_annotations"].tolist() if "aspect_annotations" in df_test.columns else [[] for _ in range(n_rows)]

    batch_enc = tokenizer(
        raw_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    all_input_ids = batch_enc["input_ids"]
    all_attention_masks = batch_enc["attention_mask"]
    all_offsets = batch_enc["offset_mapping"].numpy()

    if all(c in df_test.columns for c in FEATURE_COLS_20):
        features_raw_20 = df_test[FEATURE_COLS_20].values.astype(np.float32)
    else:
        features_raw_20 = np.stack([extract_20_features_for_new_review(t) for t in raw_texts], axis=0)

    all_aspect_labels = torch.zeros((n_rows, 6), dtype=torch.long)
    all_rationale_masks = torch.zeros((n_rows, 6, max_len), dtype=torch.float32)
    all_sentiment_ids = torch.zeros(n_rows, dtype=torch.long)

    for idx in range(n_rows):
        raw_text = raw_texts[idx]
        offsets = all_offsets[idx]
        s_id = int(sent_col_vals[idx]) if pd.notna(sent_col_vals[idx]) else 2
        all_sentiment_ids[idx] = max(0, min(2, s_id))

        anns = parse_annotations_safe(anns_col[idx])
        asp_sent_votes = defaultdict(list)
        asp_span_texts = defaultdict(list)
        valid_anns = []

        for ann in anns:
            asp = ann.get("aspect")
            sent = ann.get("sentiment", "Neutral")
            if asp not in ASPECT_ID_MAP or asp == "None":
                continue
            asp_idx = ASPECT_ID_MAP[asp] - 1
            s_char = int(ann.get("start_char", 0))
            e_char = int(ann.get("end_char", 0))
            span_txt = str(ann.get("span_text", "")).strip()
            asp_sent_votes[asp_idx].append(SENTIMENT_TO_ID.get(sent, 2))
            if span_txt:
                asp_span_texts[asp_idx].append(span_txt)
            valid_anns.append((asp, asp_idx, s_char, e_char))

        if valid_anns:
            word_spans = []
            for m in re.finditer(r"[a-zA-Z]+", raw_text):
                w_low = m.group(0).lower()
                word_spans.append((m.start(), m.end(), w_low, cached_lemmatize(w_low)))

            tok_aspect_kw = {asp: set() for asp in CANONICAL_ASPECTS}
            for t_i, (tok_s, tok_e) in enumerate(offsets):
                if tok_e <= tok_s:
                    continue
                sub_w = re.sub(r"[^a-z]", "", raw_text[tok_s:tok_e].lower())
                sub_lem = cached_lemmatize(sub_w) if sub_w else ""
                overlapping_words = [
                    (w_low, w_lem) for (ws, we, w_low, w_lem) in word_spans
                    if tok_s < we and tok_e > ws
                ]
                for asp in CANONICAL_ASPECTS:
                    kw_set = ASPECT_KEYWORDS[asp]
                    if (sub_w and (sub_w in kw_set or sub_lem in kw_set)) or any(
                        (wl in kw_set or wlem in kw_set) for wl, wlem in overlapping_words
                    ):
                        tok_aspect_kw[asp].add(t_i)

            has_multi_asp = len(asp_sent_votes) > 1
            tok_min_dist = np.full((6, max_len), 999, dtype=np.int32)
            rationale_masks = torch.zeros((6, max_len), dtype=torch.float32)

            for asp, asp_idx, s_char, e_char in valid_anns:
                span_toks = [
                    t_i for t_i, (tok_s, tok_e) in enumerate(offsets)
                    if tok_e > tok_s and tok_s < e_char and tok_e > s_char
                ]
                if not span_toks:
                    continue
                overlaps_other = has_multi_asp and any(
                    (o_idx != asp_idx and max(s_char, o_s) < min(e_char, o_e))
                    for _, o_idx, o_s, o_e in valid_anns
                )
                kw_toks = [t_i for t_i in span_toks if t_i in tok_aspect_kw[asp]]
                if (overlaps_other or has_multi_asp) and kw_toks:
                    for t_i in span_toks:
                        d = min(abs(t_i - k_i) for k_i in kw_toks)
                        if d <= 4:
                            rationale_masks[asp_idx, t_i] = 1.0
                            if d < tok_min_dist[asp_idx, t_i]:
                                tok_min_dist[asp_idx, t_i] = d
                else:
                    for t_i in span_toks:
                        rationale_masks[asp_idx, t_i] = 1.0
                        d = min((abs(t_i - k_i) for k_i in kw_toks), default=50)
                        if d < tok_min_dist[asp_idx, t_i]:
                            tok_min_dist[asp_idx, t_i] = d

            if has_multi_asp:
                overlap_tok_indices = torch.where(rationale_masks.sum(dim=0) > 1.0)[0].tolist()
                for t_i in overlap_tok_indices:
                    active_asps = [a_i for a_i in range(6) if rationale_masks[a_i, t_i] > 0.5]
                    best_a = min(active_asps, key=lambda a_i: tok_min_dist[a_i, t_i])
                    rationale_masks[:, t_i] = 0.0
                    rationale_masks[best_a, t_i] = 1.0

            all_rationale_masks[idx] = rationale_masks

        for asp_idx, votes in asp_sent_votes.items():
            maj_sent = Counter(votes).most_common(1)[0][0]
            joined_span = " ".join(asp_span_texts[asp_idx]).strip()
            cal_sent = compute_span_calibrated_sentiment(
                joined_span if joined_span else raw_text,
                maj_sent
            )
            all_aspect_labels[idx, asp_idx] = cal_sent

    _CACHED_TEST_CRAM_TENSORS = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "features_raw_20": features_raw_20,
        "aspect_sent_labels": all_aspect_labels,
        "rationale_masks": all_rationale_masks,
        "sentiment_ids": all_sentiment_ids
    }
    return _CACHED_TEST_CRAM_TENSORS


def prepare_paper1_or_9_test_tensors(df_test, tokenizer, use_wsd: bool = False, max_len: int = 256) -> Dict[str, Any]:
    global _CACHED_TEST_P1_TENSORS, _CACHED_TEST_P9_TENSORS
    if not use_wsd and _CACHED_TEST_P1_TENSORS is not None:
        return _CACHED_TEST_P1_TENSORS
    if use_wsd and _CACHED_TEST_P9_TENSORS is not None:
        return _CACHED_TEST_P9_TENSORS

    n_rows = len(df_test)
    raw_texts = [unicodedata.normalize("NFC", str(t)) for t in df_test["review_text"]]
    anns_col = df_test["aspect_annotations"].tolist() if "aspect_annotations" in df_test.columns else [[] for _ in range(n_rows)]

    if use_wsd:
        proc_texts = [enrich_text_with_wordnet_wsd(t) for t in raw_texts]
        batch_enc = tokenizer(
            proc_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        all_input_ids = batch_enc["input_ids"]
        all_attention_masks = batch_enc["attention_mask"]
        all_offsets = batch_enc["offset_mapping"].numpy()
    else:
        proc_texts = [paper1_clean_text(t) for t in raw_texts]
        batch_enc = tokenizer(
            proc_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        all_input_ids = batch_enc["input_ids"]
        all_attention_masks = batch_enc["attention_mask"]
        all_offsets = None

    all_aspect_labels = torch.zeros((n_rows, 6), dtype=torch.long)
    all_rationale_masks = torch.zeros((n_rows, 6, max_len), dtype=torch.float32)

    for idx in range(n_rows):
        anns = parse_annotations_safe(anns_col[idx])
        votes = defaultdict(list)
        if use_wsd:
            offsets = all_offsets[idx]
            raw_len = len(raw_texts[idx])
            for ann in anns:
                asp = ann.get("aspect")
                sent = ann.get("sentiment", "Neutral")
                if asp in ASPECT_ID_MAP and asp != "None":
                    a_idx = ASPECT_ID_MAP[asp] - 1
                    votes[a_idx].append(SENTIMENT_TO_ID.get(sent, 2))
                    s_c = int(ann.get("start_char", 0))
                    e_c = int(ann.get("end_char", 0))
                    for t_i, (ts, te) in enumerate(offsets):
                        if te > ts and ts < e_c and te > s_c and ts < raw_len:
                            all_rationale_masks[idx, a_idx, t_i] = 1.0
            for a_idx, v_list in votes.items():
                all_aspect_labels[idx, a_idx] = Counter(v_list).most_common(1)[0][0]
        else:
            tok_strs = tokenizer.convert_ids_to_tokens(all_input_ids[idx].tolist())
            span_toks_map = defaultdict(set)
            for ann in anns:
                asp = ann.get("aspect")
                sent = ann.get("sentiment", "Neutral")
                if asp in ASPECT_ID_MAP and asp != "None":
                    a_idx = ASPECT_ID_MAP[asp] - 1
                    votes[a_idx].append(SENTIMENT_TO_ID.get(sent, 2))
                    raw_sp = str(ann.get("span_text", "")).strip()
                    span_toks_map[a_idx].update(paper1_clean_text(raw_sp).split())
            for a_idx, v_list in votes.items():
                all_aspect_labels[idx, a_idx] = Counter(v_list).most_common(1)[0][0]
                target_words = span_toks_map[a_idx]
                if target_words:
                    for t_i, t_str in enumerate(tok_strs):
                        clean_sub = t_str.replace("##", "")
                        if clean_sub in target_words and all_attention_masks[idx, t_i].item() == 1:
                            all_rationale_masks[idx, a_idx, t_i] = 1.0

    result = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "aspect_sent_labels": all_aspect_labels,
        "rationale_masks": all_rationale_masks
    }
    if use_wsd:
        _CACHED_TEST_P9_TENSORS = result
    else:
        _CACHED_TEST_P1_TENSORS = result
    return result


def run_test_split_benchmark(
    model_key: str,
    progress_cb=None,
    fixed_model_path: Optional[Path] = None,
    fixed_display_id: Optional[str] = None
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch chưa được cài đặt trong môi trường hiện tại.")

    model_path = fixed_model_path if fixed_model_path is not None else resolve_model_file(model_key)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_key}")

    display_id = fixed_display_id if fixed_display_id else compute_model_display_id(model_path, get_model_dir())
    if progress_cb:
        progress_cb(2.0, f"Đang tải mô hình {display_id} và tập test (10% = 999 mẫu)...")

    with MODEL_EXEC_LOCK:
        bundle = load_model_bundle(model_path)
        arch = bundle["arch"]
        ckpt = bundle["ckpt"]
        net = bundle["model"]
        tok = bundle["tokenizer"]
        f_mean = bundle["feat_mean"]
        f_std = bundle["feat_std"]
        epoch_val = bundle["epoch"]

        df_test = load_test_dataframe()
        n_samples = len(df_test)
        batch_size = 32

        if arch == "cram_absa":
            if progress_cb:
                progress_cb(8.0, f"Đang chuẩn bị dữ liệu tập test ({n_samples} mẫu) cho {display_id}...")
            t_data = prepare_cram_test_tensors(df_test, tok)
            raw_20 = t_data["features_raw_20"]
            in_features = net.feature_proj[0].in_features
            if in_features == 20:
                raw_feat = raw_20
            else:
                raw_feat = raw_20[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19]]
                if raw_feat.shape[1] < in_features:
                    raw_feat = np.pad(raw_feat, ((0, 0), (0, in_features - raw_feat.shape[1])))
                elif raw_feat.shape[1] > in_features:
                    raw_feat = raw_feat[:, :in_features]

            if f_mean is not None and f_std is not None:
                f_mean_arr = np.asarray(f_mean, dtype=np.float32)
                f_std_arr = np.asarray(f_std, dtype=np.float32)
                c_mean = f_mean_arr[:in_features] if len(f_mean_arr) >= in_features else np.pad(f_mean_arr, (0, in_features - len(f_mean_arr)))
                c_std = f_std_arr[:in_features] if len(f_std_arr) >= in_features else np.pad(f_std_arr, (0, in_features - len(f_std_arr)), constant_values=1.0)
                norm_feat = (raw_feat - c_mean) / np.where(c_std < 1e-6, 1.0, c_std)
            else:
                norm_feat = raw_feat

            dense_all = torch.tensor(norm_feat, dtype=torch.float32)
            total_batches = max(1, math.ceil(n_samples / batch_size))

            total_loss = 0.0
            all_pol_preds, all_pol_trues = [], []
            all_asp_preds, all_asp_trues = [], []
            rat_inter, rat_pred_sum, rat_true_sum = 0.0, 0.0, 0.0
            overlap_rates = []
            pos_w = torch.tensor([2.5], device=DEVICE)

            with torch.no_grad():
                for b_idx in range(total_batches):
                    s_i = b_idx * batch_size
                    e_i = min(n_samples, s_i + batch_size)
                    input_ids = t_data["input_ids"][s_i:e_i].to(DEVICE)
                    attention_mask = t_data["attention_mask"][s_i:e_i].to(DEVICE)
                    dense_features = dense_all[s_i:e_i].to(DEVICE)
                    asp_y = t_data["aspect_sent_labels"][s_i:e_i].to(DEVICE)
                    true_mask = t_data["rationale_masks"][s_i:e_i].to(DEVICE)
                    sent_ids = t_data["sentiment_ids"][s_i:e_i]

                    outputs = net(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_features)

                    ce_loss = F.cross_entropy(outputs["aspect_logits"].view(-1, 4), asp_y.view(-1)).item()
                    valid_mask = attention_mask.unsqueeze(1).expand_as(true_mask).float()
                    bce_raw = F.binary_cross_entropy_with_logits(
                        outputs["rationale_logits"], true_mask, pos_weight=pos_w, reduction="none"
                    )
                    loss_rat = ((bce_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item()
                    total_loss += (ce_loss + 0.15 * loss_rat)

                    if net.corn_out_dim == 2:
                        rev_pol_preds = decode_corn_predictions(outputs["corn_logits"]).cpu().numpy()
                        rev_pol_trues = sent_ids.numpy()
                        all_pol_preds.append(rev_pol_preds)
                        all_pol_trues.append(rev_pol_trues)

                    asp_true_np = asp_y.cpu().numpy()
                    asp_pol_pred_np = outputs["aspect_logits"][:, :, 1:4].argmax(dim=-1).cpu().numpy()
                    active_mask = asp_true_np > 0
                    if np.any(active_mask):
                        all_pol_trues.append(asp_true_np[active_mask] - 1)
                        all_pol_preds.append(asp_pol_pred_np[active_mask])

                    asp_preds = outputs["aspect_logits"].argmax(dim=-1)
                    all_asp_preds.append(asp_preds.cpu().numpy())
                    all_asp_trues.append(asp_true_np)

                    active_asp_gate = (asp_preds > 0).unsqueeze(-1).float()
                    gated_probs = outputs["rationale_probs"] * active_asp_gate
                    max_prob_per_tok = gated_probs.max(dim=1, keepdim=True).values
                    pred_mask = ((gated_probs >= 0.5) & (gated_probs >= max_prob_per_tok * 0.92)).float()

                    rat_inter += (pred_mask * true_mask).sum().item()
                    rat_pred_sum += pred_mask.sum().item()
                    rat_true_sum += true_mask.sum().item()

                    active_asp_per_tok = pred_mask.sum(dim=1)
                    leaked_toks = (active_asp_per_tok > 1.0).float().sum().item()
                    active_toks = (active_asp_per_tok >= 1.0).float().sum().item()
                    overlap_rates.append(leaked_toks / max(active_toks, 1.0))

                    if progress_cb:
                        pct = 10.0 + 88.0 * ((b_idx + 1) / total_batches)
                        progress_cb(pct, f"Đang chạy tập test {display_id}: {e_i}/{n_samples} mẫu ({b_idx + 1}/{total_batches} batch)...")

            if all_pol_trues and all_pol_preds:
                y_true_pol = np.concatenate(all_pol_trues)
                y_pred_pol = np.concatenate(all_pol_preds)
                acc_at_0 = float(accuracy_score(y_true_pol, y_pred_pol) * 100.0)
                per_cls_mae = [
                    np.abs(y_true_pol[y_true_pol == c] - y_pred_pol[y_true_pol == c]).mean()
                    for c in range(3) if np.any(y_true_pol == c)
                ]
                macro_mae = float(np.mean(per_cls_mae)) if per_cls_mae else 0.0
            else:
                acc_at_0 = 0.0
                macro_mae = 0.0

            Y_asp_true = np.vstack(all_asp_trues)
            Y_asp_pred = np.vstack(all_asp_preds)
            aspect_f1s = [
                f1_score(Y_asp_true[:, a], Y_asp_pred[:, a], average="macro", zero_division=0) * 100.0
                for a in range(6)
            ]
            macro_aspect_f1 = float(np.mean(aspect_f1s))
            rationale_f1 = (2.0 * rat_inter / max(rat_pred_sum + rat_true_sum, 1.0)) * 100.0
            leakage_rate = float(np.mean(overlap_rates)) * 100.0 if overlap_rates else 0.0
            _release_device_memory()

            return {
                "model": display_id,
                "epoch": epoch_val,
                "val_loss": round(total_loss / total_batches, 4),
                "acc_at_0": round(acc_at_0, 2),
                "aspect_macro_f1": round(macro_aspect_f1, 2),
                "macro_mae": round(macro_mae, 4),
                "rationale_f1": round(rationale_f1, 2),
                "leakage_rate": round(leakage_rate, 2)
            }

        elif arch in ("paper1_bert", "paper9_wsd_bilstm_gcn"):
            use_wsd = (arch == "paper9_wsd_bilstm_gcn")
            if progress_cb:
                progress_cb(8.0, f"Đang tiền xử lý tập test ({n_samples} mẫu) cho {display_id}...")
            t_data = prepare_paper1_or_9_test_tensors(df_test, tok, use_wsd=use_wsd)
            total_batches = max(1, math.ceil(n_samples / batch_size))

            total_loss = 0.0
            asp_preds, asp_trues, asp_pol_preds = [], [], []
            rat_inter, rat_pred_sum, rat_true_sum = 0.0, 0.0, 0.0
            overlap_rates = []

            with torch.no_grad():
                for b_idx in range(total_batches):
                    s_i = b_idx * batch_size
                    e_i = min(n_samples, s_i + batch_size)
                    input_ids = t_data["input_ids"][s_i:e_i].to(DEVICE)
                    attention_mask = t_data["attention_mask"][s_i:e_i].to(DEVICE)
                    asp_y = t_data["aspect_sent_labels"][s_i:e_i].to(DEVICE)
                    rat_mask = t_data["rationale_masks"][s_i:e_i].to(DEVICE)

                    outputs = net(input_ids=input_ids, attention_mask=attention_mask)
                    loss = (
                        F.cross_entropy(outputs["aspect_logits"].view(-1, 4), asp_y.view(-1))
                        + 0.25 * F.binary_cross_entropy(outputs["rationale_probs"].clamp(1e-6, 1.0 - 1e-6), rat_mask)
                    )
                    total_loss += loss.item()

                    asp_preds.append(outputs["aspect_logits"].argmax(dim=-1).cpu().numpy())
                    asp_pol_preds.append(outputs["aspect_logits"][:, :, 1:4].argmax(dim=-1).cpu().numpy())
                    asp_trues.append(asp_y.cpu().numpy())

                    pred_mask = (outputs["rationale_probs"] >= 0.5).float()
                    rat_inter += (pred_mask * rat_mask).sum().item()
                    rat_pred_sum += pred_mask.sum().item()
                    rat_true_sum += rat_mask.sum().item()

                    active_per_tok = pred_mask.sum(dim=1)
                    leaked = (active_per_tok > 1.0).float().sum().item()
                    active = (active_per_tok >= 1.0).float().sum().item()
                    overlap_rates.append(leaked / max(active, 1.0))

                    if progress_cb:
                        pct = 10.0 + 88.0 * ((b_idx + 1) / total_batches)
                        progress_cb(pct, f"Đang chạy tập test {display_id}: {e_i}/{n_samples} mẫu ({b_idx + 1}/{total_batches} batch)...")

            Y_true_a = np.vstack(asp_trues)
            Y_pred_a = np.vstack(asp_preds)
            Y_pred_pol_all = np.vstack(asp_pol_preds)

            active_mask = Y_true_a > 0
            if np.any(active_mask):
                y_true_pol = Y_true_a[active_mask] - 1
                y_pred_pol = Y_pred_pol_all[active_mask]
                acc_at_0 = float(accuracy_score(y_true_pol, y_pred_pol) * 100.0)
                per_cls_mae = [
                    np.abs(y_true_pol[y_true_pol == c] - y_pred_pol[y_true_pol == c]).mean()
                    for c in range(3) if np.any(y_true_pol == c)
                ]
                macro_mae = float(np.mean(per_cls_mae)) if per_cls_mae else 0.0
            else:
                acc_at_0 = 0.0
                macro_mae = 0.0

            asp_f1s = [f1_score(Y_true_a[:, a], Y_pred_a[:, a], average="macro", zero_division=0) * 100.0 for a in range(6)]
            macro_asp_f1 = float(np.mean(asp_f1s))
            rationale_f1 = (2.0 * rat_inter / max(rat_pred_sum + rat_true_sum, 1.0)) * 100.0
            leakage_val = float(np.mean(overlap_rates)) * 100.0 if overlap_rates else 0.0
            _release_device_memory()

            return {
                "model": display_id,
                "epoch": epoch_val,
                "val_loss": round(total_loss / total_batches, 4),
                "acc_at_0": round(acc_at_0, 2),
                "aspect_macro_f1": round(macro_asp_f1, 2),
                "macro_mae": round(macro_mae, 4),
                "rationale_f1": round(rationale_f1, 2),
                "leakage_rate": round(leakage_val, 2)
            }

        else:
            # Paper 4: Word2Vec KoHA + BM25 + TF-IGM SVM & Multi-Region CNN
            word2idx = ckpt.get("word2idx", {})
            vocab_size = net.embedding.num_embeddings
            igm_weights = ckpt.get("igm_weights", np.zeros(vocab_size, dtype=np.float32))
            koha_exp = {k: set(v) for k, v in ckpt.get("koha_expanded", {}).items()}
            for asp in CANONICAL_ASPECTS:
                if asp not in koha_exp or not koha_exp[asp]:
                    koha_exp[asp] = set(ASPECT_KEYWORDS.get(asp, set()))
            N_sents = max(int(ckpt.get("N_sents", 10000)), 1)
            avg_sent_len = float(ckpt.get("avg_sent_len", 12.0))
            doc_freq_sent = ckpt.get("doc_freq_sent", {})

            def compute_bm25(query_terms: set, sent_tokens: List[str]) -> float:
                if not sent_tokens or not query_terms:
                    return 0.0
                s_len = len(sent_tokens)
                tf_map = Counter(sent_tokens)
                score = 0.0
                for q in query_terms:
                    if q in tf_map:
                        f_qs = tf_map[q]
                        df_q = doc_freq_sent.get(q, 0)
                        idf_q = math.log((N_sents - df_q + 0.5) / (df_q + 0.5) + 1.0)
                        num = f_qs * (1.5 + 1.0)
                        den = f_qs + 1.5 * (1.0 - 0.75 + 0.75 * (s_len / max(avg_sent_len, 1.0)))
                        score += idf_q * (num / max(den, 1e-6))
                return score

            svm_w_np = net.svm_weights.detach().cpu().numpy()
            svm_b_np = net.svm_biases.detach().cpu().numpy()

            Y_pred_asp_list, Y_pred_pol_list, blended_logits_list = [], [], []
            Y_true_asp = np.zeros((n_samples, 6), dtype=int)
            rat_inter, rat_pred_sum, rat_true_sum = 0.0, 0.0, 0.0
            overlap_rates = []

            p4_batch = 64
            total_batches = max(1, math.ceil(n_samples / p4_batch))

            with torch.no_grad():
                for b_idx in range(total_batches):
                    s_i = b_idx * p4_batch
                    e_i = min(n_samples, s_i + p4_batch)
                    batch_df = df_test.iloc[s_i:e_i]

                    seq_batch = []
                    tfigm_batch = np.zeros((e_i - s_i, vocab_size), dtype=np.float32)

                    for local_i, (_, row) in enumerate(batch_df.iterrows()):
                        global_i = s_i + local_i
                        txt = str(row["review_text"])
                        toks = paper4_preprocess_tokens(txt)
                        ids = [word2idx[t] for t in toks if t in word2idx][:128]
                        if len(ids) < 128:
                            ids = ids + [0] * (128 - len(ids))
                        seq_batch.append(ids)

                        counts = Counter(toks)
                        for t, freq in counts.items():
                            if t in word2idx:
                                vid = word2idx[t]
                                tfigm_batch[local_i, vid] = math.log(1.0 + freq) * (1.0 + 7.0 * float(igm_weights[vid]))
                        v_norm = np.linalg.norm(tfigm_batch[local_i])
                        if v_norm > 0:
                            tfigm_batch[local_i] /= v_norm

                        true_toks_per_asp = defaultdict(set)
                        votes = defaultdict(list)
                        for ann in parse_annotations_safe(row.get("aspect_annotations", [])):
                            asp = ann.get("aspect")
                            if asp in ASPECT_ID_MAP and asp != "None":
                                a_i = ASPECT_ID_MAP[asp] - 1
                                votes[a_i].append(SENTIMENT_TO_ID.get(ann.get("sentiment", "Neutral"), 2))
                                true_toks_per_asp[a_i].update(paper4_preprocess_tokens(ann.get("span_text", "")))
                        for a_i, v_l in votes.items():
                            Y_true_asp[global_i, a_i] = Counter(v_l).most_common(1)[0][0]

                        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
                        if not raw_sents and txt.strip():
                            raw_sents = [txt.strip()]
                        clusters = {asp: [] for asp in CANONICAL_ASPECTS}
                        for s in raw_sents:
                            s_toks = paper4_preprocess_tokens(s)
                            for asp in CANONICAL_ASPECTS:
                                sc = compute_bm25(koha_exp[asp], s_toks)
                                if sc > 0.0:
                                    clusters[asp].append((sc, s))

                        tok_aspect_count = Counter()
                        for a_i, asp in enumerate(CANONICAL_ASPECTS):
                            sents = [s for _, s in sorted(clusters[asp], key=lambda x: x[0], reverse=True)]
                            ptoks = set(paper4_preprocess_tokens(" ".join(sents[:1]))) if sents else set()
                            for t in ptoks:
                                tok_aspect_count[t] += 1
                            t_set = true_toks_per_asp[a_i]
                            rat_inter += len(ptoks & t_set)
                            rat_pred_sum += len(ptoks)
                            rat_true_sum += len(t_set)

                        if tok_aspect_count:
                            leaked = sum(1 for c in tok_aspect_count.values() if c > 1)
                            overlap_rates.append(leaked / len(tok_aspect_count))

                    bx_seq = torch.tensor(seq_batch, dtype=torch.long, device=DEVICE)
                    cnn_asp_np = net.forward_cnn(bx_seq).cpu().numpy()
                    svm_asp_np = np.stack([tfigm_batch.dot(svm_w_np[a].T) + svm_b_np[a] for a in range(6)], axis=1)
                    blended_asp = 0.7 * svm_asp_np + 0.3 * cnn_asp_np

                    blended_logits_list.append(blended_asp)
                    Y_pred_asp_list.append(blended_asp.argmax(axis=-1))
                    Y_pred_pol_list.append(blended_asp[:, :, 1:4].argmax(axis=-1))

                    if progress_cb:
                        pct = 10.0 + 88.0 * ((b_idx + 1) / total_batches)
                        progress_cb(pct, f"Đang chạy tập test {display_id}: {e_i}/{n_samples} mẫu ({b_idx + 1}/{total_batches} batch)...")

            Y_pred_asp = np.vstack(Y_pred_asp_list)
            Y_pred_pol_all = np.vstack(Y_pred_pol_list)
            blended_all = np.vstack(blended_logits_list)

            val_loss = float(F.cross_entropy(
                torch.tensor(blended_all, dtype=torch.float32).view(-1, 4),
                torch.tensor(Y_true_asp, dtype=torch.long).view(-1)
            ).item())

            active_mask = Y_true_asp > 0
            if np.any(active_mask):
                y_true_pol = Y_true_asp[active_mask] - 1
                y_pred_pol = Y_pred_pol_all[active_mask]
                acc0 = float(accuracy_score(y_true_pol, y_pred_pol) * 100.0)
                per_cls_mae = [
                    np.abs(y_true_pol[y_true_pol == c] - y_pred_pol[y_true_pol == c]).mean()
                    for c in range(3) if np.any(y_true_pol == c)
                ]
                macro_mae = float(np.mean(per_cls_mae)) if per_cls_mae else 0.0
            else:
                acc0 = 0.0
                macro_mae = 0.0

            asp_f1s = [f1_score(Y_true_asp[:, a], Y_pred_asp[:, a], average="macro", zero_division=0) * 100.0 for a in range(6)]
            rationale_f1 = (2.0 * rat_inter / max(rat_pred_sum + rat_true_sum, 1.0)) * 100.0
            leakage_rate = float(np.mean(overlap_rates)) * 100.0 if overlap_rates else 0.0
            _release_device_memory()

            return {
                "model": display_id,
                "epoch": epoch_val,
                "val_loss": round(val_loss, 4),
                "acc_at_0": round(acc0, 2),
                "aspect_macro_f1": round(float(np.mean(asp_f1s)), 2),
                "macro_mae": round(macro_mae, 4),
                "rationale_f1": round(rationale_f1, 2),
                "leakage_rate": round(leakage_rate, 2)
            }


def _benchmark_worker(model_key: str, model_path: Path, display_id: str):
    def _update(pct: float, msg: str):
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["progress"] = round(float(pct), 1)
            BENCHMARK_STATE["message"] = msg

    try:
        res = run_test_split_benchmark(
            model_key,
            progress_cb=_update,
            fixed_model_path=model_path,
            fixed_display_id=display_id
        )
        history_key = model_path.resolve().as_posix()
        with BENCHMARK_LOCK:
            BENCHMARK_HISTORY[history_key] = res
            BENCHMARK_STATE["progress"] = 100.0
            BENCHMARK_STATE["message"] = f"Đã hoàn tất đánh giá tập test cho {res['model']}."
            BENCHMARK_STATE["result"] = res
            BENCHMARK_STATE["is_running"] = False
    except Exception as e:
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["error"] = str(e)
            BENCHMARK_STATE["message"] = f"Lỗi khi chạy Benchmark: {e}"
            BENCHMARK_STATE["is_running"] = False


def start_benchmark_async(model_key: str) -> Tuple[bool, str]:
    model_path = resolve_model_file(model_key)
    if model_path is None or not model_path.exists():
        return False, "Vui lòng bấm 'Chọn File Model' để chọn một file mô hình hợp lệ trước khi chạy Benchmark."

    display_id = compute_model_display_id(model_path, get_model_dir())
    with BENCHMARK_LOCK:
        if BENCHMARK_STATE["is_running"]:
            return False, "Đang có tiến trình Benchmark đang chạy, vui lòng đợi hoàn tất."
        BENCHMARK_STATE["is_running"] = True
        BENCHMARK_STATE["progress"] = 1.0
        BENCHMARK_STATE["message"] = f"Đang khởi tạo đánh giá tập test cho {display_id}..."
        BENCHMARK_STATE["model_id"] = display_id
        BENCHMARK_STATE["result"] = None
        BENCHMARK_STATE["error"] = None

    t = threading.Thread(target=_benchmark_worker, args=(str(model_path), model_path, display_id), daemon=True)
    t.start()
    return True, "OK"


def get_benchmark_data(selected_model_key: str) -> Dict[str, Any]:
    cur_dir = get_model_dir()
    mp = resolve_model_file(selected_model_key)
    selected_info = get_model_file_info(mp) if (mp is not None and mp.is_file()) else None
    selected_display_id = selected_info["display_id"] if selected_info else ""
    selected_abs_key = selected_info["abs_path"] if selected_info else ""

    with BENCHMARK_LOCK:
        all_rows = list(BENCHMARK_HISTORY.values())

    selected_metrics = None
    if selected_abs_key and selected_abs_key in BENCHMARK_HISTORY:
        r = BENCHMARK_HISTORY[selected_abs_key]
        selected_metrics = {
            "acc_at_0": r["acc_at_0"],
            "aspect_macro_f1": r["aspect_macro_f1"],
            "macro_mae": r["macro_mae"],
            "rationale_f1": r["rationale_f1"],
            "leakage_rate": r["leakage_rate"],
            "val_loss": r["val_loss"]
        }
    elif selected_display_id and selected_display_id in BENCHMARK_HISTORY:
        r = BENCHMARK_HISTORY[selected_display_id]
        selected_metrics = {
            "acc_at_0": r["acc_at_0"],
            "aspect_macro_f1": r["aspect_macro_f1"],
            "macro_mae": r["macro_mae"],
            "rationale_f1": r["rationale_f1"],
            "leakage_rate": r["leakage_rate"],
            "val_loss": r["val_loss"]
        }
    elif all_rows and not selected_display_id:
        r = all_rows[-1]
        selected_metrics = {
            "acc_at_0": r["acc_at_0"],
            "aspect_macro_f1": r["aspect_macro_f1"],
            "macro_mae": r["macro_mae"],
            "rationale_f1": r["rationale_f1"],
            "leakage_rate": r["leakage_rate"],
            "val_loss": r["val_loss"]
        }

    return {
        "has_model": selected_info is not None,
        "selected_model": selected_info["abs_path"] if selected_info else "",
        "selected_display_id": selected_display_id,
        "selected_model_info": selected_info,
        "model_dir": str(cur_dir),
        "folder_name": cur_dir.name,
        "message": "Chưa có mô hình nào chạy Benchmark. Hãy bấm 'Chọn File Model' rồi bấm 'Chạy Benchmark'." if not all_rows else "OK",
        "metrics": selected_metrics,
        "rows": all_rows
    }


def evaluate_new_review(review_text: str, model_key: str) -> Dict[str, Any]:
    raw_text = unicodedata.normalize("NFC", str(review_text).strip())
    if not raw_text:
        return {
            "ok": False,
            "error": "Vui lòng nhập nội dung review.",
            "formatted_text": ""
        }

    model_path = resolve_model_file(model_key)
    if model_path is None or not model_path.exists():
        return {
            "ok": False,
            "error": f"Không tìm thấy mô hình: {model_key}",
            "formatted_text": f"Không tìm thấy mô hình: {model_key}"
        }

    if not TORCH_AVAILABLE:
        return {
            "ok": False,
            "error": "Môi trường hiện tại chưa cài đặt PyTorch/Transformers để chạy file mô hình.",
            "formatted_text": "Môi trường hiện tại chưa cài đặt PyTorch/Transformers để chạy file mô hình."
        }

    try:
        with MODEL_EXEC_LOCK:
            bundle = load_model_bundle(model_path)
            arch = bundle["arch"]
            ckpt = bundle["ckpt"]
            net = bundle["model"]
            tok = bundle["tokenizer"]
            f_mean = bundle["feat_mean"]
            f_std = bundle["feat_std"]

            mentioned_aspects = []
            lines = []
            aspect_details = {}

            with torch.no_grad():
                if arch == "paper4_bm25_tfigm_svm_cnn":
                    word2idx = ckpt.get("word2idx", {})
                    igm_weights = ckpt.get("igm_weights", np.zeros(net.embedding.num_embeddings, dtype=np.float32))
                    koha_exp = {k: set(v) for k, v in ckpt.get("koha_expanded", {}).items()}
                    toks = paper4_preprocess_tokens(raw_text)
                    ids = [word2idx[t] for t in toks if t in word2idx][:128]
                    if len(ids) < 128:
                        ids = ids + [0] * (128 - len(ids))
                    seq_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)

                    vec = np.zeros(net.embedding.num_embeddings, dtype=np.float32)
                    for t, freq in Counter(toks).items():
                        if t in word2idx:
                            vid = word2idx[t]
                            vec[vid] = math.log(1.0 + freq) * (1.0 + 7.0 * float(igm_weights[vid]))
                    v_norm = np.linalg.norm(vec)
                    if v_norm > 0:
                        vec = vec / v_norm
                    v_igm = torch.tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                    cnn_asp_l = net.forward_cnn(seq_t)
                    svm_asp_l = torch.einsum("bd,acd->bac", v_igm, net.svm_weights) + net.svm_biases
                    blended_asp = (0.7 * svm_asp_l + 0.3 * cnn_asp_l).squeeze(0).cpu().numpy()

                    raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
                    if not raw_sents:
                        raw_sents = [raw_text]

                    for a_idx, asp in enumerate(CANONICAL_ASPECTS):
                        pred_cls = int(np.argmax(blended_asp[a_idx]))
                        if pred_cls != 0:
                            mentioned_aspects.append(asp)
                            q_set = koha_exp.get(asp, ASPECT_KEYWORDS.get(asp, set()))
                            scored_s = []
                            for s_str in raw_sents:
                                s_u = [cached_lemmatize(t) for t in re.findall(r"[a-z]+", s_str.lower()) if t not in RAW_STOPWORDS]
                                hits = sum(1 for w in s_u if w in q_set)
                                scored_s.append((hits, s_str))
                            scored_s.sort(key=lambda x: x[0], reverse=True)
                            ev_span = scored_s[0][1] if scored_s else raw_text
                            cal_sent = compute_span_calibrated_sentiment(ev_span, pred_cls)
                            sent_label = ID_TO_SENTIMENT.get(cal_sent, ID_TO_SENTIMENT[pred_cls])
                            aspect_details[asp] = f"{sent_label} - {ev_span}"
                        else:
                            aspect_details[asp] = ""
                else:
                    enc = tok(raw_text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
                    input_ids = enc["input_ids"].to(DEVICE)
                    attention_mask = enc["attention_mask"].to(DEVICE)

                    if arch == "cram_absa":
                        in_features = net.feature_proj[0].in_features
                        if in_features == 20:
                            raw_feat = extract_20_features_for_new_review(raw_text)
                        else:
                            raw_feat = extract_18_features_for_new_review(raw_text)
                            if len(raw_feat) < in_features:
                                raw_feat = np.pad(raw_feat, (0, in_features - len(raw_feat)))
                            elif len(raw_feat) > in_features:
                                raw_feat = raw_feat[:in_features]

                        if f_mean is not None and f_std is not None:
                            c_mean = f_mean[:in_features] if len(f_mean) >= in_features else np.pad(f_mean, (0, in_features - len(f_mean)))
                            c_std = f_std[:in_features] if len(f_std) >= in_features else np.pad(f_std, (0, in_features - len(f_std)), constant_values=1.0)
                            norm_feat = (raw_feat - c_mean) / np.where(c_std < 1e-6, 1.0, c_std)
                        else:
                            norm_feat = raw_feat

                        dense_tensor = torch.tensor(norm_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        outputs = net(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_tensor)
                    else:
                        outputs = net(input_ids=input_ids, attention_mask=attention_mask)

                    asp_probs = F.softmax(outputs["aspect_logits"].squeeze(0), dim=-1).cpu().numpy()
                    for a_idx, asp in enumerate(CANONICAL_ASPECTS):
                        pred_cls = int(np.argmax(asp_probs[a_idx]))
                        if pred_cls != 0:
                            mentioned_aspects.append(asp)
                            if arch == "cram_absa":
                                ev_span = net.generate_aspect_commentary(a_idx, outputs, input_ids, attention_mask, tok, raw_review_text=raw_text)
                            else:
                                ev_span = net.extract_evidence_span(a_idx, raw_text, outputs, input_ids, attention_mask, tok)
                            cal_sent = compute_span_calibrated_sentiment(ev_span, pred_cls)
                            sent_label = ID_TO_SENTIMENT.get(cal_sent, ID_TO_SENTIMENT[pred_cls])
                            aspect_details[asp] = f"{sent_label} - {ev_span}"
                        else:
                            aspect_details[asp] = ""

            aspect_str = ", ".join(mentioned_aspects) if mentioned_aspects else "None"
            lines.append(f"- Aspect: {aspect_str}")
            for asp in CANONICAL_ASPECTS:
                if aspect_details[asp]:
                    lines.append(f"- {asp}: {aspect_details[asp]}")
                else:
                    lines.append(f"- {asp}:")

            return {
                "ok": True,
                "aspect_line": aspect_str,
                "formatted_text": "\n".join(lines)
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "formatted_text": f"Lỗi khi chạy mô hình: {e}"
        }


def generate_csv_bytes() -> bytes:
    with BENCHMARK_LOCK:
        rows = list(BENCHMARK_HISTORY.values())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Model", "Epoch", "Test Loss", "Acc@0 (%)",
        "6-Aspect Macro-F1 (%)", "Macro-MAE", "Rationale F1 (%)", "Leakage (%)"
    ])
    for r in rows:
        writer.writerow([
            r["model"], r["epoch"], r["val_loss"], r["acc_at_0"],
            r["aspect_macro_f1"], r["macro_mae"], r["rationale_f1"], r["leakage_rate"]
        ])
    return buf.getvalue().encode("utf-8-sig")


APPLE_GUI_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CRAM-ABSA Benchmark</title>
  <style>
    :root {
      --primary: #0066cc;
      --primary-focus: #0071e3;
      --ink: #1d1d1f;
      --body-on-dark: #ffffff;
      --body-muted: #cccccc;
      --ink-muted-80: #333333;
      --ink-muted-48: #7a7a7a;
      --divider-soft: #f0f0f0;
      --hairline: #e0e0e0;
      --canvas: #ffffff;
      --canvas-parchment: #f5f5f7;
      --surface-pearl: #fafafc;
      --surface-tile-1: #272729;
      --surface-black: #000000;
      --rounded-md: 11px;
      --rounded-lg: 16px;
      --rounded-pill: 9999px;
      --font-display: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
      --font-text: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
      --font-mono: "SF Mono", "Menlo", "Monaco", monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    html, body {
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      font-family: var(--font-text);
      font-size: 16px;
      font-weight: 400;
      line-height: 1.47;
      letter-spacing: -0.3px;
      color: var(--ink);
      background-color: var(--canvas-parchment);
    }

    .workspace-shell {
      display: flex;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
    }

    .sidebar {
      width: 248px;
      height: 100vh;
      background-color: var(--canvas-parchment);
      border-right: 1px solid var(--hairline);
      padding: 22px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      flex-shrink: 0;
      overflow-y: auto;
    }

    .sidebar-brand {
      font-family: var(--font-display);
      font-size: 18px;
      font-weight: 700;
      color: var(--ink);
      padding: 0 12px 18px;
      border-bottom: 1px solid var(--hairline);
      margin-bottom: 16px;
      letter-spacing: -0.2px;
    }

    .sidebar-section-label {
      font-size: 11px;
      font-weight: 600;
      color: var(--ink-muted-48);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 0 12px 8px;
    }

    .nav-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      padding: 10px 14px;
      border-radius: var(--rounded-md);
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink-muted-80);
      font-family: var(--font-text);
      font-size: 14px;
      cursor: pointer;
      text-align: left;
      transition: background-color 0.15s ease, color 0.15s ease;
    }

    .nav-item:hover {
      background-color: rgba(0, 0, 0, 0.04);
    }

    .nav-item svg {
      width: 18px;
      height: 18px;
      color: var(--ink-muted-48);
      flex-shrink: 0;
    }

    .nav-item.active {
      background-color: var(--canvas);
      color: var(--primary);
      font-weight: 600;
      border-color: var(--hairline);
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    .nav-item.active svg {
      color: var(--primary);
    }

    .main-stage {
      flex: 1;
      height: 100vh;
      min-width: 0;
      padding: 20px 28px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow: hidden;
    }

    .control-bar {
      background: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      flex-shrink: 0;
    }

    .control-file-area {
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
      min-width: 0;
    }

    .file-path-box {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      background-color: var(--surface-pearl);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-md);
      padding: 8px 14px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--ink);
      overflow: hidden;
    }

    .file-path-text {
      white-space: nowrap;
      overflow-x: auto;
      scrollbar-width: none;
      color: var(--ink);
      font-weight: 500;
    }

    .file-path-text::-webkit-scrollbar {
      display: none;
    }

    .file-path-text.placeholder {
      font-family: var(--font-text);
      color: var(--ink-muted-48);
      font-weight: 400;
    }

    .file-size-badge {
      flex-shrink: 0;
      font-family: var(--font-text);
      font-size: 12px;
      font-weight: 600;
      color: var(--primary);
      background: rgba(0, 102, 204, 0.09);
      padding: 2px 8px;
      border-radius: var(--rounded-pill);
      display: none;
    }

    .control-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-text);
      font-size: 14px;
      font-weight: 600;
      padding: 9px 16px;
      border-radius: var(--rounded-pill);
      cursor: pointer;
      text-decoration: none;
      border: 1px solid transparent;
      white-space: nowrap;
      transition: opacity 0.15s ease, background-color 0.15s ease;
    }

    .btn:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .btn svg {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
    }

    .btn-primary {
      background-color: var(--primary);
      color: var(--body-on-dark);
    }

    .btn-outline {
      background-color: var(--canvas);
      color: var(--primary);
      border-color: var(--primary);
    }

    .progress-banner {
      display: none;
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 12px 18px;
      flex-direction: column;
      gap: 8px;
      flex-shrink: 0;
    }

    .progress-banner.active {
      display: flex;
    }

    .progress-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
    }

    .progress-track {
      width: 100%;
      height: 7px;
      background-color: var(--canvas-parchment);
      border-radius: var(--rounded-pill);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      background-color: var(--primary);
      transition: width 0.25s ease;
    }

    .view-panel {
      display: none;
      flex-direction: column;
      gap: 16px;
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding-right: 4px;
    }

    .view-panel.active {
      display: flex;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      flex-shrink: 0;
    }

    .kpi-card {
      background-color: var(--surface-tile-1);
      color: var(--body-on-dark);
      border-radius: var(--rounded-lg);
      padding: 15px 18px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .kpi-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--body-muted);
    }

    .kpi-value {
      font-family: var(--font-display);
      font-size: 24px;
      font-weight: 600;
    }

    .card-surface {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 20px 22px;
    }

    .card-surface.table-card {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .table-scroll-container {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overflow-x: auto;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 12px;
      flex-shrink: 0;
    }

    .apple-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    .apple-table th {
      position: sticky;
      top: 0;
      background-color: var(--canvas);
      text-align: left;
      font-size: 12px;
      font-weight: 600;
      color: var(--ink-muted-48);
      padding: 10px 12px;
      border-bottom: 1px solid var(--hairline);
      z-index: 1;
    }

    .apple-table td {
      padding: 12px;
      border-bottom: 1px solid var(--divider-soft);
    }

    .model-path-cell {
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
    }

    .apple-textarea {
      width: 100%;
      min-height: 115px;
      font-family: var(--font-text);
      font-size: 15px;
      line-height: 1.5;
      color: var(--ink);
      background-color: var(--surface-pearl);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-md);
      padding: 12px 14px;
      resize: vertical;
      outline: none;
    }

    .apple-textarea:focus {
      border-color: var(--primary-focus);
      background-color: var(--canvas);
    }

    .output-box {
      background-color: var(--surface-tile-1);
      color: var(--body-on-dark);
      border-radius: var(--rounded-lg);
      padding: 18px 20px;
      font-family: var(--font-mono);
      font-size: 13.5px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 190px;
    }

    .metrics-guide-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
    }

    .metric-guide-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .metric-guide-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .metric-guide-name {
      font-family: var(--font-display);
      font-size: 16px;
      font-weight: 700;
      color: var(--ink);
    }

    .metric-direction {
      font-size: 14px;
      font-weight: 700;
      padding: 2px 10px;
      border-radius: var(--rounded-pill);
      line-height: 1.2;
    }

    .metric-direction.up {
      background-color: rgba(29, 131, 72, 0.12);
      color: #196f3d;
    }

    .metric-direction.down {
      background-color: rgba(0, 102, 204, 0.12);
      color: var(--primary);
    }

    .metric-guide-desc {
      font-size: 14px;
      line-height: 1.55;
      color: var(--ink-muted-80);
    }

    .viz-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 12px 18px;
      flex-shrink: 0;
    }

    .viz-status-note {
      font-size: 13.5px;
      color: var(--ink-muted-80);
    }

    .viz-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }

    .viz-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
    }

    .viz-card.full-width {
      grid-column: 1 / -1;
    }

    .viz-card-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
    }

    .btn-sm {
      font-size: 12.5px;
      padding: 6px 12px;
    }

    .viz-canvas-wrap {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: #ffffff;
      border-radius: var(--rounded-md);
      overflow: hidden;
    }

    .viz-canvas {
      width: 100%;
      height: auto;
      display: block;
    }
  </style>
</head>
<body>
  <div class="workspace-shell">
    <aside class="sidebar">
      <div>
        <div class="sidebar-brand">CRAM-ABSA Benchmark</div>
        <div class="sidebar-section-label">Menu</div>
        <ul class="nav-list">
          <li>
            <button class="nav-item active" id="nav-bench" onclick="switchTab('bench')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
              </svg>
              <span>Bảng Đánh Giá</span>
            </button>
          </li>
          <li>
            <button class="nav-item" id="nav-viz" onclick="switchTab('viz')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
                <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
              </svg>
              <span>Visualization</span>
            </button>
          </li>
          <li>
            <button class="nav-item" id="nav-playground" onclick="switchTab('playground')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              <span>Chạy thử Model</span>
            </button>
          </li>
          <li>
            <button class="nav-item" id="nav-metrics" onclick="switchTab('metrics')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
              <span>Ý Nghĩa Các Chỉ Số</span>
            </button>
          </li>
        </ul>
      </div>
    </aside>

    <main class="main-stage">
      <div class="control-bar">
        <div class="control-file-area">
          <button class="btn btn-outline" onclick="browseModelFile()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            <span>Chọn File Model</span>
          </button>
          <div class="file-path-box" id="selectedFilePathBox" title="Chưa chọn file model">
            <span class="file-path-text placeholder" id="selectedFilePathText">Chưa chọn file model — Bấm 'Chọn File Model' để mở file (.pt, .pth, .bin, .safetensors, .ckpt)</span>
            <span class="file-size-badge" id="selectedFileSizeBadge"></span>
          </div>
        </div>
        <div class="control-actions">
          <button id="btnRunBenchmark" class="btn btn-primary" onclick="runBenchmarkOnTestSplit()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span id="btnRunBenchmarkText">Chạy Benchmark</span>
          </button>
          <button class="btn btn-outline" onclick="exportCSV()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Xuất CSV</span>
          </button>
        </div>
      </div>

      <div id="progressBanner" class="progress-banner">
        <div class="progress-top">
          <span id="progressMessage">Đang chạy tập test...</span>
          <span id="progressPercent">0%</span>
        </div>
        <div class="progress-track">
          <div id="progressFill" class="progress-fill"></div>
        </div>
      </div>

      <section class="view-panel active" id="panel-bench">
        <div class="kpi-grid">
          <div class="kpi-card">
            <span class="kpi-label">Acc@0</span>
            <span class="kpi-value" id="kpiAcc">-</span>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">6-Aspect F1</span>
            <span class="kpi-value" id="kpiF1">-</span>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Macro-MAE</span>
            <span class="kpi-value" id="kpiMae">-</span>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Rationale F1</span>
            <span class="kpi-value" id="kpiRat">-</span>
          </div>
          <div class="kpi-card">
            <span class="kpi-label">Leakage</span>
            <span class="kpi-value" id="kpiLeak">-</span>
          </div>
        </div>

        <div class="card-surface table-card">
          <div class="card-title">Bảng đánh giá</div>
          <div class="table-scroll-container">
            <table class="apple-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Epoch</th>
                  <th>Test Loss</th>
                  <th>Acc@0 (%)</th>
                  <th>6-Aspect F1 (%)</th>
                  <th>Macro-MAE</th>
                  <th>Rationale F1 (%)</th>
                  <th>Leakage (%)</th>
                </tr>
              </thead>
              <tbody id="benchmarkTableBody"></tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="view-panel" id="panel-viz">
        <div class="viz-toolbar">
          <span class="viz-status-note" id="vizStatusNote">Biểu đồ tự động cập nhật từ các mô hình đã chạy trong phiên Benchmark hiện tại (Acc@0, 6-Aspect F1, Rationale F1, Leakage, Macro-MAE).</span>
          <button class="btn btn-primary" onclick="exportAllVizImages()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Xuất Tất Cả Hình Ảnh</span>
          </button>
        </div>

        <div class="viz-grid">
          <div class="viz-card full-width">
            <div class="viz-card-actions">
              <button class="btn btn-outline btn-sm" onclick="saveSingleChart('vizCanvasCore', 'benchmark_core_metrics.png')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span>Lưu ảnh</span>
              </button>
            </div>
            <div class="viz-canvas-wrap">
              <canvas id="vizCanvasCore" class="viz-canvas" width="2000" height="720"></canvas>
            </div>
          </div>

          <div class="viz-card">
            <div class="viz-card-actions">
              <button class="btn btn-outline btn-sm" onclick="saveSingleChart('vizCanvasLeakage', 'benchmark_leakage.png')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span>Lưu ảnh</span>
              </button>
            </div>
            <div class="viz-canvas-wrap">
              <canvas id="vizCanvasLeakage" class="viz-canvas" width="1100" height="640"></canvas>
            </div>
          </div>

          <div class="viz-card">
            <div class="viz-card-actions">
              <button class="btn btn-outline btn-sm" onclick="saveSingleChart('vizCanvasMae', 'benchmark_macro_mae.png')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span>Lưu ảnh</span>
              </button>
            </div>
            <div class="viz-canvas-wrap">
              <canvas id="vizCanvasMae" class="viz-canvas" width="1100" height="640"></canvas>
            </div>
          </div>
        </div>
      </section>

      <section class="view-panel" id="panel-playground">
        <div class="card-surface">
          <div class="card-title">Nhập Review</div>
          <textarea id="reviewInput" class="apple-textarea" placeholder="Nhập review khách sạn bằng tiếng Anh..."></textarea>
          <div style="display:flex; gap:10px; margin-top:14px;">
            <button class="btn btn-primary" onclick="runModelPrediction()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              <span>Chạy thử</span>
            </button>
            <button class="btn btn-outline" onclick="copyStructuredOutput()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              <span id="copyBtnText">Sao chép</span>
            </button>
          </div>
        </div>

        <div class="card-surface">
          <div class="card-title">Output</div>
          <pre class="output-box" id="structuredOutputBox">- Aspect:
- Service:
- Facility:
- Experience:
- Amenity:
- Loyalty:
- Branding:</pre>
        </div>
      </section>

      <section class="view-panel" id="panel-metrics">
        <div class="card-surface" style="padding: 18px 22px;">
          <div class="card-title" style="margin-bottom: 6px;">Ý Nghĩa Các Chỉ Số Đánh Giá (Benchmark Metrics)</div>
          <p style="font-size: 14px; color: var(--ink-muted-80);">
            Toàn bộ các mô hình được đánh giá trực tiếp trên cùng <strong>tập kiểm thử độc lập (10% Test Split = 999 mẫu)</strong> chia sẵn theo tỷ lệ 80/10/10 từ bộ dữ liệu 9.990 đánh giá khách sạn.
          </p>
        </div>

        <div class="metrics-guide-grid">
          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">1. Acc@0 (%) — Độ chính xác tuyệt đối phân cực</span>
              <span class="metric-direction up">↑</span>
            </div>
            <p class="metric-guide-desc">
              Tỷ lệ phần trăm dự đoán đúng chính xác nhãn cảm xúc (<em>Negative / Neutral / Positive</em>, độ lệch bậc bằng 0) trên các khía cạnh được đề cập và phân cực tổng thể của bài đánh giá.
            </p>
          </div>

          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">2. 6-Aspect F1 (%) — Điểm F1 vĩ mô 6 khía cạnh</span>
              <span class="metric-direction up">↑</span>
            </div>
            <p class="metric-guide-desc">
              Trung bình cộng Macro-F1 trên 6 khía cạnh khách sạn (<em>Service, Facility, Experience, Amenity, Loyalty, Branding</em>), mỗi khía cạnh gồm 4 trạng thái (<em>Not Mentioned, Negative, Neutral, Positive</em>). Phản ánh khả năng vừa phát hiện khía cạnh vừa phân loại cảm xúc trước dữ liệu mất cân bằng.
            </p>
          </div>

          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">3. Macro-MAE — Sai số tuyệt đối trung bình vĩ mô</span>
              <span class="metric-direction down">↓</span>
            </div>
            <p class="metric-guide-desc">
              Sai số khoảng cách trung bình giữa nhãn cảm xúc dự đoán và thực tế trên thang thứ tự (<em>0: Negative, 1: Neutral, 2: Positive</em>), tính trung bình đều qua từng lớp cảm xúc để phạt nặng lỗi đảo cực (nhầm <em>Positive</em> thành <em>Negative</em> bị phạt 2 đơn vị).
            </p>
          </div>

          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">4. Rationale F1 (%) — Độ chuẩn xác trích xuất dẫn chứng</span>
              <span class="metric-direction up">↑</span>
            </div>
            <p class="metric-guide-desc">
              Điểm F1 mức từ (token-level overlap) đo mức độ trùng khớp giữa vùng từ khóa dẫn chứng (rationale span) do mô hình trích xuất và vùng dẫn chứng thực tế do con người gán nhãn cho từng khía cạnh.
            </p>
          </div>

          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">5. Leakage (%) — Tỷ lệ chồng lấn/rò rỉ dẫn chứng</span>
              <span class="metric-direction down">↓</span>
            </div>
            <p class="metric-guide-desc">
              Tỷ lệ phần trăm các token dẫn chứng bị mô hình gán trùng lặp cho từ 2 khía cạnh trở lên trong cùng một bài đánh giá. Chỉ số này càng thấp chứng tỏ cơ chế Orthogonal/Competitive Masking tách biệt dẫn chứng giữa các khía cạnh càng rõ ràng.
            </p>
          </div>

          <div class="metric-guide-card">
            <div class="metric-guide-header">
              <span class="metric-guide-name">6. Test Loss &amp; Epoch — Hàm mất mát &amp; Vòng huấn luyện</span>
              <span class="metric-direction down">↓</span>
            </div>
            <p class="metric-guide-desc">
              <strong>Test Loss</strong> là tổng hợp hàm mất mát phân loại khía cạnh (Cross-Entropy) và hàm mất mát dẫn chứng (Binary Cross-Entropy) trên tập Test 10%. <strong>Epoch</strong> cho biết vòng huấn luyện mà tại đó trọng số checkpoint đạt chất lượng tốt nhất.
            </p>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    let currentSelectedModel = "";
    let currentBenchmarkRows = [];
    let isExportingCSV = false;
    let pollTimer = null;

    function switchTab(tabId) {
      document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach(el => el.classList.remove("active"));
      document.getElementById("nav-" + tabId).classList.add("active");
      document.getElementById("panel-" + tabId).classList.add("active");
      if (tabId === "viz") {
        renderAllVisualizations(currentBenchmarkRows);
      }
    }

    function shortModelLabel(rawModelPath) {
      const s = String(rawModelPath || "");
      const parts = s.split("/").filter(Boolean);
      if (parts.length >= 2) {
        const file = parts[parts.length - 1];
        const folder = parts.find(p => p.toLowerCase().startsWith("model lần")) || parts[parts.length - 2];
        return folder + "/" + file;
      }
      return s;
    }

    function drawEmptyCanvasState(canvas, message) {
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#e5e5ea";
      ctx.lineWidth = 2;
      ctx.strokeRect(20, 20, W - 40, H - 40);
      ctx.fillStyle = "#7a7a7a";
      ctx.font = "500 26px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(message, W / 2, H / 2);
      ctx.restore();
    }

    function drawGroupedBarChartNoTitle(canvasId, rows, seriesList, yMax, yUnit, decimals) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      if (!rows || rows.length === 0) {
        drawEmptyCanvasState(canvas, "Chưa có dữ liệu phiên Benchmark — Hãy chạy ít nhất 1 mô hình");
        return;
      }

      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, W, H);

      const padTop = 76;
      const padBottom = 115;
      const padLeft = 115;
      const padRight = 50;
      const plotW = W - padLeft - padRight;
      const plotH = H - padTop - padBottom;

      // Legend at top (no chart title)
      const legendFont = "600 22px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif";
      ctx.font = legendFont;
      let totalLegendW = 0;
      const itemWidths = seriesList.map(s => {
        const w = 28 + 10 + ctx.measureText(s.label).width;
        totalLegendW += w;
        return w;
      });
      totalLegendW += Math.max(0, seriesList.length - 1) * 36;
      let curLegendX = (W - totalLegendW) / 2;
      const legendY = 34;

      seriesList.forEach((s, idx) => {
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.roundRect(curLegendX, legendY - 12, 26, 20, 5);
        ctx.fill();
        ctx.fillStyle = "#1d1d1f";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(s.label, curLegendX + 36, legendY - 2);
        curLegendX += itemWidths[idx] + 36;
      });

      // Y-axis gridlines & labels
      const steps = 5;
      ctx.font = "500 20px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif";
      for (let i = 0; i <= steps; i++) {
        const ratio = i / steps;
        const val = yMax * (1 - ratio);
        const y = padTop + ratio * plotH;

        ctx.strokeStyle = i === steps ? "#8e8e93" : "#e5e5ea";
        ctx.lineWidth = i === steps ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotW, y);
        ctx.stroke();

        ctx.fillStyle = "#515154";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        const tickStr = decimals <= 1 ? val.toFixed(decimals === 0 ? 0 : 1) : val.toFixed(2);
        ctx.fillText(tickStr + yUnit, padLeft - 16, y);
      }

      // Bars
      const nGroups = rows.length;
      const nSeries = seriesList.length;
      const groupStep = plotW / nGroups;
      const maxGroupW = Math.min(groupStep * 0.72, nSeries * 92);
      const barGap = nSeries > 1 ? 10 : 0;
      const barW = Math.max(22, (maxGroupW - barGap * (nSeries - 1)) / nSeries);

      rows.forEach((row, gIdx) => {
        const groupCenterX = padLeft + gIdx * groupStep + groupStep / 2;
        const actualGroupW = nSeries * barW + (nSeries - 1) * barGap;
        const startX = groupCenterX - actualGroupW / 2;

        seriesList.forEach((s, sIdx) => {
          const rawVal = Number(row[s.key] || 0);
          const clampedVal = Math.max(0, Math.min(yMax, rawVal));
          const barH = (clampedVal / yMax) * plotH;
          const bx = startX + sIdx * (barW + barGap);
          const by = padTop + plotH - barH;

          ctx.fillStyle = s.color;
          ctx.beginPath();
          ctx.roundRect(bx, by, barW, Math.max(barH, 3), [6, 6, 0, 0]);
          ctx.fill();

          // Value label above bar
          ctx.fillStyle = "#1d1d1f";
          ctx.font = "600 18px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(rawVal.toFixed(decimals) + yUnit, bx + barW / 2, by - 7);
        });

        // X-axis model label
        const label = shortModelLabel(row.model);
        ctx.fillStyle = "#1d1d1f";
        ctx.font = "600 20px 'SF Mono', 'Menlo', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(label, groupCenterX, padTop + plotH + 20);
      });

      ctx.restore();
    }

    function renderAllVisualizations(rows) {
      const list = rows || [];
      drawGroupedBarChartNoTitle(
        "vizCanvasCore",
        list,
        [
          { key: "acc_at_0", label: "Acc@0 (%) ↑", color: "#0066cc" },
          { key: "aspect_macro_f1", label: "6-Aspect F1 (%) ↑", color: "#196f3d" },
          { key: "rationale_f1", label: "Rationale F1 (%) ↑", color: "#d35400" }
        ],
        100,
        "%",
        2
      );

      const maxLeak = list.length > 0
        ? Math.max(20, Math.ceil(Math.max(...list.map(r => Number(r.leakage_rate || 0))) * 1.25 / 10) * 10)
        : 100;
      drawGroupedBarChartNoTitle(
        "vizCanvasLeakage",
        list,
        [
          { key: "leakage_rate", label: "Leakage (%) ↓", color: "#c0392b" }
        ],
        Math.min(100, maxLeak),
        "%",
        2
      );

      const maxMae = list.length > 0
        ? Math.max(0.5, Math.ceil(Math.max(...list.map(r => Number(r.macro_mae || 0))) * 1.25 * 10) / 10)
        : 1.0;
      drawGroupedBarChartNoTitle(
        "vizCanvasMae",
        list,
        [
          { key: "macro_mae", label: "Macro-MAE ↓", color: "#6c3483" }
        ],
        maxMae,
        "",
        4
      );
    }

    async function saveSingleChart(canvasId, defaultFilename) {
      if (!currentBenchmarkRows || currentBenchmarkRows.length === 0) {
        alert("Chưa có dữ liệu Benchmark trong phiên hiện tại. Vui lòng chạy Benchmark ít nhất 1 mô hình trước khi xuất ảnh.");
        return;
      }
      renderAllVisualizations(currentBenchmarkRows);
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const dataUrl = canvas.toDataURL("image/png");
      const res = await fetch("/api/save_viz_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          default_name: defaultFilename,
          data_url: dataUrl
        })
      });
      const data = await res.json();
      if (data.cancelled) return;
      if (!data.ok) {
        alert(data.error || "Không thể lưu hình ảnh.");
        return;
      }
      document.getElementById("vizStatusNote").textContent = "Đã lưu ảnh tại: " + data.saved_path;
    }

    async function exportAllVizImages() {
      if (!currentBenchmarkRows || currentBenchmarkRows.length === 0) {
        alert("Chưa có dữ liệu Benchmark trong phiên hiện tại. Vui lòng chạy Benchmark ít nhất 1 mô hình trước khi xuất ảnh.");
        return;
      }
      renderAllVisualizations(currentBenchmarkRows);
      const images = [
        {
          filename: "benchmark_core_metrics.png",
          data_url: document.getElementById("vizCanvasCore").toDataURL("image/png")
        },
        {
          filename: "benchmark_leakage.png",
          data_url: document.getElementById("vizCanvasLeakage").toDataURL("image/png")
        },
        {
          filename: "benchmark_macro_mae.png",
          data_url: document.getElementById("vizCanvasMae").toDataURL("image/png")
        }
      ];
      const res = await fetch("/api/save_viz_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images })
      });
      const data = await res.json();
      if (data.cancelled) return;
      if (!data.ok) {
        alert(data.error || "Không thể lưu các hình ảnh.");
        return;
      }
      document.getElementById("vizStatusNote").textContent = "Đã lưu " + (data.saved_files ? data.saved_files.length : 3) + " ảnh vào thư mục: " + data.saved_dir;
    }

    function updateSelectedFileUI(modelInfo) {
      const boxEl = document.getElementById("selectedFilePathBox");
      const textEl = document.getElementById("selectedFilePathText");
      const badgeEl = document.getElementById("selectedFileSizeBadge");

      if (!modelInfo || !modelInfo.abs_path) {
        currentSelectedModel = "";
        textEl.textContent = "Chưa chọn file model — Bấm 'Chọn File Model' để mở file (.pt, .pth, .bin, .safetensors, .ckpt)";
        textEl.classList.add("placeholder");
        boxEl.title = "Chưa chọn file model";
        badgeEl.style.display = "none";
        badgeEl.textContent = "";
      } else {
        currentSelectedModel = modelInfo.abs_path;
        textEl.textContent = modelInfo.abs_path;
        textEl.classList.remove("placeholder");
        boxEl.title = modelInfo.abs_path;
        if (modelInfo.size_label) {
          badgeEl.textContent = modelInfo.size_label;
          badgeEl.style.display = "inline-block";
        } else {
          badgeEl.style.display = "none";
        }
      }
    }

    async function loadInitialState() {
      const res = await fetch("/api/benchmark");
      const data = await res.json();
      if (data.selected_model_info) {
        updateSelectedFileUI(data.selected_model_info);
      }
      renderBenchmarkTableAndKPIs(data);
    }

    async function browseModelFile() {
      const res = await fetch("/api/browse_model_file", { method: "POST" });
      const data = await res.json();
      if (data.cancelled) {
        return;
      }
      if (!data.ok) {
        if (data.error) alert(data.error);
        return;
      }
      updateSelectedFileUI(data.model);
      await loadBenchmarkData();
    }

    async function runBenchmarkOnTestSplit() {
      if (!currentSelectedModel) {
        alert("Vui lòng bấm 'Chọn File Model' để chọn một file mô hình trước khi chạy Benchmark.");
        return;
      }
      const btn = document.getElementById("btnRunBenchmark");
      const btnTxt = document.getElementById("btnRunBenchmarkText");
      const banner = document.getElementById("progressBanner");
      btn.disabled = true;
      btnTxt.textContent = "Đang chạy...";
      banner.classList.add("active");
      document.getElementById("progressFill").style.width = "2%";
      document.getElementById("progressPercent").textContent = "2%";
      document.getElementById("progressMessage").textContent = "Đang khởi động đánh giá trên tập test (80/10/10)...";

      const res = await fetch("/api/benchmark/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: currentSelectedModel })
      });
      const data = await res.json();
      if (!data.ok) {
        btn.disabled = false;
        btnTxt.textContent = "Chạy Benchmark";
        banner.classList.remove("active");
        alert(data.error || "Không thể bắt đầu Benchmark.");
        return;
      }

      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(pollBenchmarkStatus, 400);
    }

    async function pollBenchmarkStatus() {
      const res = await fetch("/api/benchmark/status");
      const st = await res.json();
      const pct = Number(st.progress || 0);
      document.getElementById("progressFill").style.width = pct + "%";
      document.getElementById("progressPercent").textContent = pct.toFixed(1) + "%";
      document.getElementById("progressMessage").textContent = st.message || "";

      if (!st.is_running) {
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        const btn = document.getElementById("btnRunBenchmark");
        const btnTxt = document.getElementById("btnRunBenchmarkText");
        btn.disabled = false;
        btnTxt.textContent = "Chạy Benchmark";
        if (st.error) {
          alert(st.message || st.error);
        }
        await loadBenchmarkData();
        setTimeout(() => {
          document.getElementById("progressBanner").classList.remove("active");
        }, 1800);
      }
    }

    function renderBenchmarkTableAndKPIs(data) {
      const tbody = document.getElementById("benchmarkTableBody");
      tbody.innerHTML = "";

      if (data.metrics) {
        const m = data.metrics;
        document.getElementById("kpiAcc").textContent = Number(m.acc_at_0).toFixed(2) + "%";
        document.getElementById("kpiF1").textContent = Number(m.aspect_macro_f1).toFixed(2) + "%";
        document.getElementById("kpiMae").textContent = Number(m.macro_mae).toFixed(4);
        document.getElementById("kpiRat").textContent = Number(m.rationale_f1).toFixed(2) + "%";
        document.getElementById("kpiLeak").textContent = Number(m.leakage_rate).toFixed(2) + "%";
      } else {
        document.getElementById("kpiAcc").textContent = "-";
        document.getElementById("kpiF1").textContent = "-";
        document.getElementById("kpiMae").textContent = "-";
        document.getElementById("kpiRat").textContent = "-";
        document.getElementById("kpiLeak").textContent = "-";
      }

      const rows = data.rows || [];
      currentBenchmarkRows = rows;
      renderAllVisualizations(currentBenchmarkRows);
      if (rows.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="8" style="text-align:center; color:#7a7a7a;">${data.message || "Chưa có mô hình nào chạy Benchmark."}</td>`;
        tbody.appendChild(tr);
        return;
      }

      rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="model-path-cell">${r.model}</td>
          <td>${r.epoch}</td>
          <td>${Number(r.val_loss).toFixed(4)}</td>
          <td>${Number(r.acc_at_0).toFixed(2)}%</td>
          <td>${Number(r.aspect_macro_f1).toFixed(2)}%</td>
          <td>${Number(r.macro_mae).toFixed(4)}</td>
          <td>${Number(r.rationale_f1).toFixed(2)}%</td>
          <td>${Number(r.leakage_rate).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function loadBenchmarkData() {
      const res = await fetch("/api/benchmark?model=" + encodeURIComponent(currentSelectedModel || ""));
      const data = await res.json();
      renderBenchmarkTableAndKPIs(data);
    }

    function exportCSV() {
      if (isExportingCSV) return;
      isExportingCSV = true;
      const a = document.createElement("a");
      a.href = "/api/export_csv";
      a.download = "benchmark.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => { isExportingCSV = false; }, 1000);
    }

    async function runModelPrediction() {
      if (!currentSelectedModel) {
        alert("Vui lòng bấm 'Chọn File Model' trước khi chạy thử.");
        return;
      }
      const txt = document.getElementById("reviewInput").value;
      const outBox = document.getElementById("structuredOutputBox");
      outBox.textContent = "Đang chạy mô hình...";
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_text: txt, model: currentSelectedModel })
      });
      const data = await res.json();
      outBox.textContent = data.formatted_text || data.error || "Không có";
    }

    function copyStructuredOutput() {
      const text = document.getElementById("structuredOutputBox").textContent;
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById("copyBtnText");
        btn.textContent = "Đã chép";
        setTimeout(() => { btn.textContent = "Sao chép"; }, 1500);
      });
    }

    window.addEventListener("DOMContentLoaded", () => {
      loadInitialState();
    });
  </script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class BenchmarkRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format_str, *args):
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            body = APPLE_GUI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/benchmark":
            sel = qs.get("model", [""])[0]
            data = get_benchmark_data(sel)
            self._send_json(data)
            return

        if path == "/api/benchmark/status":
            with BENCHMARK_LOCK:
                st = dict(BENCHMARK_STATE)
            self._send_json(st)
            return

        if path == "/api/export_csv":
            csv_bytes = generate_csv_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="benchmark.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/browse_model_file", "/api/browse_model_dir"):
            chosen = choose_model_file_dialog()
            if not chosen:
                self._send_json({"ok": False, "cancelled": True})
                return
            ok, msg, info = set_selected_model_file(chosen)
            if not ok or info is None:
                self._send_json({"ok": False, "error": msg}, status=400)
                return
            self._send_json({
                "ok": True,
                "model": info
            })
            return

        if parsed.path == "/api/benchmark/start":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                req = json.loads(raw_body)
            except Exception:
                req = {}
            model_key = req.get("model", "")
            ok, msg = start_benchmark_async(model_key)
            if not ok:
                self._send_json({"ok": False, "error": msg}, status=400)
                return
            self._send_json({"ok": True, "message": msg})
            return

        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                req = json.loads(raw_body)
            except Exception:
                req = {}
            review_txt = req.get("review_text", "")
            model_name = req.get("model", "")
            result = evaluate_new_review(review_txt, model_name)
            self._send_json(result)
            return

        if parsed.path == "/api/save_viz_image":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                req = json.loads(raw_body)
            except Exception:
                req = {}
            default_name = str(req.get("default_name", "benchmark_chart.png") or "benchmark_chart.png")
            data_url = str(req.get("data_url", "") or "")
            if not data_url:
                self._send_json({"ok": False, "error": "Dữ liệu hình ảnh trống."}, status=400)
                return
            chosen_file = choose_save_file_dialog(default_name=default_name)
            if not chosen_file:
                self._send_json({"ok": False, "cancelled": True})
                return
            try:
                out_path = Path(chosen_file).expanduser().resolve()
                save_data_url_png(data_url, out_path)
                self._send_json({"ok": True, "saved_path": str(out_path)})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if parsed.path == "/api/save_viz_batch":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                req = json.loads(raw_body)
            except Exception:
                req = {}
            images = req.get("images", [])
            if not isinstance(images, list) or not images:
                self._send_json({"ok": False, "error": "Không có hình ảnh để lưu."}, status=400)
                return
            chosen_dir = choose_save_directory_dialog()
            if not chosen_dir:
                self._send_json({"ok": False, "cancelled": True})
                return
            try:
                out_dir = Path(chosen_dir).expanduser().resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                saved_files = []
                for item in images:
                    if not isinstance(item, dict):
                        continue
                    fname = Path(str(item.get("filename", "chart.png"))).name
                    if not fname.lower().endswith(".png"):
                        fname += ".png"
                    d_url = str(item.get("data_url", "") or "")
                    if d_url:
                        dest = out_dir / fname
                        save_data_url_png(d_url, dest)
                        saved_files.append(str(dest))
                self._send_json({
                    "ok": True,
                    "saved_dir": str(out_dir),
                    "saved_files": saved_files
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        self._send_json({"error": "Not found"}, status=404)


def find_free_port(preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="CRAM-ABSA Benchmark GUI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.model_dir:
        set_model_dir(args.model_dir)

    port = find_free_port(args.port)
    server = ThreadedHTTPServer(("127.0.0.1", port), BenchmarkRequestHandler)
    url = f"http://127.0.0.1:{port}"

    print(f"CRAM-ABSA Benchmark GUI: {url}")
    if args.self_test:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.4)
        server.shutdown()
        return

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

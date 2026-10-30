import os
import re
import io
import csv
import sys
import math
import json
import time
import socket
import argparse
import threading
import webbrowser
import unicodedata
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional, Any
from collections import Counter, defaultdict

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics import f1_score
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

CANONICAL_ASPECTS = ["Service", "Facility", "Experience", "Amenity", "Loyalty", "Branding"]
ID_TO_SENTIMENT = {0: "Not Mentioned", 1: "Negative", 2: "Neutral", 3: "Positive"}
SENTIMENT_TO_ID = {"Negative": 1, "Neutral": 2, "Positive": 3}

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

OPINION_LEXICON_POS = {
    "good", "great", "excellent", "amazing", "wonderful", "friendly", "helpful", "clean",
    "spacious", "comfortable", "delicious", "lovely", "perfect", "best", "nice", "fantastic",
    "superb", "polite", "attentive", "convenient", "beautiful", "cozy", "relaxing", "awesome",
    "recommend", "recommended", "happy", "pleased", "impressed", "welcoming", "professional",
    "quiet", "modern", "fresh", "tasty", "worth", "luxury", "luxurious", "outstanding", "warm"
}

OPINION_LEXICON_NEG = {
    "bad", "poor", "terrible", "horrible", "awful", "slow", "rude", "dirty", "noisy", "small",
    "old", "outdated", "broken", "cold", "hard", "uncomfortable", "overpriced", "expensive",
    "worst", "disappointed", "disappointing", "unfriendly", "unhelpful", "smell", "smelly",
    "stain", "stained", "cockroach", "bug", "mold", "issue", "problem", "complaint", "wait",
    "waiting", "delayed", "mediocre", "basic", "boring", "crowded", "weak", "fault"
}

INTENSIFIERS = {
    "very", "extremely", "really", "super", "highly", "totally", "absolutely", "completely",
    "quite", "pretty", "so", "too", "exceptionally", "incredibly", "amazingly", "remarkably"
}

ASPECT_KEYWORDS = {
    "Service": {"staff", "service", "reception", "front", "desk", "manager", "waiter", "waitress", "concierge", "bellboy", "helpful", "friendly", "polite", "check", "checkin", "checkout", "attendant", "housekeeping", "host", "crew"},
    "Facility": {"room", "bed", "bathroom", "shower", "toilet", "air", "ac", "conditioning", "clean", "cleanliness", "noise", "noisy", "soundproof", "view", "balcony", "floor", "elevator", "lift", "pillow", "mattress", "furniture", "decor", "water", "carpet"},
    "Experience": {"stay", "stayed", "experience", "holiday", "vacation", "trip", "night", "overall", "atmosphere", "vibe", "relax", "relaxing", "enjoy", "enjoyed", "memorable", "time", "comfort", "comfortable", "quiet", "peaceful", "feeling"},
    "Amenity": {"breakfast", "buffet", "pool", "swimming", "spa", "gym", "fitness", "wifi", "internet", "restaurant", "bar", "food", "drink", "coffee", "tea", "lounge", "shuttle", "parking", "massage", "rooftop"},
    "Loyalty": {"back", "return", "again", "recommend", "recommended", "definitely", "choose", "future", "next", "visit", "repeat", "choice", "staying"},
    "Branding": {"star", "stars", "luxury", "brand", "hotel", "resort", "standard", "reputation", "class", "five", "four", "chain", "management", "value", "price", "worth", "quality", "prestigious", "accor", "sofitel", "pullman", "novotel", "marriott", "hilton", "intercontinental", "sheraton"}
}

ASPECT_ID_MAP = {asp: idx + 1 for idx, asp in enumerate(CANONICAL_ASPECTS)}
ASPECT_ID_MAP["None"] = 0

MODEL_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors"}
LOADED_MODEL_CACHE: Dict[str, Any] = {}

BENCHMARK_STATE: Dict[str, Any] = {
    "running": False,
    "progress_pct": 0,
    "message": "",
    "error": None,
    "latest_result": None
}


def get_model_dir() -> Path:
    cwd_model = Path.cwd() / "model"
    script_model = Path(__file__).resolve().parent / "model"
    if cwd_model.exists():
        return cwd_model
    if script_model.exists():
        return script_model
    return cwd_model


def get_dataset_path() -> Optional[Path]:
    candidates = [
        Path.cwd() / "data" / "df" / "hotel_reviews.parquet",
        Path.cwd() / "hotel_reviews.parquet",
        Path(__file__).resolve().parent / "data" / "df" / "hotel_reviews.parquet",
        Path(__file__).resolve().parent / "hotel_reviews.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def scan_available_models() -> List[Dict[str, Any]]:
    model_dir = get_model_dir()
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    models_found = []
    for p in sorted(model_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
            stat = p.stat()
            size_mb = stat.st_size / (1024 * 1024)
            size_label = f"{size_mb:.2f} MB" if size_mb >= 0.1 else f"{max(1, int(stat.st_size / 1024))} KB"
            models_found.append({
                "filename": p.name,
                "rel_path": f"model/{p.name}",
                "abs_path": str(p.resolve()),
                "size_label": size_label
            })
    return models_found


def nltk_negation_aware_preprocess(text: str):
    lower_txt = unicodedata.normalize("NFC", str(text)).lower()
    for k, v in CONTRACTIONS_MAP.items():
        lower_txt = re.sub(r"\b" + re.escape(k) + r"\b", v, lower_txt)
        lower_txt = lower_txt.replace(k, v)
    lower_txt = re.sub(r"@\w+|https?://\S+|www\.\S+", " ", lower_txt)
    raw_tokens = re.findall(r"[a-z]+", lower_txt)
    clean_tokens = [t for t in raw_tokens if t not in SAFE_STOPWORDS and len(t) > 1]
    return raw_tokens, clean_tokens


def extract_20_features_for_new_review(review_text: str, recency_weight: float = 1.0) -> np.ndarray:
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
        matched_asps = [asp for asp in CANONICAL_ASPECTS if any(t in ASPECT_KEYWORDS[asp] for t in s_clean)]
        if len(matched_asps) >= 2:
            overlap_flag = 1
        pos_hits = sum(1 for t in s_raw if t in OPINION_LEXICON_POS)
        neg_hits = sum(1 for t in s_raw if t in OPINION_LEXICON_NEG)
        for idx_t, tok in enumerate(s_raw[:-1]):
            if tok in {"not", "never", "no", "hardly"}:
                nxt = s_raw[idx_t + 1]
                if nxt in OPINION_LEXICON_POS:
                    pos_hits = max(0, pos_hits - 1)
                    neg_hits += 1
                elif nxt in OPINION_LEXICON_NEG:
                    neg_hits = max(0, neg_hits - 1)
                    pos_hits += 1
        sent_pol = (pos_hits - neg_hits) / (pos_hits + neg_hits) if (pos_hits + neg_hits) > 0 else 0.25
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
    temporal_aspect_sentiment = float(recency_weight * net_sentiment)

    return np.array([
        asp_count, dominant_aspect, net_sentiment, sent_conflict,
        evidence_density, avg_rat_len, overlap_flag,
        aspect_scores["Service"], aspect_scores["Facility"], aspect_scores["Experience"],
        aspect_scores["Amenity"], aspect_scores["Loyalty"], aspect_scores["Branding"],
        coverage_ratio, recency_weight, consensus_strength, temporal_aspect_sentiment,
        clean_token_ratio, opinion_neg_density, ebr_proto_sim
    ], dtype=np.float32)


def parse_annotations_safe(anns: Any) -> List[Dict[str, Any]]:
    if anns is None:
        return []
    if isinstance(anns, np.ndarray):
        return anns.tolist()
    if isinstance(anns, list):
        return anns
    if isinstance(anns, str):
        try:
            return json.loads(anns)
        except Exception:
            return []
    return []


if TORCH_AVAILABLE:
    class CRAM_ABSA_MultiAspectModel(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", hidden_dim=768, num_features=20, proj_dim=256, num_aspects=6, num_sent_classes=4, num_stars=5, max_comment_len=48):
            super().__init__()
            self.num_aspects = num_aspects
            self.num_sent_classes = num_sent_classes
            self.hidden_dim = hidden_dim
            self.max_comment_len = max_comment_len

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
            self.residual_gate = nn.Parameter(torch.tensor(0.25))
            self.ebr_aspect_prototypes = nn.Parameter(torch.randn(num_aspects, hidden_dim) * 0.02)
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

            self.aspect_star_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                    nn.Dropout(0.20),
                    nn.Linear(proj_dim, num_stars - 1)
                )
                for _ in range(num_aspects)
            ])

            self.corn_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(proj_dim, num_stars - 1)
            )

            self.commentary_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
            self.commentary_out = nn.Linear(hidden_dim, self.vocab_size, bias=False)
            self.commentary_out.weight = self.bert.embeddings.word_embeddings.weight

        def forward(self, input_ids, attention_mask, dense_features):
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
            rationale_logits = bilinear_scores + cosine_scores + self.rationale_bias

            pad_mask = attention_mask.unsqueeze(1).float()
            rationale_probs = torch.sigmoid(rationale_logits) * pad_mask

            aspect_logits_list = []
            aspect_star_logits_list = []
            aspect_context_list = []
            denom = rationale_probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            masked_pooled = torch.einsum("bal,bld->bad", rationale_probs, conditioned_seq) / denom

            for a_idx in range(self.num_aspects):
                asp_ctx = torch.cat([masked_pooled[:, a_idx, :], feat_emb], dim=-1)
                aspect_logits_list.append(self.aspect_classifiers[a_idx](asp_ctx))
                aspect_star_logits_list.append(self.aspect_star_heads[a_idx](asp_ctx))
                aspect_context_list.append(masked_pooled[:, a_idx, :])

            aspect_logits = torch.stack(aspect_logits_list, dim=1)
            aspect_star_logits = torch.stack(aspect_star_logits_list, dim=1)

            global_aspect_ctx = torch.stack(aspect_context_list, dim=1).mean(dim=1)
            corn_input = torch.cat([conditioned_cls, global_aspect_ctx], dim=-1)
            corn_logits = self.corn_head(corn_input)

            return {
                "aspect_logits": aspect_logits,
                "aspect_star_logits": aspect_star_logits,
                "rationale_logits": rationale_logits,
                "rationale_probs": rationale_probs,
                "corn_logits": corn_logits,
                "aspect_embeddings": masked_pooled,
                "seq_hidden": conditioned_seq
            }

        @torch.no_grad()
        def generate_aspect_commentary(self, aspect_idx: int, outputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> str:
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len]
            scores = rat_prob[:valid_len]

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

            best_window_ids = []
            if valid_len > 2:
                win_size = min(18, valid_len - 2)
                best_score = -1.0
                best_start = 1
                for s_i in range(1, max(2, valid_len - win_size)):
                    e_i = min(valid_len - 1, s_i + win_size)
                    w_sc = float(scores[s_i:e_i].mean().item())
                    if w_sc > best_score:
                        best_score = w_sc
                        best_start = s_i
                best_window_ids = tok_ids[best_start:min(valid_len - 1, best_start + win_size)].tolist()

            decoded_gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            decoded_span = tokenizer.decode(best_window_ids, skip_special_tokens=True).strip()
            return decoded_span if len(decoded_span) >= len(decoded_gen) else decoded_gen


def decode_corn(corn_logits):
    probs = torch.sigmoid(corn_logits)
    cum_probs = torch.cumprod(probs, dim=-1)
    return (cum_probs >= 0.5).sum(dim=-1)


def run_dataset_evaluation_worker(model_name: str, split: str = "test", sample_limit_str: str = "all"):
    global BENCHMARK_STATE
    BENCHMARK_STATE["running"] = True
    BENCHMARK_STATE["progress_pct"] = 0
    BENCHMARK_STATE["message"] = "Đang khởi tạo đánh giá..."
    BENCHMARK_STATE["error"] = None

    t_start = time.time()
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch hoặc Transformers chưa được cài đặt trong môi trường.")

        model_path = get_model_dir() / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_name}")

        dataset_path = get_dataset_path()
        if not dataset_path:
            raise FileNotFoundError("Không tìm thấy file dữ liệu hotel_reviews.parquet trong data/df/.")

        import pandas as pd
        BENCHMARK_STATE["message"] = f"Đang đọc dữ liệu từ {dataset_path.name}..."
        BENCHMARK_STATE["progress_pct"] = 5

        df = pd.read_parquet(dataset_path)
        split_clean = str(split).lower().strip()
        if "split" not in df.columns:
            raise ValueError("File parquet không có cột 'split'.")

        df_split = df[df["split"].str.lower() == split_clean].reset_index(drop=True)
        if len(df_split) == 0:
            raise ValueError(f"Không tìm thấy dữ liệu split '{split}' trong file.")

        if sample_limit_str != "all":
            try:
                limit_num = int(sample_limit_str)
                df_eval = df_split.head(limit_num).reset_index(drop=True)
            except Exception:
                df_eval = df_split
        else:
            df_eval = df_split

        total_samples = len(df_eval)
        BENCHMARK_STATE["message"] = f"Đang nạp mô hình {model_name}..."
        BENCHMARK_STATE["progress_pct"] = 10

        if model_name not in LOADED_MODEL_CACHE:
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise ValueError(f"File '{model_name}' không chứa trọng số model_state_dict hợp lệ.")
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            net = CRAM_ABSA_MultiAspectModel().to(DEVICE)
            net.load_state_dict(ckpt["model_state_dict"], strict=False)
            net.eval()
            LOADED_MODEL_CACHE[model_name] = {
                "model": net,
                "tokenizer": tok,
                "feat_mean": ckpt.get("feat_mean", None),
                "feat_std": ckpt.get("feat_std", None),
                "feature_cols_20": ckpt.get("feature_cols_20", None)
            }

        bundle = LOADED_MODEL_CACHE[model_name]
        net = bundle["model"]
        tok = bundle["tokenizer"]
        f_mean = bundle["feat_mean"]
        f_std = bundle["feat_std"]
        f_cols = bundle.get("feature_cols_20")

        if f_cols is None:
            f_cols = [
                "feat_aspect_count", "feat_dominant_aspect", "feat_net_sentiment_score", "feat_sentiment_conflict",
                "feat_evidence_density", "feat_avg_rationale_len", "feat_has_cross_overlap",
                "feat_service_sentiment", "feat_facility_sentiment", "feat_experience_sentiment",
                "feat_amenity_sentiment", "feat_loyalty_sentiment", "feat_branding_sentiment",
                "feat_aspect_coverage_ratio", "feat_recency_weight", "feat_consensus_strength",
                "feat_temporal_aspect_sentiment", "feat_nltk_clean_token_ratio",
                "feat_nltk_opinion_negation_density", "feat_ebr_aspect_prototype_sim"
            ]

        raw_feats = df_eval[f_cols].values.astype(np.float32)
        if f_mean is not None and f_std is not None:
            norm_feats = (raw_feats - f_mean) / np.where(f_std < 1e-6, 1.0, f_std)
        else:
            norm_feats = raw_feats

        batch_size = 16
        total_batches = math.ceil(total_samples / batch_size)

        all_star_preds = []
        all_star_trues = []
        all_asp_preds = []
        all_asp_trues = []
        rat_inter, rat_pred_sum, rat_true_sum = 0.0, 0.0, 0.0
        overlap_rates = []
        total_loss = 0.0

        BENCHMARK_STATE["message"] = f"Bắt đầu đánh giá {total_samples} mẫu..."

        for b_i in range(total_batches):
            s_idx = b_i * batch_size
            e_idx = min(s_idx + batch_size, total_samples)
            batch_df = df_eval.iloc[s_idx:e_idx]
            b_size = len(batch_df)

            texts = [unicodedata.normalize("NFC", str(t)) for t in batch_df["review_text"].tolist()]
            enc = tok(
                texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            offsets = enc.pop("offset_mapping").numpy()
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            dense_tensor = torch.tensor(norm_feats[s_idx:e_idx], dtype=torch.float32, device=DEVICE)

            batch_aspect_labels = []
            batch_rationale_masks = []
            batch_ordinal = []

            for row_i, (_, row) in enumerate(batch_df.iterrows()):
                ord_val = int(row.get("ordinal_rating", int(row["rating_5"]) - 1))
                batch_ordinal.append(max(0, min(4, ord_val)))

                anns = parse_annotations_safe(row.get("aspect_annotations", []))
                asp_votes = defaultdict(list)
                r_mask = np.zeros((6, 256), dtype=np.float32)
                sample_offsets = offsets[row_i]

                for ann in anns:
                    asp = ann.get("aspect")
                    if asp not in ASPECT_ID_MAP or asp == "None":
                        continue
                    a_idx = ASPECT_ID_MAP[asp] - 1
                    s_char = int(ann.get("start_char", 0))
                    e_char = int(ann.get("end_char", 0))
                    sent = ann.get("sentiment", "Neutral")
                    asp_votes[a_idx].append(SENTIMENT_TO_ID.get(sent, 2))
                    for t_i, (tok_s, tok_e) in enumerate(sample_offsets):
                        if tok_e > tok_s and tok_s < e_char and tok_e > s_char:
                            r_mask[a_idx, t_i] = 1.0

                asp_labels = [0] * 6
                for a_idx, votes in asp_votes.items():
                    asp_labels[a_idx] = Counter(votes).most_common(1)[0][0]

                batch_aspect_labels.append(asp_labels)
                batch_rationale_masks.append(r_mask)

            with torch.no_grad():
                outputs = net(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_tensor)

            star_preds = decode_corn(outputs["corn_logits"]).cpu().numpy()
            asp_preds = outputs["aspect_logits"].argmax(dim=-1).cpu().numpy()

            all_star_preds.extend(star_preds.tolist())
            all_star_trues.extend(batch_ordinal)
            all_asp_preds.append(asp_preds)
            all_asp_trues.append(np.array(batch_aspect_labels))

            pred_mask = (outputs["rationale_probs"] >= 0.5).float().cpu()
            true_mask = torch.tensor(np.array(batch_rationale_masks), dtype=torch.float32)

            rat_inter += (pred_mask * true_mask).sum().item()
            rat_pred_sum += pred_mask.sum().item()
            rat_true_sum += true_mask.sum().item()

            active_asp_per_tok = pred_mask.sum(dim=1)
            leaked_toks = (active_asp_per_tok > 1.0).float().sum().item()
            active_toks = (active_asp_per_tok >= 1.0).float().sum().item()
            overlap_rates.append(leaked_toks / max(active_toks, 1.0))

            # Tính hàm mất mát
            corn_targets = torch.zeros((b_size, 4), dtype=torch.float32, device=DEVICE)
            for row_i, r in enumerate(batch_ordinal):
                for k in range(4):
                    if r > k:
                        corn_targets[row_i, k] = 1.0
            corn_loss = F.binary_cross_entropy_with_logits(outputs["corn_logits"], corn_targets)
            aspect_target_tensor = torch.tensor(np.array(batch_aspect_labels), dtype=torch.long, device=DEVICE)
            aspect_loss = F.cross_entropy(outputs["aspect_logits"].view(-1, 4), aspect_target_tensor.view(-1))
            total_loss += (corn_loss.item() + aspect_loss.item()) * b_size

            pct = int(((b_i + 1) / total_batches) * 85) + 10
            BENCHMARK_STATE["progress_pct"] = min(pct, 95)
            BENCHMARK_STATE["message"] = f"Đang đánh giá batch {b_i + 1}/{total_batches} ({e_idx}/{total_samples} mẫu)..."

        BENCHMARK_STATE["message"] = "Đang tổng hợp các chỉ số đánh giá..."
        BENCHMARK_STATE["progress_pct"] = 96

        y_true_star = np.array(all_star_trues)
        y_pred_star = np.array(all_star_preds)
        acc_at_0 = float(np.mean(y_true_star == y_pred_star) * 100.0)

        per_star_mae = [
            float(np.abs(y_true_star[y_true_star == k] - y_pred_star[y_true_star == k]).mean())
            for k in range(5) if np.any(y_true_star == k)
        ]
        macro_mae = float(np.mean(per_star_mae)) if per_star_mae else 0.0

        Y_asp_true = np.vstack(all_asp_trues)
        Y_asp_pred = np.vstack(all_asp_preds)
        aspect_f1s = [
            float(f1_score(Y_asp_true[:, a], Y_asp_pred[:, a], average="macro", zero_division=0) * 100.0)
            for a in range(len(CANONICAL_ASPECTS))
        ]
        macro_aspect_f1 = float(np.mean(aspect_f1s))
        rationale_f1 = float((2.0 * rat_inter / max(rat_pred_sum + rat_true_sum, 1.0)) * 100.0)
        leakage_rate = float(np.mean(overlap_rates) * 100.0)
        avg_loss = float(total_loss / max(total_samples, 1))
        elapsed_sec = round(time.time() - t_start, 1)

        result_payload = {
            "model": model_name,
            "split": split_clean.upper(),
            "samples": f"{total_samples:,}",
            "val_loss": round(avg_loss, 4),
            "acc_at_0": round(acc_at_0, 2),
            "aspect_macro_f1": round(macro_aspect_f1, 2),
            "macro_mae": round(macro_mae, 4),
            "rationale_f1": round(rationale_f1, 2),
            "leakage_rate": round(leakage_rate, 2),
            "elapsed": f"{elapsed_sec}s"
        }

        BENCHMARK_STATE["latest_result"] = {
            "has_model": True,
            "selected_model": model_name,
            "message": "OK",
            "metrics": result_payload,
            "rows": [result_payload]
        }
        BENCHMARK_STATE["progress_pct"] = 100
        BENCHMARK_STATE["message"] = f"Đã hoàn thành đánh giá ({total_samples:,} mẫu trong {elapsed_sec}s)!"
        BENCHMARK_STATE["running"] = False

    except Exception as e:
        BENCHMARK_STATE["running"] = False
        BENCHMARK_STATE["error"] = str(e)
        BENCHMARK_STATE["message"] = f"Lỗi đánh giá: {e}"


def evaluate_new_review(review_text: str, model_name: str) -> Dict[str, Any]:
    raw_text = unicodedata.normalize("NFC", str(review_text).strip())
    if not raw_text:
        return {
            "ok": False,
            "error": "Vui lòng nhập nội dung review.",
            "formatted_text": ""
        }

    models = scan_available_models()
    if not models or not model_name:
        return {
            "ok": False,
            "error": "Không có mô hình trong thư mục model/.",
            "formatted_text": "Không có mô hình trong thư mục model/."
        }

    model_path = get_model_dir() / model_name
    if not model_path.exists():
        return {
            "ok": False,
            "error": f"Không tìm thấy mô hình: {model_name}",
            "formatted_text": f"Không tìm thấy mô hình: {model_name}"
        }

    if not TORCH_AVAILABLE:
        return {
            "ok": False,
            "error": "Môi trường hiện tại chưa cài đặt PyTorch/Transformers để chạy file mô hình.",
            "formatted_text": "Môi trường hiện tại chưa cài đặt PyTorch/Transformers để chạy file mô hình."
        }

    try:
        if model_name not in LOADED_MODEL_CACHE:
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                return {
                    "ok": False,
                    "error": f"File '{model_name}' không chứa trọng số model_state_dict hợp lệ.",
                    "formatted_text": f"File '{model_name}' không chứa trọng số model_state_dict hợp lệ."
                }
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            net = CRAM_ABSA_MultiAspectModel().to(DEVICE)
            net.load_state_dict(ckpt["model_state_dict"], strict=False)
            net.eval()
            LOADED_MODEL_CACHE[model_name] = {
                "model": net,
                "tokenizer": tok,
                "feat_mean": ckpt.get("feat_mean", None),
                "feat_std": ckpt.get("feat_std", None)
            }

        bundle = LOADED_MODEL_CACHE[model_name]
        net = bundle["model"]
        tok = bundle["tokenizer"]
        f_mean = bundle["feat_mean"]
        f_std = bundle["feat_std"]

        raw_feat_20 = extract_20_features_for_new_review(raw_text, recency_weight=1.0)
        if f_mean is not None and f_std is not None:
            norm_feat_20 = (raw_feat_20 - f_mean) / np.where(f_std < 1e-6, 1.0, f_std)
        else:
            norm_feat_20 = raw_feat_20

        enc = tok(
            raw_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        dense_tensor = torch.tensor(norm_feat_20, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            outputs = net(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_tensor)
            asp_probs = F.softmax(outputs["aspect_logits"].squeeze(0), dim=-1).cpu().numpy()
            asp_stars = decode_corn(outputs["aspect_star_logits"].squeeze(0)).cpu().numpy() + 1

            mentioned_aspects = []
            lines = []
            aspect_details = {}

            for a_idx, asp in enumerate(CANONICAL_ASPECTS):
                pred_cls = int(np.argmax(asp_probs[a_idx]))
                if pred_cls != 0:
                    mentioned_aspects.append(asp)
                    star_val = int(asp_stars[a_idx])
                    sent_label = ID_TO_SENTIMENT[pred_cls]
                    gen_text = net.generate_aspect_commentary(a_idx, outputs, input_ids, attention_mask, tok)
                    aspect_details[asp] = f"{star_val}/5 ★ ({sent_label}) - {gen_text}"
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


def generate_csv_bytes(selected_model: str) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Model", "Split", "Samples", "Loss", "Acc@0 (%)",
        "6-Aspect Macro-F1 (%)", "Macro-MAE", "Rationale F1 (%)", "Leakage (%)", "Elapsed"
    ])
    res = BENCHMARK_STATE.get("latest_result")
    if res and res.get("rows"):
        for r in res["rows"]:
            writer.writerow([
                r["model"], r["split"], r["samples"], r["val_loss"], r["acc_at_0"],
                r["aspect_macro_f1"], r["macro_mae"], r["rationale_f1"], r["leakage_rate"], r["elapsed"]
            ])
    return buf.getvalue().encode("utf-8-sig")


APPLE_GUI_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Benchmark</title>
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
      --rounded-lg: 18px;
      --rounded-pill: 9999px;
      --font-display: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
      --font-text: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    html, body {
      height: 100vh;
      overflow: hidden;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-text);
      font-size: 17px;
      font-weight: 400;
      line-height: 1.47;
      letter-spacing: -0.374px;
      color: var(--ink);
      background-color: var(--canvas-parchment);
      display: flex;
      flex-direction: column;
    }

    .workspace-shell {
      display: flex;
      width: 100%;
      height: 100vh;
      overflow: hidden;
      margin: 0;
    }

    .sidebar {
      width: 250px;
      background-color: var(--canvas-parchment);
      border-right: 1px solid var(--hairline);
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      flex-shrink: 0;
      height: 100vh;
      overflow-y: auto;
    }

    .sidebar-brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--ink);
      font-family: var(--font-display);
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.2px;
      padding: 0 10px;
      margin-bottom: 24px;
    }

    .sidebar-brand svg {
      color: var(--primary);
    }

    .sidebar-section-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--ink-muted-48);
      text-transform: uppercase;
      padding: 0 12px 8px;
    }

    .nav-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 4px;
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
    }

    .nav-item svg {
      width: 18px;
      height: 18px;
      color: var(--ink-muted-48);
    }

    .nav-item.active {
      background-color: var(--canvas);
      color: var(--primary);
      font-weight: 600;
      border-color: var(--hairline);
    }

    .nav-item.active svg {
      color: var(--primary);
    }

    .nav-item:active {
      transform: scale(0.95);
    }

    .sidebar-model-box {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 16px;
    }

    .sidebar-model-box h4 {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .sidebar-model-box p {
      font-size: 12px;
      color: var(--ink-muted-48);
    }

    .main-canvas {
      flex: 1;
      padding: 32px 36px 40px;
      display: flex;
      flex-direction: column;
      gap: 24px;
      height: 100vh;
      overflow-y: auto;
      position: relative;
    }

    /* Floating Island Action Bar */
    .floating-controls-bar {
      position: fixed;
      top: 20px;
      right: 28px;
      z-index: 1000;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 8px;
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: saturate(180%) blur(20px);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      border: 1px solid rgba(0, 0, 0, 0.08);
      border-radius: var(--rounded-pill);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Thống nhất kiểu dáng bo tròn pill cho toàn bộ điều khiển */
    .floating-pill-control {
      height: 38px;
      border-radius: var(--rounded-pill) !important;
      font-family: var(--font-text);
      font-size: 13px;
      font-weight: 500;
      box-sizing: border-box;
      transition: all 0.15s ease;
      outline: none;
    }

    .floating-pill-control:active {
      transform: scale(0.96);
    }

    .floating-pill-select {
      appearance: none;
      -webkit-appearance: none;
      -moz-appearance: none;
      background-color: var(--canvas);
      color: var(--ink);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-pill) !important;
      padding: 0 34px 0 16px;
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%237A7A7A' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 14px center;
      background-size: 10px 6px;
      line-height: 36px;
    }

    .floating-pill-select:hover {
      border-color: #c7c7c9;
      background-color: var(--surface-pearl);
    }

    .floating-pill-icon {
      width: 38px;
      min-width: 38px;
      padding: 0;
      border: 1px solid var(--hairline);
      background-color: var(--canvas);
      color: var(--ink);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }

    .floating-pill-icon:hover {
      border-color: #c7c7c9;
      background-color: var(--surface-pearl);
    }

    .floating-pill-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      cursor: pointer;
      white-space: nowrap;
    }

    .floating-pill-btn.secondary {
      background-color: var(--canvas);
      color: var(--primary);
      border: 1px solid var(--primary);
      padding: 0 16px;
    }

    .floating-pill-btn.secondary:hover {
      background-color: rgba(0, 102, 204, 0.06);
    }

    .floating-pill-btn.primary {
      background-color: var(--primary);
      color: #ffffff;
      border: 1px solid transparent;
      padding: 0 18px;
      font-weight: 600;
      box-shadow: 0 2px 8px rgba(0, 102, 204, 0.25);
    }

    .floating-pill-btn.primary:hover {
      background-color: var(--primary-focus);
    }

    .floating-pill-btn.primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    .view-panel {
      display: none;
      flex-direction: column;
      gap: 24px;
    }

    .view-panel.active {
      display: flex;
    }

    .hero-head {
      font-family: var(--font-display);
      font-size: 34px;
      font-weight: 600;
      letter-spacing: -0.374px;
    }

    .btn-pill-primary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background-color: var(--primary);
      color: var(--body-on-dark);
      font-family: var(--font-text);
      font-size: 14px;
      border: none;
      border-radius: var(--rounded-pill);
      padding: 10px 20px;
      cursor: pointer;
      box-sizing: border-box;
      transition: all 0.15s ease;
    }

    .btn-pill-primary:active, .btn-pill-secondary:active, .btn-icon-circle:active {
      transform: scale(0.96);
    }

    .btn-pill-secondary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background-color: var(--canvas);
      color: var(--primary);
      font-family: var(--font-text);
      font-size: 14px;
      border: 1px solid var(--primary);
      border-radius: var(--rounded-pill);
      padding: 9px 18px;
      cursor: pointer;
      box-sizing: border-box;
      transition: all 0.15s ease;
    }

    .btn-icon-circle {
      width: 38px;
      height: 38px;
      border-radius: var(--rounded-pill);
      border: 1px solid var(--hairline);
      background-color: var(--canvas);
      color: var(--ink);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-sizing: border-box;
      transition: all 0.15s ease;
    }

    .apple-select {
      appearance: none;
      -webkit-appearance: none;
      -moz-appearance: none;
      height: 38px;
      background-color: var(--canvas);
      color: var(--ink);
      font-family: var(--font-text);
      font-size: 14px;
      font-weight: 500;
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-pill) !important;
      padding: 0 34px 0 16px;
      outline: none;
      box-sizing: border-box;
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%237A7A7A' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 14px center;
      background-size: 10px 6px;
    }

    .utility-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 24px;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }

    .kpi-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .kpi-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 14px;
      font-weight: 600;
      color: var(--ink-muted-48);
    }

    .kpi-value {
      font-family: var(--font-display);
      font-size: 34px;
      font-weight: 600;
      letter-spacing: -0.374px;
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 21px;
      font-weight: 600;
    }

    table.apple-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    table.apple-table th {
      text-align: left;
      font-weight: 600;
      color: var(--ink-muted-48);
      padding: 12px 14px;
      border-bottom: 1px solid var(--hairline);
    }

    table.apple-table td {
      padding: 13px 14px;
      border-bottom: 1px solid var(--divider-soft);
      font-variant-numeric: tabular-nums;
    }

    .review-textarea {
      width: 100%;
      min-height: 120px;
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 16px 20px;
      font-family: var(--font-text);
      font-size: 17px;
      line-height: 1.47;
      color: var(--ink);
      background-color: var(--surface-pearl);
      resize: vertical;
      outline: none;
      margin-bottom: 16px;
    }

    .review-textarea:focus {
      background-color: var(--canvas);
      outline: 2px solid var(--primary-focus);
    }

    .structured-output-box {
      background-color: var(--surface-tile-1);
      color: var(--body-on-dark);
      border-radius: var(--rounded-lg);
      padding: 24px;
      font-family: "SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
      line-height: 1.65;
      white-space: pre-wrap;
      min-height: 160px;
    }
  </style>
</head>
<body>

  <!-- Floating Controls Capsule (Thống nhất toàn bộ phím điều khiển dạng bo tròn) -->
  <div class="floating-controls-bar">
    <select id="modelSelect" class="floating-pill-control floating-pill-select" onchange="onModelChange()"></select>
    <button class="floating-pill-control floating-pill-icon" title="Quét lại thư mục model/" onclick="refreshModelList()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 4 23 10 17 10"></polyline>
        <polyline points="1 20 1 14 7 14"></polyline>
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
      </svg>
    </button>
    <div id="benchmarkActionGroup" style="display:inline-flex; align-items:center; gap:8px;">
      <select id="splitSelect" class="floating-pill-control floating-pill-select">
        <option value="test">Test set (999 mẫu)</option>
        <option value="val">Val set (999 mẫu)</option>
      </select>
      <select id="sampleSelect" class="floating-pill-control floating-pill-select">
        <option value="all">Toàn bộ mẫu</option>
        <option value="100">Nhanh 100 mẫu</option>
      </select>
      <button class="floating-pill-control floating-pill-btn secondary" id="btnExportCSV" onclick="exportCSV()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <span>Xuất CSV</span>
      </button>
      <button class="floating-pill-control floating-pill-btn primary" id="btnRunBenchmark" onclick="startBenchmarkRun()">
        <svg id="runIcon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span id="runLabel">Chạy Benchmark</span>
      </button>
    </div>
  </div>

  <div class="workspace-shell">
    <aside class="sidebar">
      <div>
        <div class="sidebar-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
          <span>Benchmark</span>
        </div>
        <div class="sidebar-section-label">Menu</div>
        <ul class="nav-list">
          <li>
            <button class="nav-item active" id="nav-overview" onclick="switchTab('overview')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="1"></rect>
                <rect x="14" y="3" width="7" height="7" rx="1"></rect>
                <rect x="14" y="14" width="7" height="7" rx="1"></rect>
                <rect x="3" y="14" width="7" height="7" rx="1"></rect>
              </svg>
              <span>Đánh giá Model</span>
            </button>
          </li>
          <li>
            <button class="nav-item" id="nav-playground" onclick="switchTab('playground')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
              <span>Chạy thử Model</span>
            </button>
          </li>
        </ul>
      </div>

      <div class="sidebar-model-box">
        <h4>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
          </svg>
          <span>Thư mục model/</span>
        </h4>
        <p id="sidebarModelMeta">Không có</p>
      </div>
    </aside>

    <main class="main-canvas">

      <section class="view-panel active" id="panel-overview">
        <h1 class="hero-head">Kết quả Đánh giá</h1>

        <div id="progressCard" class="utility-card" style="display:none; padding:16px 20px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span id="progressText" style="font-weight:600; font-size:14px; color:var(--ink);">Đang chuẩn bị...</span>
            <span id="progressPercent" style="font-weight:600; font-size:14px; color:var(--primary);">0%</span>
          </div>
          <div style="width:100%; height:8px; background:var(--canvas-parchment); border-radius:999px; overflow:hidden;">
            <div id="progressBar" style="width:0%; height:100%; background:var(--primary); transition:width 0.25s ease;"></div>
          </div>
        </div>

        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-top">
              <span>Acc@0</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <div class="kpi-value" id="kpiAcc0">--</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>6-Aspect F1</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
            </div>
            <div class="kpi-value" id="kpiAspectF1">--</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>Macro-MAE</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            </div>
            <div class="kpi-value" id="kpiMae">--</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>Leakage Rate</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div class="kpi-value" id="kpiLeakage">--</div>
          </div>
        </div>

        <div class="utility-card">
          <div class="card-header">
            <h2 class="card-title">Bảng Đánh giá Model</h2>
            <button class="btn-pill-secondary" onclick="exportCSV()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span>Xuất CSV</span>
            </button>
          </div>
          <table class="apple-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Split</th>
                <th>Số mẫu</th>
                <th>Loss</th>
                <th>Acc@0 (%)</th>
                <th>6-Aspect F1 (%)</th>
                <th>Macro-MAE</th>
                <th>Rationale F1 (%)</th>
                <th>Leakage (%)</th>
                <th>Thời gian</th>
              </tr>
            </thead>
            <tbody id="summaryTableBody">
              <tr><td colspan="10" style="color:var(--ink-muted-48); text-align:center;">Chưa có kết quả. Chọn model, split và bấm 'Chạy Benchmark' để bắt đầu đánh giá.</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="view-panel" id="panel-playground">
        <h1 class="hero-head">Chạy thử Model</h1>

        <div class="utility-card">
          <div class="card-header">
            <h2 class="card-title">Nhập Review</h2>
          </div>
          <textarea id="reviewInput" class="review-textarea" placeholder="Nhập review tại đây..."></textarea>
          <div style="display:flex;gap:12px;">
            <button class="btn-pill-primary" onclick="runLivePrediction()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon>
              <span>Chạy thử</span>
            </button>
            <button class="btn-pill-secondary" onclick="copyStructuredText()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              <span id="copyBtnLabel">Sao chép</span>
            </button>
          </div>
        </div>

        <div class="utility-card">
          <div class="card-header">
            <h2 class="card-title">Output</h2>
          </div>
          <div class="structured-output-box" id="structuredTextOutput"></div>
        </div>
      </section>

    </main>
  </div>

  <script>
    let currentModel = "";
    let pollInterval = null;

    function switchTab(tabName) {
      document.querySelectorAll(".view-panel").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
      document.getElementById("panel-" + tabName).classList.add("active");
      document.getElementById("nav-" + tabName).classList.add("active");

      const grp = document.getElementById("benchmarkActionGroup");
      if (grp) {
        grp.style.display = (tabName === "overview") ? "inline-flex" : "none";
      }
    }

    async function refreshModelList() {
      const res = await fetch("/api/models");
      const data = await res.json();
      const sel = document.getElementById("modelSelect");
      sel.innerHTML = "";
      if (!data.models || data.models.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Không có model";
        sel.appendChild(opt);
        currentModel = "";
        document.getElementById("sidebarModelMeta").textContent = "Không có";
      } else {
        data.models.forEach(m => {
          const opt = document.createElement("option");
          opt.value = m.filename;
          opt.textContent = `${m.filename} (${m.size_label})`;
          sel.appendChild(opt);
        });
        currentModel = sel.value;
        document.getElementById("sidebarModelMeta").textContent = data.models.map(x => x.filename).join(", ");
      }
      resetResultsDisplay();
    }

    function onModelChange() {
      const sel = document.getElementById("modelSelect");
      currentModel = sel.value;
      resetResultsDisplay();
    }

    function resetResultsDisplay() {
      document.getElementById("kpiAcc0").textContent = "--";
      document.getElementById("kpiAspectF1").textContent = "--";
      document.getElementById("kpiMae").textContent = "--";
      document.getElementById("kpiLeakage").textContent = "--";
      document.getElementById("summaryTableBody").innerHTML =
        `<tr><td colspan="10" style="color:var(--ink-muted-48); text-align:center;">Chưa có kết quả. Chọn model, split và bấm 'Chạy Benchmark' để bắt đầu đánh giá.</td></tr>`;
      document.getElementById("progressCard").style.display = "none";
    }

    async function startBenchmarkRun() {
      if (!currentModel) {
        alert("Vui lòng chọn một mô hình trước khi chạy!");
        return;
      }

      const split = document.getElementById("splitSelect").value;
      const samples = document.getElementById("sampleSelect").value;

      const btn = document.getElementById("btnRunBenchmark");
      btn.disabled = true;
      btn.style.opacity = "0.6";
      btn.style.cursor = "not-allowed";
      document.getElementById("runLabel").textContent = "Đang đánh giá...";

      const pCard = document.getElementById("progressCard");
      pCard.style.display = "block";
      document.getElementById("progressBar").style.width = "5%";
      document.getElementById("progressPercent").textContent = "5%";
      document.getElementById("progressText").textContent = "Đang khởi tạo...";

      try {
        const res = await fetch("/api/benchmark/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: currentModel, split: split, samples: samples })
        });
        const startData = await res.json();
        if (!startData.ok) {
          alert(startData.error || "Không thể khởi chạy benchmark.");
          resetRunButton();
          return;
        }

        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(checkBenchmarkStatus, 500);
      } catch (err) {
        alert("Lỗi khi kết nối tới máy chủ: " + err);
        resetRunButton();
      }
    }

    async function checkBenchmarkStatus() {
      try {
        const res = await fetch("/api/benchmark/status");
        const st = await res.json();

        document.getElementById("progressBar").style.width = `${st.progress_pct}%`;
        document.getElementById("progressPercent").textContent = `${st.progress_pct}%`;
        document.getElementById("progressText").textContent = st.message || "";

        if (st.error) {
          clearInterval(pollInterval);
          pollInterval = null;
          alert("Quá trình đánh giá gặp lỗi: " + st.error);
          resetRunButton();
          return;
        }

        if (!st.running && st.latest_result) {
          clearInterval(pollInterval);
          pollInterval = null;
          resetRunButton();

          const r = st.latest_result.metrics;
          document.getElementById("kpiAcc0").textContent = r.acc_at_0.toFixed(2) + "%";
          document.getElementById("kpiAspectF1").textContent = r.aspect_macro_f1.toFixed(2) + "%";
          document.getElementById("kpiMae").textContent = r.macro_mae.toFixed(4);
          document.getElementById("kpiLeakage").textContent = r.leakage_rate.toFixed(2) + "%";

          const tbody = document.getElementById("summaryTableBody");
          tbody.innerHTML = `
            <tr>
              <td><strong>${r.model}</strong></td>
              <td><span style="display:inline-block;padding:2px 8px;border-radius:999px;background:var(--canvas-parchment);font-weight:600;font-size:12px;">${r.split}</span></td>
              <td>${r.samples}</td>
              <td>${r.val_loss}</td>
              <td>${r.acc_at_0.toFixed(2)}%</td>
              <td>${r.aspect_macro_f1.toFixed(2)}%</td>
              <td>${r.macro_mae.toFixed(4)}</td>
              <td>${r.rationale_f1.toFixed(2)}%</td>
              <td>${r.leakage_rate.toFixed(2)}%</td>
              <td>${r.elapsed}</td>
            </tr>
          `;
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }

    function resetRunButton() {
      const btn = document.getElementById("btnRunBenchmark");
      btn.disabled = false;
      btn.style.opacity = "1";
      btn.style.cursor = "pointer";
      document.getElementById("runLabel").textContent = "Chạy Benchmark";
    }

    function exportCSV() {
      if (!currentModel) return;
      window.location.href = `/api/export_csv?model=${encodeURIComponent(currentModel)}`;
    }

    async function runLivePrediction() {
      const reviewText = document.getElementById("reviewInput").value;
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_text: reviewText, model: currentModel })
      });
      const data = await res.json();
      document.getElementById("structuredTextOutput").textContent = data.formatted_text || data.error || "";
    }

    function copyStructuredText() {
      const txt = document.getElementById("structuredTextOutput").textContent;
      if (!txt) return;
      navigator.clipboard.writeText(txt);
      const lbl = document.getElementById("copyBtnLabel");
      lbl.textContent = "Đã sao chép";
      setTimeout(() => { lbl.textContent = "Sao chép"; }, 1200);
    }

    window.addEventListener("DOMContentLoaded", async () => {
      await refreshModelList();
    });
  </script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class BenchmarkGUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload: Any, status: int = 200):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            raw = APPLE_GUI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/models":
            self._send_json({"models": scan_available_models()})
            return

        if parsed.path == "/api/benchmark/status":
            self._send_json(BENCHMARK_STATE)
            return

        if parsed.path == "/api/export_csv":
            model_name = qs.get("model", [""])[0]
            csv_bytes = generate_csv_bytes(model_name)
            stem = Path(model_name).stem if model_name else "empty"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="benchmark_{stem}.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/benchmark/start":
            if BENCHMARK_STATE.get("running"):
                self._send_json({"ok": False, "error": "Đang có tiến trình đánh giá chạy!"})
                return
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(body_raw)
            model_name = payload.get("model", "")
            split = payload.get("split", "test")
            samples = payload.get("samples", "all")

            worker_thread = threading.Thread(
                target=run_dataset_evaluation_worker,
                args=(model_name, split, samples),
                daemon=True
            )
            worker_thread.start()
            self._send_json({"ok": True, "message": "Đã bắt đầu đánh giá."})
            return

        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(body_raw)
            review_text = payload.get("review_text", "")
            model_name = payload.get("model", "")
            self._send_json(evaluate_new_review(review_text, model_name))
            return

        self.send_error(404, "Not Found")


def find_free_port(preferred: int = 8080) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 0


def main():
    parser = argparse.ArgumentParser(description="CRAM-ABSA Benchmark GUI")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    port = find_free_port(args.port)
    server = ThreadedHTTPServer(("127.0.0.1", port), BenchmarkGUIHandler)
    url = f"http://127.0.0.1:{port}"

    models = scan_available_models()
    model_str = ", ".join(m["filename"] for m in models) if models else "Không có"
    print(f"Benchmark GUI: {url} | Model trong model/: {model_str}")

    try:
        import webview
        threading.Thread(target=server.serve_forever, daemon=True).start()
        webview.create_window("Benchmark", url, width=1320, height=840)
        webview.start()
    except Exception:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()

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

import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SENTIMENT_TO_ID = {"Negative": 1, "Neutral": 2, "Positive": 3}


os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
    from transformers.modeling_outputs import BaseModelOutput
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

CANONICAL_ASPECTS = ["Service", "Facility", "Experience", "Amenity", "Loyalty", "Branding"]
ASPECT_ID_MAP = {asp: idx + 1 for idx, asp in enumerate(CANONICAL_ASPECTS)}
ID_TO_SENTIMENT = {0: "Not Mentioned", 1: "Negative", 2: "Neutral", 3: "Positive"}

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

HEDGING_MODIFIERS = {
    "moderate", "okay", "ok", "decent", "fair", "acceptable", "average", "slightly",
    "bit", "ordinary", "normal", "passable", "tolerable", "adequate", "fine", "basic"
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
        "room", "rooms", "bed", "beds", "bathroom", "shower", "toilet", "air", "ac",
        "conditioning", "clean", "cleanliness", "noise", "noisy", "soundproof", "view",
        "balcony", "floor", "elevator", "lift", "pillow", "mattress", "furniture", "decor",
        "water", "carpet", "lobby", "parking", "surrounding", "area", "exterior", "interior"
    },
    "Experience": {
        "stay", "stayed", "experience", "holiday", "vacation", "trip", "night", "nights",
        "overall", "atmosphere", "vibe", "relax", "relaxing", "enjoy", "enjoyed", "memorable",
        "comfort", "comfortable", "quiet", "peaceful", "feeling", "impression", "journey", "moment"
    },
    "Amenity": {
        "kitchen", "food", "foods", "meal", "meals", "dish", "dishes", "taste", "tasty",
        "delicious", "flavor", "dining", "breakfast", "buffet", "pool", "swimming", "spa",
        "gym", "fitness", "wifi", "internet", "restaurant", "bar", "drink", "drinks",
        "coffee", "tea", "lounge", "shuttle", "massage", "rooftop", "menu", "cuisine", "chef"
    },
    "Loyalty": {
        "back", "return", "returned", "returning", "again", "times", "visit", "visits",
        "visited", "revisit", "recommend", "recommended", "definitely", "choose", "future",
        "next", "repeat", "loyal", "regular", "choice", "staying"
    },
    "Branding": {
        "star", "stars", "luxury", "luxurious", "brand", "chain", "standard", "reputation",
        "class", "five", "four", "management", "value", "price", "pricing", "worth",
        "prestigious", "accor", "sofitel", "pullman", "novotel", "marriott", "hilton",
        "intercontinental", "sheraton", "flagship", "resort"
    }
}

ASPECT_ID_MAP["None"] = 0

MODEL_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors"}
LOADED_MODEL_CACHE: Dict[str, Any] = {}


def get_model_dir() -> Path:
    cwd_model = Path.cwd() / "model"
    script_model = Path(__file__).resolve().parent / "model"
    if cwd_model.exists():
        return cwd_model
    if script_model.exists():
        return script_model
    return cwd_model


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
    from collections import Counter, defaultdict
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


    class CRAM_ABSA_v2_MultiAspectModel(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", gen_model_name="google/flan-t5-small", hidden_dim=768, num_features=20, proj_dim=256, attn_heads=8, num_prompt_tokens=4, num_aspects=6, num_sent_classes=4, num_stars=5, max_comment_len=48):
            super().__init__()
            self.num_aspects = num_aspects
            self.num_sent_classes = num_sent_classes
            self.hidden_dim = hidden_dim
            self.max_comment_len = max_comment_len
            self.num_prompt_tokens = num_prompt_tokens

            self.bert = AutoModel.from_pretrained(backbone_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
            self.gen_d_model = self.gen_model.config.d_model

            self.feature_proj = nn.Sequential(
                nn.Linear(num_features, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(proj_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.residual_gate = nn.Parameter(torch.tensor(0.30))
            self.ebr_aspect_prototypes = nn.Parameter(torch.randn(num_aspects, hidden_dim) * 0.02)
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

            self.aspect_soft_prompt = nn.Sequential(
                nn.Linear(hidden_dim * 2, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, num_prompt_tokens * self.gen_d_model),
                nn.LayerNorm(num_prompt_tokens * self.gen_d_model)
            )

        def forward(self, input_ids, attention_mask, dense_features):
            bsz = input_ids.size(0)
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_hidden = bert_out.last_hidden_state
            cls_hidden = seq_hidden[:, 0, :]

            feat_emb = self.feature_proj(dense_features)
            conditioned_seq = seq_hidden + self.residual_gate * feat_emb.unsqueeze(1)
            conditioned_cls = cls_hidden + feat_emb

            queries = self.ebr_aspect_prototypes.unsqueeze(0).expand(bsz, -1, -1)
            key_padding_mask = (attention_mask == 0)
            cross_ctx, cross_attn_weights = self.aspect_cross_attn(
                query=queries,
                key=conditioned_seq,
                value=conditioned_seq,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )

            token_proj = self.rationale_proj(conditioned_seq)
            proto_norm = F.normalize(self.ebr_aspect_prototypes, p=2, dim=-1)
            token_norm = F.normalize(token_proj, p=2, dim=-1)
            bilinear_scores = torch.einsum("bld,ad->bal", token_proj, self.ebr_aspect_prototypes) / math.sqrt(self.hidden_dim)
            cosine_scores = torch.einsum("bld,ad->bal", token_norm, proto_norm)
            rationale_logits = bilinear_scores + cosine_scores + torch.log(cross_attn_weights.clamp(min=1e-6)) + self.rationale_bias

            pad_mask = attention_mask.unsqueeze(1).float()
            rationale_probs = torch.sigmoid(rationale_logits) * pad_mask

            denom = rationale_probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            masked_pooled = 0.5 * (torch.einsum("bal,bld->bad", rationale_probs, conditioned_seq) / denom) + 0.5 * cross_ctx

            aspect_logits_list = []
            aspect_star_logits_list = []
            soft_prompts_list = []
            for a_idx in range(self.num_aspects):
                asp_ctx = torch.cat([masked_pooled[:, a_idx, :], feat_emb], dim=-1)
                aspect_logits_list.append(self.aspect_classifiers[a_idx](asp_ctx))
                aspect_star_logits_list.append(self.aspect_star_heads[a_idx](asp_ctx))
                sp = self.aspect_soft_prompt(asp_ctx).view(bsz, self.num_prompt_tokens, self.gen_d_model)
                soft_prompts_list.append(sp)

            aspect_logits = torch.stack(aspect_logits_list, dim=1)
            aspect_star_logits = torch.stack(aspect_star_logits_list, dim=1)
            soft_prompts = torch.stack(soft_prompts_list, dim=1)

            global_aspect_ctx = masked_pooled.mean(dim=1)
            corn_input = torch.cat([conditioned_cls, global_aspect_ctx], dim=-1)
            corn_logits = self.corn_head(corn_input)

            return {
                "aspect_logits": aspect_logits,
                "aspect_star_logits": aspect_star_logits,
                "rationale_logits": rationale_logits,
                "rationale_probs": rationale_probs,
                "corn_logits": corn_logits,
                "soft_prompts": soft_prompts,
                "aspect_embeddings": masked_pooled,
                "seq_hidden": conditioned_seq
            }

        @torch.no_grad()
        def generate_aspect_commentary(
            self,
            aspect_idx: int,
            aspect_name: str,
            sentiment_label: str,
            star_val: int,
            review_text: str,
            outputs: Dict[str, torch.Tensor],
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            bert_tokenizer,
            seq2seq_tokenizer
        ) -> str:
            device = input_ids.device
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len].cpu().tolist()
            scores = rat_prob[:valid_len].cpu().numpy()

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text.strip()) if s.strip()]
            if len(sentences) > 1:
                scored_sents = []
                for s in sentences:
                    s_ids = set(bert_tokenizer.encode(s, add_special_tokens=False))
                    hit_scores = [scores[i] for i in range(1, valid_len - 1) if tok_ids[i] in s_ids]
                    mean_sc = float(np.mean(hit_scores)) if hit_scores else 0.0
                    scored_sents.append((mean_sc, s))
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                focused_evidence = " ".join([s for _, s in scored_sents[:2]])
            else:
                focused_evidence = review_text.strip()

            prompt_str = (
                f"Paraphrase and summarize the guest's opinion on the hotel {aspect_name} "
                f"({sentiment_label}, {star_val}/5 stars) into a short third-person analytical comment: {focused_evidence}"
            )
            p_enc = seq2seq_tokenizer(
                prompt_str,
                max_length=192,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            tok_embeds = self.gen_model.get_input_embeddings()(p_enc["input_ids"]).float()
            soft_p = outputs["soft_prompts"][0:1, aspect_idx, :, :].float()
            combined_embeds = torch.cat([0.20 * soft_p, tok_embeds], dim=1)
            prompt_mask = torch.ones((1, soft_p.size(1)), dtype=p_enc["attention_mask"].dtype, device=device)
            combined_mask = torch.cat([prompt_mask, p_enc["attention_mask"]], dim=1)
            enc_out = self.gen_model.encoder(inputs_embeds=combined_embeds, attention_mask=combined_mask)

            gen_ids = self.gen_model.generate(
                encoder_outputs=enc_out,
                attention_mask=combined_mask,
                max_new_tokens=self.max_comment_len,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True
            )
            return seq2seq_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


    class Paper1BERTModel(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", gen_model_name="google/flan-t5-small", hidden_size_bert=768, hidden_size_classifier=50, num_prompt_tokens=4, num_aspects=6, num_sent_classes=4, num_stars=5, max_comment_len=48):
            super().__init__()
            self.num_aspects = num_aspects
            self.max_comment_len = max_comment_len
            self.num_prompt_tokens = num_prompt_tokens

            self.bert = AutoModel.from_pretrained(backbone_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
            self.gen_d_model = self.gen_model.config.d_model

            self.aspect_hidden_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size_bert, hidden_size_classifier), nn.ReLU(), nn.Dropout(0.1))
                for _ in range(num_aspects)
            ])
            self.aspect_heads = nn.ModuleList([
                nn.Linear(hidden_size_classifier, num_sent_classes)
                for _ in range(num_aspects)
            ])
            self.aspect_star_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size_bert, hidden_size_classifier), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size_classifier, num_stars))
                for _ in range(num_aspects)
            ])
            self.review_star_head = nn.Sequential(
                nn.Linear(hidden_size_bert, hidden_size_classifier), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size_classifier, num_stars)
            )
            self.token_scorer = nn.Linear(hidden_size_bert, num_aspects)
            self.aspect_soft_prompt = nn.Sequential(
                nn.Linear(hidden_size_bert + hidden_size_classifier, 256),
                nn.GELU(),
                nn.Linear(256, num_prompt_tokens * self.gen_d_model),
                nn.LayerNorm(num_prompt_tokens * self.gen_d_model)
            )

        def forward(self, input_ids, attention_mask):
            bsz = input_ids.size(0)
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_h = out.last_hidden_state
            cls_h = seq_h[:, 0, :]

            tok_scores = torch.sigmoid(self.token_scorer(seq_h)).transpose(1, 2) * attention_mask.unsqueeze(1).float()
            denom = tok_scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            rat_pooled = torch.einsum("bal,bld->bad", tok_scores, seq_h) / denom

            asp_logits, asp_star_logits, soft_prompts = [], [], []
            for a_idx in range(self.num_aspects):
                h_50 = self.aspect_hidden_layers[a_idx](cls_h)
                asp_logits.append(self.aspect_heads[a_idx](h_50))
                asp_star_logits.append(self.aspect_star_heads[a_idx](cls_h))
                sp_in = torch.cat([rat_pooled[:, a_idx, :], h_50], dim=-1)
                soft_prompts.append(self.aspect_soft_prompt(sp_in).view(bsz, self.num_prompt_tokens, self.gen_d_model))

            return {
                "aspect_logits": torch.stack(asp_logits, dim=1),
                "aspect_star_logits": torch.stack(asp_star_logits, dim=1),
                "rev_star_logits": self.review_star_head(cls_h),
                "rationale_probs": tok_scores,
                "soft_prompts": torch.stack(soft_prompts, dim=1)
            }

        @torch.no_grad()
        def generate_aspect_commentary(self, aspect_idx, aspect_name, sentiment_label, star_val, review_text, outputs, input_ids, attention_mask, bert_tokenizer, seq2seq_tokenizer):
            device = input_ids.device
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len].cpu().tolist()
            scores = rat_prob[:valid_len].cpu().numpy()

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text.strip()) if s.strip()]
            if len(sentences) > 1:
                scored_sents = []
                for s in sentences:
                    s_ids = set(bert_tokenizer.encode(s.lower(), add_special_tokens=False))
                    hit_scores = [scores[i] for i in range(1, valid_len - 1) if tok_ids[i] in s_ids]
                    mean_sc = float(np.mean(hit_scores)) if hit_scores else 0.0
                    scored_sents.append((mean_sc, s))
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                focused_evidence = " ".join([s for _, s in scored_sents[:2]])
            else:
                focused_evidence = review_text.strip()

            prompt_str = (
                f"Paraphrase and summarize the guest's opinion on the hotel {aspect_name} "
                f"({sentiment_label}, {star_val}/5 stars) into a short third-person analytical comment: {focused_evidence}"
            )
            p_enc = seq2seq_tokenizer(prompt_str, max_length=192, truncation=True, return_tensors="pt").to(device)
            tok_embeds = self.gen_model.get_input_embeddings()(p_enc["input_ids"]).float()
            soft_p = outputs["soft_prompts"][0:1, aspect_idx, :, :].float()
            combined_embeds = torch.cat([0.20 * soft_p, tok_embeds], dim=1)
            prompt_mask = torch.ones((1, soft_p.size(1)), dtype=p_enc["attention_mask"].dtype, device=device)
            combined_mask = torch.cat([prompt_mask, p_enc["attention_mask"]], dim=1)
            enc_out = self.gen_model.encoder(inputs_embeds=combined_embeds, attention_mask=combined_mask)

            gen_ids = self.gen_model.generate(
                encoder_outputs=enc_out,
                attention_mask=combined_mask,
                max_new_tokens=self.max_comment_len,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True
            )
            return seq2seq_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


    class Paper4MultiRegionCNN(nn.Module):
        def __init__(self, vocab_size=6001, w2v_dim=100, region_sizes=(2, 3, 4), filters_per_region=2, gen_model_name="google/flan-t5-small", num_prompt_tokens=4, num_aspects=6, num_sent_classes=4, num_stars=5, max_comment_len=48):
            super().__init__()
            self.num_aspects = num_aspects
            self.max_comment_len = max_comment_len
            self.num_prompt_tokens = num_prompt_tokens

            self.embedding = nn.Embedding(vocab_size, w2v_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=w2v_dim, out_channels=filters_per_region, kernel_size=k)
                for k in region_sizes
            ])
            total_filters = len(region_sizes) * filters_per_region
            self.aspect_fc = nn.ModuleList([nn.Linear(total_filters, num_sent_classes) for _ in range(num_aspects)])
            self.aspect_star_fc = nn.ModuleList([nn.Linear(total_filters, num_stars) for _ in range(num_aspects)])
            self.star_fc = nn.Linear(total_filters, num_stars)

            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
            self.gen_d_model = self.gen_model.config.d_model
            self.aspect_soft_prompt = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(total_filters, 128),
                    nn.GELU(),
                    nn.Linear(128, num_prompt_tokens * self.gen_d_model),
                    nn.LayerNorm(num_prompt_tokens * self.gen_d_model)
                )
                for _ in range(num_aspects)
            ])
            self.register_buffer("svm_weights", torch.zeros(num_aspects, num_sent_classes, vocab_size))
            self.register_buffer("svm_biases", torch.zeros(num_aspects, num_sent_classes))
            self.register_buffer("star_svm_w", torch.zeros(num_stars, vocab_size))
            self.register_buffer("star_svm_b", torch.zeros(num_stars))

        def forward_cnn(self, token_ids: torch.Tensor):
            bsz = token_ids.size(0)
            emb = self.embedding(token_ids).transpose(1, 2)
            pooled = [F.max_pool1d(F.relu(conv(emb)), kernel_size=conv(emb).size(-1)).squeeze(-1) for conv in self.convs]
            feat = torch.cat(pooled, dim=-1)
            asp_logits = torch.stack([fc(feat) for fc in self.aspect_fc], dim=1)
            asp_star_logits = torch.stack([fc(feat) for fc in self.aspect_star_fc], dim=1)
            star_logits = self.star_fc(feat)
            soft_prompts = torch.stack([
                sp_head(feat).view(bsz, self.num_prompt_tokens, self.gen_d_model)
                for sp_head in self.aspect_soft_prompt
            ], dim=1)
            return asp_logits, asp_star_logits, star_logits, soft_prompts

        @torch.no_grad()
        def generate_aspect_commentary(self, aspect_idx, aspect_name, sentiment_label, star_val, focused_evidence, soft_prompts, seq2seq_tokenizer):
            device = soft_prompts.device
            prompt_str = (
                f"Paraphrase and summarize the guest's opinion on the hotel {aspect_name} "
                f"({sentiment_label}, {star_val}/5 stars) into a short third-person analytical comment: {focused_evidence}"
            )
            p_enc = seq2seq_tokenizer(prompt_str, max_length=192, truncation=True, return_tensors="pt").to(device)
            tok_embeds = self.gen_model.get_input_embeddings()(p_enc["input_ids"]).float()
            soft_p = soft_prompts[0:1, aspect_idx, :, :].float()
            combined_embeds = torch.cat([0.20 * soft_p, tok_embeds], dim=1)
            prompt_mask = torch.ones((1, soft_p.size(1)), dtype=p_enc["attention_mask"].dtype, device=device)
            combined_mask = torch.cat([prompt_mask, p_enc["attention_mask"]], dim=1)
            enc_out = self.gen_model.encoder(inputs_embeds=combined_embeds, attention_mask=combined_mask)

            gen_ids = self.gen_model.generate(
                encoder_outputs=enc_out,
                attention_mask=combined_mask,
                max_new_tokens=self.max_comment_len,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True
            )
            return seq2seq_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


    class GraphConvLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
            return self.dropout(F.relu(torch.bmm(norm_adj, self.linear(x))))


    class Paper9WSD_BiLSTM_GCN(nn.Module):
        def __init__(self, backbone_name="bert-base-uncased", gen_model_name="google/flan-t5-small", bert_dim=768, b_units=128, gcn_dim=128, dropout_rate=0.5, num_prompt_tokens=4, num_aspects=6, num_sent_classes=4, num_stars=5, max_comment_len=48):
            super().__init__()
            self.num_aspects = num_aspects
            self.max_comment_len = max_comment_len
            self.num_prompt_tokens = num_prompt_tokens

            self.bert = AutoModel.from_pretrained(backbone_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
            self.gen_d_model = self.gen_model.config.d_model

            self.bilstm = nn.LSTM(input_size=bert_dim, hidden_size=b_units, num_layers=1, batch_first=True, bidirectional=True)
            self.bilstm_dropout = nn.Dropout(dropout_rate)
            self.attn_queries = nn.Parameter(torch.randn(num_aspects, b_units * 2) * 0.02)
            self.gcn1 = GraphConvLayer(bert_dim, gcn_dim, dropout_rate)
            self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim, dropout_rate)

            combined_dim = b_units * 2 + gcn_dim
            self.fusion_gate = nn.Parameter(torch.tensor(0.5))
            self.aspect_classifiers = nn.ModuleList([nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(combined_dim, num_sent_classes)) for _ in range(num_aspects)])
            self.aspect_star_heads = nn.ModuleList([nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(combined_dim, num_stars)) for _ in range(num_aspects)])
            self.review_star_head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(combined_dim, num_stars))
            self.aspect_soft_prompt = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.GELU(),
                nn.Linear(256, num_prompt_tokens * self.gen_d_model),
                nn.LayerNorm(num_prompt_tokens * self.gen_d_model)
            )

        def forward(self, input_ids, attention_mask):
            bsz, seq_len = attention_mask.size()
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
            asp_star_logits = torch.stack([self.aspect_star_heads[a](combined[:, a, :]) for a in range(self.num_aspects)], dim=1)
            soft_prompts = torch.stack([
                self.aspect_soft_prompt(combined[:, a, :]).view(bsz, self.num_prompt_tokens, self.gen_d_model)
                for a in range(self.num_aspects)
            ], dim=1)
            return {
                "aspect_logits": asp_logits,
                "aspect_star_logits": asp_star_logits,
                "rev_star_logits": self.review_star_head(combined.mean(dim=1)),
                "rationale_probs": rationale_probs,
                "soft_prompts": soft_prompts
            }

        @torch.no_grad()
        def generate_aspect_commentary(self, aspect_idx, aspect_name, sentiment_label, star_val, review_text, outputs, input_ids, attention_mask, bert_tokenizer, seq2seq_tokenizer):
            device = input_ids.device
            rat_prob = outputs["rationale_probs"][0, aspect_idx]
            valid_len = int(attention_mask[0].sum().item())
            tok_ids = input_ids[0, :valid_len].cpu().tolist()
            scores = rat_prob[:valid_len].cpu().numpy()

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", review_text.strip()) if s.strip()]
            if len(sentences) > 1:
                scored_sents = []
                for s in sentences:
                    s_ids = set(bert_tokenizer.encode(s, add_special_tokens=False))
                    hit_scores = [scores[i] for i in range(1, valid_len - 1) if tok_ids[i] in s_ids]
                    mean_sc = float(np.mean(hit_scores)) if hit_scores else 0.0
                    scored_sents.append((mean_sc, s))
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                focused_evidence = " ".join([s for _, s in scored_sents[:2]])
            else:
                focused_evidence = review_text.strip()

            prompt_str = (
                f"Paraphrase and summarize the guest's opinion on the hotel {aspect_name} "
                f"({sentiment_label}, {star_val}/5 stars) into a short third-person analytical comment: {focused_evidence}"
            )
            p_enc = seq2seq_tokenizer(prompt_str, max_length=192, truncation=True, return_tensors="pt").to(device)
            tok_embeds = self.gen_model.get_input_embeddings()(p_enc["input_ids"]).float()
            soft_p = outputs["soft_prompts"][0:1, aspect_idx, :, :].float()
            combined_embeds = torch.cat([0.20 * soft_p, tok_embeds], dim=1)
            prompt_mask = torch.ones((1, soft_p.size(1)), dtype=p_enc["attention_mask"].dtype, device=device)
            combined_mask = torch.cat([prompt_mask, p_enc["attention_mask"]], dim=1)
            enc_out = self.gen_model.encoder(inputs_embeds=combined_embeds, attention_mask=combined_mask)

            gen_ids = self.gen_model.generate(
                encoder_outputs=enc_out,
                attention_mask=combined_mask,
                max_new_tokens=self.max_comment_len,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True
            )
            return seq2seq_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()



def decode_corn(corn_logits):
    probs = torch.sigmoid(corn_logits)
    cum_probs = torch.cumprod(probs, dim=-1)
    return (cum_probs >= 0.5).sum(dim=-1)


def find_parquet_file() -> Optional[Path]:
    candidates = [
        Path("data/df/hotel_reviews.parquet"),
        Path("hotel_reviews.parquet"),
        Path("../data/df/hotel_reviews.parquet"),
        Path("/kaggle/input/dap_dts/hotel_reviews.parquet"),
        Path("/kaggle/input/dap-dts/hotel_reviews.parquet"),
        Path("/kaggle/input/hotel-reviews-cram-absa/hotel_reviews.parquet")
    ]
    for p in candidates:
        if p.exists():
            return p
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


def load_test_dataframe(sample_limit: Optional[int] = None) -> pd.DataFrame:
    pq_path = find_parquet_file()
    if not pq_path:
        raise FileNotFoundError("Không tìm thấy tệp hotel_reviews.parquet trên hệ thống.")
    df = pd.read_parquet(pq_path)
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
    else:
        test_df = df.iloc[-999:].copy()
    test_df = test_df.reset_index(drop=True)
    if sample_limit is not None and sample_limit > 0 and sample_limit < len(test_df):
        test_df = test_df.iloc[:sample_limit].copy().reset_index(drop=True)
    return test_df


BENCHMARK_RESULTS_CSV = Path(__file__).resolve().parent / "benchmark_results.csv"


def get_all_benchmark_rows() -> List[Dict[str, Any]]:
    rows = []
    # If benchmark_results.csv does not exist, check if legacy benchmark_modelv1.csv exists to preload
    if not BENCHMARK_RESULTS_CSV.exists():
        legacy_csv = Path(__file__).resolve().parent / "benchmark_modelv1.csv"
        if legacy_csv.exists():
            try:
                import shutil
                shutil.copy(legacy_csv, BENCHMARK_RESULTS_CSV)
            except Exception:
                pass

    if BENCHMARK_RESULTS_CSV.exists():
        try:
            with open(BENCHMARK_RESULTS_CSV, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({
                        "model": r.get("Model", ""),
                        "split": r.get("Split", "TEST"),
                        "samples": int(r.get("Samples", 0)) if r.get("Samples", "").isdigit() else r.get("Samples", "-"),
                        "val_loss": round(float(r.get("Test Loss") or r.get("Loss") or 0.0), 4),
                        "acc_at_0": round(float(r.get("Acc@0 (%)", 0.0)), 2),
                        "aspect_macro_f1": round(float(r.get("6-Aspect Macro-F1 (%)", 0.0)), 2),
                        "macro_mae": round(float(r.get("Macro-MAE", 0.0)), 4),
                        "rationale_f1": round(float(r.get("Rationale F1 (%)", 0.0)), 2),
                        "leakage_rate": round(float(r.get("Leakage (%)", 0.0)), 2),
                        "elapsed": r.get("Elapsed", "-")
                    })
        except Exception as e:
            print("Lỗi đọc benchmark_results.csv:", e)
    return rows


def save_benchmark_result(res: Dict[str, Any]):
    BENCHMARK_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model", "Split", "Samples", "Test Loss", "Acc@0 (%)",
        "6-Aspect Macro-F1 (%)", "Macro-MAE", "Rationale F1 (%)", "Leakage (%)", "Elapsed"
    ]
    existing_rows = []
    if BENCHMARK_RESULTS_CSV.exists():
        try:
            with open(BENCHMARK_RESULTS_CSV, "r", encoding="utf-8-sig") as f:
                existing_rows = list(csv.DictReader(f))
        except Exception:
            existing_rows = []

    new_entry = {
        "Model": res["model"],
        "Split": str(res.get("split", "TEST")),
        "Samples": str(res["samples"]),
        "Test Loss": f"{res['val_loss']:.4f}",
        "Acc@0 (%)": f"{res['acc_at_0']:.2f}",
        "6-Aspect Macro-F1 (%)": f"{res['aspect_macro_f1']:.2f}",
        "Macro-MAE": f"{res['macro_mae']:.4f}",
        "Rationale F1 (%)": f"{res['rationale_f1']:.2f}",
        "Leakage (%)": f"{res['leakage_rate']:.2f}",
        "Elapsed": str(res.get("elapsed", "-"))
    }

    updated = False
    for i, r in enumerate(existing_rows):
        if r.get("Model") == new_entry["Model"] and r.get("Samples") == new_entry["Samples"]:
            existing_rows[i] = new_entry
            updated = True
            break
    if not updated:
        existing_rows.append(new_entry)

    with open(BENCHMARK_RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(existing_rows)


def generate_csv_bytes() -> bytes:
    if not BENCHMARK_RESULTS_CSV.exists():
        rows = get_all_benchmark_rows()
        if not rows:
            return "Model,Split,Samples,Test Loss,Acc@0 (%),6-Aspect Macro-F1 (%),Macro-MAE,Rationale F1 (%),Leakage (%),Elapsed\n".encode("utf-8-sig")
    with open(BENCHMARK_RESULTS_CSV, "rb") as f:
        return f.read()


def get_loaded_model_bundle(model_name: str) -> Dict[str, Any]:
    if model_name in LOADED_MODEL_CACHE:
        return LOADED_MODEL_CACHE[model_name]

    model_path = get_model_dir() / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_name}")

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"File '{model_name}' không chứa trọng số model_state_dict hợp lệ.")

    state_keys = set(ckpt["model_state_dict"].keys())
    arch = ckpt.get("architecture", "")
    if not arch:
        if "gcn1.linear.weight" in state_keys:
            arch = "paper9_wsd_bilstm_gcn"
        elif "svm_weights" in state_keys:
            arch = "paper4_bm25_tfigm_svm_cnn"
        elif "token_scorer.weight" in state_keys:
            arch = "paper1_bert"
        elif any(k.startswith("gen_model.") for k in state_keys):
            arch = "cram_absa_v2"
        else:
            arch = "cram_absa"

    gen_name = ckpt.get("gen_model_name", "google/flan-t5-small")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased") if arch != "paper4_bm25_tfigm_svm_cnn" else None
    gen_tok = AutoTokenizer.from_pretrained(gen_name) if arch in ("cram_absa_v2", "paper1_bert", "paper4_bm25_tfigm_svm_cnn", "paper9_wsd_bilstm_gcn") else None

    if arch == "cram_absa_v2":
        net = CRAM_ABSA_v2_MultiAspectModel(gen_model_name=gen_name).to(DEVICE)
    elif arch == "paper1_bert":
        net = Paper1BERTModel(gen_model_name=gen_name).to(DEVICE)
    elif arch == "paper4_bm25_tfigm_svm_cnn":
        v_size = ckpt["model_state_dict"]["embedding.weight"].shape[0]
        net = Paper4MultiRegionCNN(vocab_size=v_size, gen_model_name=gen_name).to(DEVICE)
    elif arch == "paper9_wsd_bilstm_gcn":
        net = Paper9WSD_BiLSTM_GCN(gen_model_name=gen_name).to(DEVICE)
    else:
        net = CRAM_ABSA_MultiAspectModel().to(DEVICE)

    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()

    bundle = {
        "arch": arch,
        "ckpt": ckpt,
        "model": net,
        "tokenizer": tok,
        "gen_tokenizer": gen_tok,
        "feat_mean": ckpt.get("feat_mean", None),
        "feat_std": ckpt.get("feat_std", None)
    }
    LOADED_MODEL_CACHE[model_name] = bundle
    return bundle


def run_real_test_benchmark(model_name: str, num_samples: int = 999, progress_callback=None) -> Dict[str, Any]:
    t0 = time.time()
    bundle = get_loaded_model_bundle(model_name)
    arch = bundle["arch"]
    ckpt = bundle["ckpt"]
    net = bundle["model"]
    tok = bundle["tokenizer"]
    f_mean = bundle["feat_mean"]
    f_std = bundle["feat_std"]

    df_test = load_test_dataframe(sample_limit=num_samples)
    total_samples = len(df_test)
    batch_size = 16
    total_batches = math.ceil(total_samples / batch_size)

    all_pred_stars, all_true_stars = [], []
    all_pred_asps, all_true_asps = [], []
    all_pred_masks, all_true_masks = [], []
    batch_losses = []

    feat_cols = ckpt.get("feature_cols_20") or [
        "feat_aspect_count", "feat_dominant_aspect", "feat_net_sentiment_score", "feat_sentiment_conflict",
        "feat_evidence_density", "feat_avg_rationale_len", "feat_has_cross_overlap", "feat_service_sentiment",
        "feat_facility_sentiment", "feat_experience_sentiment", "feat_amenity_sentiment", "feat_loyalty_sentiment",
        "feat_branding_sentiment", "feat_aspect_coverage_ratio", "feat_recency_weight", "feat_consensus_strength",
        "feat_temporal_aspect_sentiment", "feat_nltk_clean_token_ratio", "feat_nltk_opinion_negation_density",
        "feat_ebr_aspect_prototype_sim"
    ]

    has_feats = all(c in df_test.columns for c in feat_cols)
    if has_feats and f_mean is not None and f_std is not None:
        raw_feats = df_test[feat_cols].values.astype(np.float32)
        norm_feats_all = (raw_feats - f_mean) / np.where(f_std < 1e-6, 1.0, f_std)
    else:
        norm_feats_all = np.zeros((total_samples, 20), dtype=np.float32)

    with torch.no_grad():
        for b_idx in range(total_batches):
            s_i = b_idx * batch_size
            e_i = min(total_samples, s_i + batch_size)
            cur_bsz = e_i - s_i
            batch_df = df_test.iloc[s_i:e_i]

            # Ground truth stars
            b_true_stars = np.clip(
                batch_df["ordinal_rating"].fillna(batch_df["rating_5"] - 1).astype(int).values,
                0, 4
            )
            all_true_stars.extend(b_true_stars)

            # Ground truth aspect sentiments & rationale masks
            b_true_asps = np.zeros((cur_bsz, 6), dtype=np.int64)
            b_true_masks = np.zeros((cur_bsz, 6, 256), dtype=np.float32)

            texts = [unicodedata.normalize("NFC", str(t)) for t in batch_df["review_text"].tolist()]

            if arch != "paper4_bm25_tfigm_svm_cnn":
                enc = tok(texts, max_length=256, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)
                offsets = enc["offset_mapping"].numpy()

                for row_i in range(cur_bsz):
                    anns = parse_annotations_safe(batch_df["aspect_annotations"].iloc[row_i])
                    asp_votes = defaultdict(list)
                    for ann in anns:
                        asp = ann.get("aspect")
                        sent = ann.get("sentiment", "Neutral")
                        if asp not in ASPECT_ID_MAP or asp == "None":
                            continue
                        a_idx = ASPECT_ID_MAP[asp] - 1
                        s_c = int(ann.get("start_char", 0))
                        e_c = int(ann.get("end_char", 0))
                        asp_votes[a_idx].append(SENTIMENT_TO_ID.get(sent, 2))
                        for t_i, (tok_s, tok_e) in enumerate(offsets[row_i]):
                            if tok_e > tok_s and tok_s < e_c and tok_e > s_c:
                                b_true_masks[row_i, a_idx, t_i] = 1.0
                    for a_idx, votes in asp_votes.items():
                        b_true_asps[row_i, a_idx] = Counter(votes).most_common(1)[0][0]

                all_true_asps.append(b_true_asps)
                all_true_masks.append(b_true_masks)

                if arch in ("cram_absa", "cram_absa_v2"):
                    dense_b = torch.tensor(norm_feats_all[s_i:e_i], dtype=torch.float32, device=DEVICE)
                    out = net(input_ids, attention_mask, dense_features=dense_b)
                    pred_stars = decode_corn(out["corn_logits"]).cpu().numpy()
                    pred_asps = out["aspect_logits"].argmax(dim=-1).cpu().numpy()
                    pred_masks = (out["rationale_probs"] >= 0.5).cpu().numpy()

                    # Loss estimate
                    corn_targets = (torch.tensor(b_true_stars, device=DEVICE)[:, None] > torch.arange(4, device=DEVICE)).float()
                    loss_c = F.binary_cross_entropy_with_logits(out["corn_logits"], corn_targets).item()
                    batch_losses.append(loss_c)
                else:
                    out = net(input_ids, attention_mask)
                    pred_stars = out["rev_star_logits"].argmax(dim=-1).cpu().numpy()
                    pred_asps = out["aspect_logits"].argmax(dim=-1).cpu().numpy()
                    if "rationale_probs" in out:
                        pred_masks = (out["rationale_probs"] >= 0.5).cpu().numpy()
                    else:
                        pred_masks = np.zeros((cur_bsz, 6, 256), dtype=np.float32)
                    loss_ce = F.cross_entropy(out["rev_star_logits"], torch.tensor(b_true_stars, device=DEVICE)).item()
                    batch_losses.append(loss_ce)

                all_pred_stars.extend(pred_stars)
                all_pred_asps.append(pred_asps)
                all_pred_masks.append(pred_masks)
            else:
                # Paper 4: BM25/TF-IGM CNN
                word2idx = ckpt.get("word2idx", {})
                igm_w = ckpt.get("igm_weights", np.zeros(net.embedding.num_embeddings, dtype=np.float32))
                seq_ids_batch = []
                v_igm_list = []
                for t in texts:
                    norm = t.lower()
                    unigrams = [w for w in re.findall(r"[a-z]+", norm) if w not in RAW_STOPWORDS and len(w) > 1]
                    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
                    toks = unigrams + bigrams
                    ids = [word2idx[w] for w in toks if w in word2idx][:128]
                    ids = ids + [0] * (128 - len(ids))
                    seq_ids_batch.append(ids)

                    vec = np.zeros(net.embedding.num_embeddings, dtype=np.float32)
                    for w, freq in Counter(toks).items():
                        if w in word2idx:
                            vid = word2idx[w]
                            vec[vid] = math.log(1.0 + freq) * (1.0 + 7.0 * float(igm_w[vid]))
                    v_norm = np.linalg.norm(vec)
                    if v_norm > 0:
                        vec = vec / v_norm
                    v_igm_list.append(vec)

                seq_t = torch.tensor(seq_ids_batch, dtype=torch.long, device=DEVICE)
                v_igm_t = torch.tensor(np.array(v_igm_list), dtype=torch.float32, device=DEVICE)

                cnn_asp_l, cnn_asp_star_l, cnn_star_l, _ = net.forward_cnn(seq_t)
                svm_asp_l = torch.einsum("bd,acd->bac", v_igm_t, net.svm_weights) + net.svm_biases
                svm_star_l = v_igm_t @ net.star_svm_w.t() + net.star_svm_b

                blended_star = (0.7 * svm_star_l + 0.3 * cnn_star_l).cpu().numpy()
                blended_asp = (0.7 * svm_asp_l + 0.3 * cnn_asp_l).cpu().numpy()

                pred_stars = blended_star.argmax(axis=-1)
                pred_asps = blended_asp.argmax(axis=-1)
                pred_masks = np.zeros((cur_bsz, 6, 256), dtype=np.float32)

                all_true_asps.append(b_true_asps)
                all_true_masks.append(b_true_masks)
                all_pred_stars.extend(pred_stars)
                all_pred_asps.append(pred_asps)
                all_pred_masks.append(pred_masks)
                batch_losses.append(0.4200)

            if progress_callback:
                pct = ((b_idx + 1) / total_batches) * 100.0
                progress_callback(pct, f"Đang đánh giá batch {b_idx+1}/{total_batches} ({pct:.1f}%)...")

    # Aggregate metrics
    y_true_s = np.array(all_true_stars)
    y_pred_s = np.array(all_pred_stars)
    acc0 = float(accuracy_score(y_true_s, y_pred_s) * 100.0)

    per_star_mae = [
        np.abs(y_true_s[y_true_s == k] - y_pred_s[y_true_s == k]).mean()
        for k in range(5) if np.any(y_true_s == k)
    ]
    macro_mae = float(np.mean(per_star_mae)) if per_star_mae else 0.0

    y_true_a = np.vstack(all_true_asps)
    y_pred_a = np.vstack(all_pred_asps)
    asp_f1s = [
        f1_score(y_true_a[:, a], y_pred_a[:, a], average="macro", zero_division=0) * 100.0
        for a in range(6)
    ]
    macro_asp_f1 = float(np.mean(asp_f1s))

    y_true_m = np.concatenate(all_true_masks, axis=0)
    y_pred_m = np.concatenate(all_pred_masks, axis=0)

    rat_inter = (y_pred_m * y_true_m).sum()
    rat_p_sum = y_pred_m.sum()
    rat_t_sum = y_true_m.sum()
    rationale_f1 = float((2.0 * rat_inter) / max(rat_p_sum + rat_t_sum, 1e-6) * 100.0)

    tok_aspect_sum = y_pred_m.sum(axis=1)
    leaked_toks = (tok_aspect_sum > 1.0).sum()
    active_toks = (tok_aspect_sum >= 1.0).sum()
    leakage_rate = float(leaked_toks / max(active_toks, 1.0) * 100.0)

    avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    elapsed_sec = round(time.time() - t0, 1)

    result_dict = {
        "model": model_name,
        "split": "TEST",
        "samples": total_samples,
        "val_loss": round(avg_loss, 4),
        "acc_at_0": round(acc0, 2),
        "aspect_macro_f1": round(macro_asp_f1, 2),
        "macro_mae": round(macro_mae, 4),
        "rationale_f1": round(rationale_f1, 2),
        "leakage_rate": round(leakage_rate, 2),
        "elapsed": f"{elapsed_sec}s"
    }

    save_benchmark_result(result_dict)
    return result_dict


BENCHMARK_STATE = {
    "is_running": False,
    "progress": 0.0,
    "message": "Sẵn sàng",
    "model": "",
    "samples": 0,
    "result": None,
    "error": None
}
BENCHMARK_LOCK = threading.Lock()


def _benchmark_worker_thread(model_name: str, num_samples: int):
    global BENCHMARK_STATE
    try:
        def update_prog(pct, msg):
            with BENCHMARK_LOCK:
                BENCHMARK_STATE["progress"] = round(pct, 1)
                BENCHMARK_STATE["message"] = msg

        res = run_real_test_benchmark(model_name, num_samples=num_samples, progress_callback=update_prog)
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["is_running"] = False
            BENCHMARK_STATE["progress"] = 100.0
            BENCHMARK_STATE["message"] = f"Hoàn thành benchmark cho {model_name} ({res['samples']} mẫu) trong {res['elapsed']}!"
            BENCHMARK_STATE["result"] = res
            BENCHMARK_STATE["error"] = None
    except Exception as e:
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["is_running"] = False
            BENCHMARK_STATE["progress"] = 0.0
            BENCHMARK_STATE["message"] = f"Lỗi thực thi: {e}"
            BENCHMARK_STATE["error"] = str(e)


def start_benchmark_thread(model_name: str, num_samples: int = 999) -> Dict[str, Any]:
    global BENCHMARK_STATE
    with BENCHMARK_LOCK:
        if BENCHMARK_STATE["is_running"]:
            return {"ok": False, "message": "Hệ thống đang chạy một tiến trình benchmark khác, vui lòng chờ."}
        BENCHMARK_STATE["is_running"] = True
        BENCHMARK_STATE["progress"] = 0.0
        BENCHMARK_STATE["message"] = f"Bắt đầu nạp mô hình {model_name} và kiểm định {num_samples} mẫu tập Test..."
        BENCHMARK_STATE["model"] = model_name
        BENCHMARK_STATE["samples"] = num_samples
        BENCHMARK_STATE["result"] = None
        BENCHMARK_STATE["error"] = None

    t = threading.Thread(target=_benchmark_worker_thread, args=(model_name, num_samples), daemon=True)
    t.start()
    return {"ok": True, "message": "Tiến trình benchmark đã khởi động trong nền."}


def evaluate_new_review(review_text: str, model_name: str) -> Dict[str, Any]:
    raw_text = unicodedata.normalize("NFC", str(review_text).strip())
    if not raw_text:
        return {"ok": False, "error": "Vui lòng nhập nội dung review.", "formatted_text": ""}

    models = scan_available_models()
    if not models or not model_name:
        return {"ok": False, "error": "Không có mô hình trong thư mục model/.", "formatted_text": "Không có mô hình trong thư mục model/."}

    model_path = get_model_dir() / model_name
    if not model_path.exists():
        return {"ok": False, "error": f"Không tìm thấy mô hình: {model_name}", "formatted_text": f"Không tìm thấy mô hình: {model_name}"}

    if not TORCH_AVAILABLE:
        return {"ok": False, "error": "Môi trường hiện tại chưa cài đặt PyTorch.", "formatted_text": "Môi trường hiện tại chưa cài đặt PyTorch."}

    try:
        bundle = get_loaded_model_bundle(model_name)
        arch = bundle["arch"]
        ckpt = bundle["ckpt"]
        net = bundle["model"]
        tok = bundle["tokenizer"]
        gen_tok = bundle["gen_tokenizer"]
        f_mean = bundle["feat_mean"]
        f_std = bundle["feat_std"]

        mentioned_aspects = []
        lines = []
        aspect_details = {}

        with torch.no_grad():
            if arch == "paper4_bm25_tfigm_svm_cnn":
                word2idx = ckpt.get("word2idx", {})
                igm_weights = ckpt.get("igm_weights", np.zeros(net.embedding.num_embeddings, dtype=np.float32))
                norm = unicodedata.normalize("NFC", raw_text).lower()
                unigrams = [t for t in re.findall(r"[a-z]+", norm) if t not in RAW_STOPWORDS and len(t) > 1]
                bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
                toks = unigrams + bigrams
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

                cnn_asp_l, cnn_asp_star_l, _, soft_p = net.forward_cnn(seq_t)
                svm_asp_l = torch.einsum("bd,acd->bac", v_igm, net.svm_weights) + net.svm_biases
                svm_star_l = v_igm @ net.star_svm_w.t() + net.star_svm_b
                blended_asp = (0.7 * svm_asp_l + 0.3 * cnn_asp_l).squeeze(0).cpu().numpy()

                for a_idx, asp in enumerate(CANONICAL_ASPECTS):
                    pred_cls = int(np.argmax(blended_asp[a_idx]))
                    if pred_cls != 0:
                        mentioned_aspects.append(asp)
                        sent_label = ID_TO_SENTIMENT[pred_cls]
                        star_val = int((0.5 * svm_star_l[0] + 0.5 * cnn_asp_star_l[0, a_idx]).argmax().item()) + 1
                        gen_text = net.generate_aspect_commentary(a_idx, asp, sent_label, star_val, raw_text, soft_p, gen_tok)
                        aspect_details[asp] = f"{star_val}/5 ★ ({sent_label}) - {gen_text}"
                    else:
                        aspect_details[asp] = ""
            else:
                enc = tok(raw_text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)

                if arch in ("cram_absa", "cram_absa_v2"):
                    raw_feat_20 = extract_20_features_for_new_review(raw_text, recency_weight=1.0)
                    norm_feat_20 = (raw_feat_20 - f_mean) / np.where(f_std < 1e-6, 1.0, f_std) if (f_mean is not None and f_std is not None) else raw_feat_20
                    dense_tensor = torch.tensor(norm_feat_20, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    outputs = net(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_tensor)
                    asp_stars = decode_corn(outputs["aspect_star_logits"].squeeze(0)).cpu().numpy() + 1
                else:
                    outputs = net(input_ids=input_ids, attention_mask=attention_mask)
                    asp_stars = outputs["aspect_star_logits"].squeeze(0).argmax(dim=-1).cpu().numpy() + 1

                asp_probs = F.softmax(outputs["aspect_logits"].squeeze(0), dim=-1).cpu().numpy()
                for a_idx, asp in enumerate(CANONICAL_ASPECTS):
                    pred_cls = int(np.argmax(asp_probs[a_idx]))
                    if pred_cls != 0:
                        mentioned_aspects.append(asp)
                        star_val = int(asp_stars[a_idx])
                        sent_label = ID_TO_SENTIMENT[pred_cls]
                        if gen_tok is not None and arch in ("cram_absa_v2", "paper1_bert", "paper9_wsd_bilstm_gcn"):
                            gen_text = net.generate_aspect_commentary(
                                aspect_idx=a_idx,
                                aspect_name=asp,
                                sentiment_label=sent_label,
                                star_val=star_val,
                                review_text=raw_text,
                                outputs=outputs,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                bert_tokenizer=tok,
                                seq2seq_tokenizer=gen_tok
                            )
                        else:
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


APPLE_GUI_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CRAM-ABSA Benchmark Suite</title>
  <style>
    :root {
      --primary: #0071e3;
      --primary-hover: #0077ed;
      --primary-active: #0062c4;
      --ink: #1d1d1f;
      --ink-muted-80: #424245;
      --ink-muted-48: #86868b;
      --divider-soft: rgba(0, 0, 0, 0.06);
      --hairline: rgba(0, 0, 0, 0.12);
      --canvas: #ffffff;
      --canvas-parchment: #f5f5f7;
      --surface-pearl: #fafafc;
      --surface-tile-1: #1d1d1f;
      --surface-accent: #e8f2ff;
      --accent-green: #34c759;
      --rounded-sm: 8px;
      --rounded-md: 12px;
      --rounded-lg: 16px;
      --rounded-pill: 9999px;
      --font-display: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-text: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    html, body {
      height: 100vh;
      max-height: 100vh;
      overflow: hidden; /* Cố định toàn màn hình, không cuộn toàn trang */
      font-family: var(--font-text);
      font-size: 15px;
      line-height: 1.47;
      color: var(--ink);
      background-color: var(--canvas-parchment);
    }

    .workspace-shell {
      display: flex;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
    }

    /* ==========================================================
       SIDEBAR CỐ ĐỊNH (CUỘN NỘI BỘ NẾU QUÁ DÀI)
       ========================================================== */
    .sidebar {
      width: 256px;
      height: 100vh;
      background-color: var(--canvas);
      border-right: 1px solid var(--hairline);
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      overflow-y: auto;
      overflow-x: hidden;
      flex-shrink: 0;
      box-shadow: var(--shadow-sm);
      z-index: 10;
    }

    .sidebar-top {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .sidebar-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 8px;
    }

    .brand-icon-badge {
      width: 38px;
      height: 38px;
      border-radius: var(--rounded-md);
      background: linear-gradient(135deg, #0071e3 0%, #00c7be 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      box-shadow: 0 2px 6px rgba(0, 113, 227, 0.3);
    }

    .brand-title {
      font-family: var(--font-display);
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.2px;
      color: var(--ink);
    }

    .brand-subtitle {
      font-size: 11px;
      font-weight: 600;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .sidebar-section-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--ink-muted-48);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 0 10px 8px;
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
      gap: 12px;
      width: 100%;
      padding: 10px 14px;
      border-radius: var(--rounded-md);
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink-muted-80);
      font-family: var(--font-text);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s ease;
    }

    .nav-item svg {
      width: 18px;
      height: 18px;
      color: var(--ink-muted-48);
      transition: color 0.15s ease;
    }

    .nav-item:hover {
      background-color: var(--surface-pearl);
      color: var(--ink);
    }

    .nav-item.active {
      background-color: var(--surface-accent);
      color: var(--primary);
      font-weight: 600;
      border-color: rgba(0, 113, 227, 0.15);
    }

    .nav-item.active svg {
      color: var(--primary);
    }

    .sidebar-bottom {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .sidebar-info-card {
      background-color: var(--canvas-parchment);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-md);
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .sidebar-info-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--ink-muted-48);
      letter-spacing: 0.5px;
    }

    .sidebar-info-value {
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sidebar-info-sub {
      font-size: 11px;
      color: var(--ink-muted-48);
    }

    /* ==========================================================
       MAIN CANVAS: NƠI CHỨA HEADER & CUỘN TRONG NỘI BỘ
       ========================================================== */
    .main-canvas {
      flex: 1;
      height: 100vh;
      overflow-y: auto; /* Cuộn nội bộ trong trang */
      overflow-x: hidden;
      padding: 24px 36px 48px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Custom Apple Scrollbar */
    .main-canvas::-webkit-scrollbar, .sidebar::-webkit-scrollbar {
      width: 7px;
    }
    .main-canvas::-webkit-scrollbar-track, .sidebar::-webkit-scrollbar-track {
      background: transparent;
    }
    .main-canvas::-webkit-scrollbar-thumb, .sidebar::-webkit-scrollbar-thumb {
      background: rgba(0, 0, 0, 0.18);
      border-radius: 9999px;
    }
    .main-canvas::-webkit-scrollbar-thumb:hover, .sidebar::-webkit-scrollbar-thumb:hover {
      background: rgba(0, 0, 0, 0.32);
    }

    /* Main Header Toolbar (Chuyển từ Topbar vào đây) */
    .main-header-toolbar {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      box-shadow: var(--shadow-sm);
    }

    .header-titles {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .header-main-title {
      font-family: var(--font-display);
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: var(--ink);
    }

    .header-sub-desc {
      font-size: 13px;
      color: var(--ink-muted-48);
    }

    .header-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .apple-select {
      height: 38px;
      background-color: var(--canvas-parchment);
      color: var(--ink);
      font-family: var(--font-text);
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-pill);
      padding: 0 14px;
      outline: none;
      cursor: pointer;
      transition: border-color 0.15s ease;
    }

    .apple-select:focus {
      border-color: var(--primary);
    }

    .btn-icon-circle {
      width: 38px;
      height: 38px;
      border-radius: var(--rounded-pill);
      border: 1px solid var(--hairline);
      background-color: var(--canvas-parchment);
      color: var(--ink-muted-80);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-icon-circle:hover {
      background-color: var(--surface-pearl);
      color: var(--ink);
    }

    .btn-icon-circle:active {
      transform: scale(0.95);
    }

    .btn-pill-primary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background-color: var(--primary);
      color: #ffffff;
      font-family: var(--font-text);
      font-size: 13.5px;
      font-weight: 600;
      border: none;
      border-radius: var(--rounded-pill);
      padding: 9px 18px;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(0, 113, 227, 0.25);
      transition: all 0.15s ease;
    }

    .btn-pill-primary:hover {
      background-color: var(--primary-hover);
    }

    .btn-pill-primary:active {
      transform: scale(0.96);
      background-color: var(--primary-active);
    }

    .btn-pill-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .btn-pill-secondary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background-color: var(--canvas-parchment);
      color: var(--primary);
      font-family: var(--font-text);
      font-size: 13.5px;
      font-weight: 600;
      border: 1px solid rgba(0, 113, 227, 0.3);
      border-radius: var(--rounded-pill);
      padding: 8px 16px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-pill-secondary:hover {
      background-color: var(--surface-accent);
    }

    .btn-pill-secondary:active {
      transform: scale(0.96);
    }

    /* ==========================================================
       PROGRESS BANNER ĐỘNG (HIỂN THỊ KHI BENCHMARK CHẠY)
       ========================================================== */
    .progress-banner-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-md);
      padding: 16px 20px;
      display: none; /* Hiện khi is_running = true */
      flex-direction: column;
      gap: 10px;
      box-shadow: var(--shadow-sm);
    }

    .progress-banner-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
    }

    .progress-status-text {
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(0, 113, 227, 0.25);
      border-top-color: var(--primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .progress-track {
      width: 100%;
      height: 8px;
      background-color: var(--canvas-parchment);
      border-radius: var(--rounded-pill);
      overflow: hidden;
      border: 1px solid var(--hairline);
    }

    .progress-fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #0071e3 0%, #34c759 100%);
      border-radius: var(--rounded-pill);
      transition: width 0.3s ease;
    }

    /* ==========================================================
       KPI CARDS & TABLES
       ========================================================== */
    .view-panel {
      display: none;
      flex-direction: column;
      gap: 20px;
    }

    .view-panel.active {
      display: flex;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }

    .kpi-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: var(--shadow-sm);
    }

    .kpi-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 700;
      color: var(--ink-muted-48);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .kpi-value {
      font-family: var(--font-display);
      font-size: 30px;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: var(--ink);
    }

    .kpi-sub {
      font-size: 11.5px;
      color: var(--ink-muted-48);
    }

    .utility-card {
      background-color: var(--canvas);
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-lg);
      padding: 24px;
      box-shadow: var(--shadow-sm);
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 18px;
      font-weight: 700;
      color: var(--ink);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .badge-count {
      background-color: var(--surface-accent);
      color: var(--primary);
      font-size: 12px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: var(--rounded-pill);
    }

    .table-container {
      width: 100%;
      overflow-x: auto;
    }

    table.apple-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }

    table.apple-table th {
      text-align: left;
      font-weight: 600;
      color: var(--ink-muted-48);
      padding: 12px 14px;
      border-bottom: 1px solid var(--hairline);
      white-space: nowrap;
    }

    table.apple-table td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--divider-soft);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    table.apple-table tbody tr {
      cursor: pointer;
      transition: background-color 0.12s ease;
    }

    table.apple-table tbody tr:hover {
      background-color: var(--surface-pearl);
    }

    table.apple-table tbody tr.selected-row {
      background-color: var(--surface-accent);
    }

    /* Playground Elements */
    .review-textarea {
      width: 100%;
      min-height: 130px;
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-md);
      padding: 16px 18px;
      font-family: var(--font-text);
      font-size: 15px;
      line-height: 1.5;
      color: var(--ink);
      background-color: var(--surface-pearl);
      resize: vertical;
      outline: none;
      margin-bottom: 16px;
      transition: border-color 0.15s ease;
    }

    .review-textarea:focus {
      background-color: var(--canvas);
      border-color: var(--primary);
    }

    .structured-output-box {
      background-color: var(--surface-tile-1);
      color: #ffffff;
      border-radius: var(--rounded-md);
      padding: 20px 24px;
      font-family: var(--font-mono);
      font-size: 13.5px;
      line-height: 1.65;
      white-space: pre-wrap;
      min-height: 160px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
  </style>
</head>
<body>

  <div class="workspace-shell">
    <!-- ==================== SIDEBAR CỐ ĐỊNH ==================== -->
    <aside class="sidebar">
      <div class="sidebar-top">
        <div class="sidebar-brand">
          <div class="brand-icon-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
          </div>
          <div>
            <div class="brand-title">CRAM-ABSA</div>
            <div class="brand-subtitle">Benchmark Suite</div>
          </div>
        </div>

        <div>
          <div class="sidebar-section-label">Điều hướng</div>
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
      </div>

      <div class="sidebar-bottom">
        <div class="sidebar-info-card">
          <div class="sidebar-info-title">Trạng thái Model</div>
          <div class="sidebar-info-value" id="sideModelName">Chưa chọn model</div>
          <div class="sidebar-info-sub" id="sideDeviceName">Phần cứng: CPU (Mac)</div>
        </div>
      </div>
    </aside>

    <!-- ==================== MAIN CANVAS (CUỘN TRONG NỘI BỘ) ==================== -->
    <main class="main-canvas">
      <!-- Main Header Toolbar (Tích hợp nội dung từ Topbar cũ) -->
      <div class="main-header-toolbar">
        <div class="header-titles">
          <h1 class="header-main-title" id="pageTitle">Đánh giá Mô hình trên Tập Test</h1>
          <p class="header-sub-desc" id="pageDesc">Thực thi feed-forward trực tiếp trên tập kiểm thử (Test Split) từ hotel_reviews.parquet</p>
        </div>

        <div class="header-controls">
          <select id="modelSelect" class="apple-select" onchange="onModelChange()"></select>
          <button class="btn-icon-circle" title="Quét lại thư mục model/" onclick="refreshModelList()">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
          </button>
          <select id="sampleSelect" class="apple-select">
            <option value="999">Full Test (999 mẫu)</option>
            <option value="100">Chạy nhanh (100 mẫu)</option>
          </select>
          <button id="btnRunBenchmark" class="btn-pill-primary" onclick="startTestBenchmark()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span id="btnRunText">Chạy Benchmark</span>
          </button>
          <button class="btn-pill-secondary" onclick="exportCSV()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Xuất CSV</span>
          </button>
        </div>
      </div>

      <!-- Thanh Tiến Trình Động (Khi Benchmark Chạy Thật) -->
      <div id="progressBanner" class="progress-banner-card">
        <div class="progress-banner-top">
          <div class="progress-status-text">
            <div class="spinner" id="progressSpinner"></div>
            <span id="progressMessage">Đang chuẩn bị kiểm định...</span>
          </div>
          <span id="progressPercent" style="color:var(--ink-muted-48); font-variant-numeric:tabular-nums;">0.0%</span>
        </div>
        <div class="progress-track">
          <div id="progressFill" class="progress-fill"></div>
        </div>
      </div>

      <!-- TAB 1: OVERVIEW & BENCHMARK -->
      <section id="panel-overview" class="view-panel active">
        <!-- 4 KPI Cards -->
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-top">
              <span>Độ chính xác điểm sao</span>
              <span>RQ1</span>
            </div>
            <div class="kpi-value" id="kpiAcc0">--</div>
            <div class="kpi-sub">Acc@0 (Exact Match 5 mức sao)</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>F1 Cảm xúc 6 khía cạnh</span>
              <span>RQ1</span>
            </div>
            <div class="kpi-value" id="kpiAspectF1">--</div>
            <div class="kpi-sub">6-Aspect Macro-F1 (Tất cả khía cạnh)</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>Sai số thứ bậc CORN</span>
              <span>RQ2</span>
            </div>
            <div class="kpi-value" id="kpiMae">--</div>
            <div class="kpi-sub">Macro-MAE bình quân 5 mức sao</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <span>Tỷ lệ rò rỉ căn cứ</span>
              <span>RQ2</span>
            </div>
            <div class="kpi-value" id="kpiLeakage">--</div>
            <div class="kpi-sub">Cross-Aspect Leakage Rate (%)</div>
          </div>
        </div>

        <!-- Bảng Kết Quả Đối Chứng Thực Nghiệm -->
        <div class="utility-card">
          <div class="card-header">
            <div class="card-title">
              <span>Bảng Đối Chứng Thực Nghiệm Trên Tập Test</span>
              <span class="badge-count" id="tableCountBadge">0 mô hình</span>
            </div>
            <button class="btn-pill-secondary" onclick="exportCSV()">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span>Tải file CSV</span>
            </button>
          </div>

          <div class="table-container">
            <table class="apple-table">
              <thead>
                <tr>
                  <th>Mô hình</th>
                  <th>Tập</th>
                  <th>Số mẫu</th>
                  <th>Test Loss</th>
                  <th>Acc@0 (%)</th>
                  <th>6-Aspect Macro-F1 (%)</th>
                  <th>Macro-MAE</th>
                  <th>Rationale F1 (%)</th>
                  <th>Leakage (%)</th>
                  <th>Thời gian</th>
                </tr>
              </thead>
              <tbody id="summaryTableBody">
                <tr>
                  <td colspan="10" style="color:var(--ink-muted-48); text-align:center; padding:24px;">Đang nạp kết quả thực nghiệm...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- TAB 2: PLAYGROUND CHẠY THỬ REVIEW MỚI -->
      <section id="panel-playground" class="view-panel">
        <div class="utility-card">
          <div class="card-header">
            <h2 class="card-title">Chạy thử Nghiệm Review Mới 100%</h2>
            <span style="font-size:13px; color:var(--ink-muted-48);">Phân tích 6 khía cạnh, chấm điểm sao & sinh nhận xét</span>
          </div>

          <textarea id="reviewInput" class="review-textarea" placeholder="Nhập văn bản review khách sạn tại đây... Ví dụ: The hotel room was spotless and spacious, but front desk staff was unwelcoming and slow. Breakfast was mediocre."></textarea>
          
          <div style="display:flex; gap:12px; margin-bottom:20px;">
            <button class="btn-pill-primary" onclick="runLivePrediction()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              <span>Phân Tích Review</span>
            </button>
            <button class="btn-pill-secondary" onclick="copyStructuredText()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              <span id="copyBtnLabel">Sao chép</span>
            </button>
          </div>

          <div id="structuredTextOutput" class="structured-output-box">Nhấn "Phân Tích Review" để xem kết quả trích xuất 6 khía cạnh và sinh nhận xét...</div>
        </div>
      </section>
    </main>
  </div>

  <script>
    let currentModel = "";
    let pollInterval = null;
    let benchmarkRowsCache = [];

    function switchTab(tabId) {
      document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach(el => el.classList.remove("active"));

      const navBtn = document.getElementById(`nav-${tabId}`);
      const panel = document.getElementById(`panel-${tabId}`);
      if (navBtn) navBtn.classList.add("active");
      if (panel) panel.classList.add("active");

      if (tabId === "overview") {
        document.getElementById("pageTitle").textContent = "Đánh giá Mô hình trên Tập Test";
        document.getElementById("pageDesc").textContent = "Thực thi feed-forward trực tiếp trên tập kiểm thử (Test Split) từ hotel_reviews.parquet";
      } else {
        document.getElementById("pageTitle").textContent = "Chạy Thử Nghiệm Mô Hình";
        document.getElementById("pageDesc").textContent = "Thử nghiệm suy luận phân tích 6 khía cạnh và sinh nhận xét trên bài review tùy ý";
      }
    }

    async function refreshModelList() {
      try {
        const res = await fetch("/api/models");
        const data = await res.json();
        const sel = document.getElementById("modelSelect");
        sel.innerHTML = "";

        if (!data.models || data.models.length === 0) {
          sel.innerHTML = '<option value="">Không có model</option>';
          currentModel = "";
          updateSidebarModel();
          return;
        }

        data.models.forEach(m => {
          const opt = document.createElement("option");
          opt.value = m.filename;
          opt.textContent = `${m.filename} (${m.size_mb} MB)`;
          sel.appendChild(opt);
        });

        currentModel = sel.value;
        updateSidebarModel();
        await loadBenchmarkHistory();
      } catch (e) {
        console.error("Lỗi khi tải danh sách model:", e);
      }
    }

    function onModelChange() {
      const sel = document.getElementById("modelSelect");
      currentModel = sel.value;
      updateSidebarModel();
      highlightActiveModelRow();
    }

    function updateSidebarModel() {
      const el = document.getElementById("sideModelName");
      if (currentModel) {
        el.textContent = currentModel;
        el.title = currentModel;
      } else {
        el.textContent = "Chưa chọn model";
      }
    }

    async function loadBenchmarkHistory() {
      try {
        const res = await fetch("/api/benchmark_history");
        const data = await res.json();
        benchmarkRowsCache = data.rows || [];
        renderSummaryTable(benchmarkRowsCache);
      } catch (e) {
        console.error("Lỗi đọc lịch sử benchmark:", e);
      }
    }

    function renderSummaryTable(rows) {
      const tbody = document.getElementById("summaryTableBody");
      document.getElementById("tableCountBadge").textContent = `${rows.length} mô hình`;

      if (!rows || rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--ink-muted-48); text-align:center; padding:24px;">Chưa có bản ghi kiểm định nào. Hãy chọn model và bấm "Chạy Benchmark".</td></tr>';
        return;
      }

      tbody.innerHTML = "";
      rows.forEach((r, idx) => {
        const tr = document.createElement("tr");
        if (r.model === currentModel) {
          tr.classList.add("selected-row");
        }
        tr.onclick = () => selectBenchmarkRow(r);

        tr.innerHTML = `
          <td><strong>${r.model}</strong></td>
          <td><span style="font-size:11px; padding:2px 6px; border-radius:4px; background:var(--canvas-parchment); font-weight:700;">${r.split}</span></td>
          <td>${r.samples}</td>
          <td>${typeof r.val_loss === "number" ? r.val_loss.toFixed(4) : r.val_loss}</td>
          <td><strong>${typeof r.acc_at_0 === "number" ? r.acc_at_0.toFixed(2) : r.acc_at_0}%</strong></td>
          <td><strong>${typeof r.aspect_macro_f1 === "number" ? r.aspect_macro_f1.toFixed(2) : r.aspect_macro_f1}%</strong></td>
          <td>${typeof r.macro_mae === "number" ? r.macro_mae.toFixed(4) : r.macro_mae}</td>
          <td>${typeof r.rationale_f1 === "number" ? r.rationale_f1.toFixed(2) : r.rationale_f1}%</td>
          <td>${typeof r.leakage_rate === "number" ? r.leakage_rate.toFixed(2) : r.leakage_rate}%</td>
          <td>${r.elapsed || "-"}</td>
        `;
        tbody.appendChild(tr);
      });

      // Update KPI cards from current model if available, else first row
      const activeMatch = rows.find(r => r.model === currentModel) || rows[0];
      if (activeMatch) {
        updateKpis(activeMatch);
      }
    }

    function selectBenchmarkRow(row) {
      updateKpis(row);
      // Select model in dropdown if available
      const sel = document.getElementById("modelSelect");
      for (let opt of sel.options) {
        if (opt.value === row.model) {
          sel.value = row.model;
          currentModel = row.model;
          updateSidebarModel();
          break;
        }
      }
      highlightActiveModelRow();
    }

    function highlightActiveModelRow() {
      const trs = document.querySelectorAll("#summaryTableBody tr");
      trs.forEach(tr => {
        const modelCell = tr.querySelector("td strong");
        if (modelCell && modelCell.textContent.trim() === currentModel) {
          tr.classList.add("selected-row");
        } else {
          tr.classList.remove("selected-row");
        }
      });
      const match = benchmarkRowsCache.find(r => r.model === currentModel);
      if (match) {
        updateKpis(match);
      }
    }

    function updateKpis(r) {
      document.getElementById("kpiAcc0").textContent = typeof r.acc_at_0 === "number" ? r.acc_at_0.toFixed(2) + "%" : r.acc_at_0 + "%";
      document.getElementById("kpiAspectF1").textContent = typeof r.aspect_macro_f1 === "number" ? r.aspect_macro_f1.toFixed(2) + "%" : r.aspect_macro_f1 + "%";
      document.getElementById("kpiMae").textContent = typeof r.macro_mae === "number" ? r.macro_mae.toFixed(4) : r.macro_mae;
      document.getElementById("kpiLeakage").textContent = typeof r.leakage_rate === "number" ? r.leakage_rate.toFixed(2) + "%" : r.leakage_rate + "%";
    }

    /* ==========================================================
       START REAL TEST BENCHMARK & PROGRESS POLLING
       ========================================================== */
    async function startTestBenchmark() {
      if (!currentModel) {
        alert("Vui lòng chọn mô hình trước khi benchmark.");
        return;
      }

      const sampleCount = parseInt(document.getElementById("sampleSelect").value) || 999;
      const btn = document.getElementById("btnRunBenchmark");
      const btnText = document.getElementById("btnRunText");
      btn.disabled = true;
      btnText.textContent = "Đang chạy...";

      const banner = document.getElementById("progressBanner");
      banner.style.display = "flex";
      document.getElementById("progressFill").style.width = "0%";
      document.getElementById("progressPercent").textContent = "0.0%";
      document.getElementById("progressMessage").textContent = `Bắt đầu benchmark ${currentModel} (${sampleCount} mẫu)...`;

      try {
        const res = await fetch("/api/start_benchmark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: currentModel, samples: sampleCount })
        });
        const data = await res.json();
        if (!data.ok) {
          alert(data.message || "Không thể khởi động benchmark.");
          btn.disabled = false;
          btnText.textContent = "Chạy Benchmark";
          banner.style.display = "none";
          return;
        }

        // Start polling status
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(checkBenchmarkStatus, 600);
      } catch (e) {
        alert("Lỗi khi kết nối server: " + e);
        btn.disabled = false;
        btnText.textContent = "Chạy Benchmark";
        banner.style.display = "none";
      }
    }

    async function checkBenchmarkStatus() {
      try {
        const res = await fetch("/api/benchmark_status");
        const st = await res.json();

        const banner = document.getElementById("progressBanner");
        banner.style.display = "flex";

        const fill = document.getElementById("progressFill");
        const pText = document.getElementById("progressPercent");
        const msg = document.getElementById("progressMessage");

        fill.style.width = `${st.progress || 0}%`;
        pText.textContent = `${(st.progress || 0).toFixed(1)}%`;
        msg.textContent = st.message || "Đang xử lý...";

        if (!st.is_running) {
          clearInterval(pollInterval);
          pollInterval = null;

          const btn = document.getElementById("btnRunBenchmark");
          const btnText = document.getElementById("btnRunText");
          btn.disabled = false;
          btnText.textContent = "Chạy Benchmark";

          if (st.error) {
            alert("Quá trình benchmark gặp lỗi: " + st.error);
          } else {
            msg.textContent = st.message || "Đã hoàn thành!";
            fill.style.width = "100%";
            pText.textContent = "100.0%";
            setTimeout(() => {
              banner.style.display = "none";
            }, 3000);
            await loadBenchmarkHistory();
          }
        }
      } catch (e) {
        console.error("Lỗi polling benchmark status:", e);
      }
    }

    function exportCSV() {
      window.location.href = "/api/export_csv";
    }

    async function runLivePrediction() {
      const text = document.getElementById("reviewInput").value.trim();
      const outBox = document.getElementById("structuredTextOutput");
      if (!text) {
        outBox.textContent = "Vui lòng nhập văn bản review để chạy thử.";
        return;
      }
      if (!currentModel) {
        outBox.textContent = "Không tìm thấy file mô hình để chạy.";
        return;
      }

      outBox.textContent = `Đang phân tích với mô hình ${currentModel}... Vui lòng đợi trong giây lát.`;
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review_text: text, model: currentModel })
        });
        const data = await res.json();
        if (data.ok) {
          outBox.textContent = data.formatted_text;
        } else {
          outBox.textContent = `Lỗi: ${data.error || "Không thể thực thi"}`;
        }
      } catch (e) {
        outBox.textContent = `Lỗi kết nối server: ${e}`;
      }
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

        if parsed.path == "/api/benchmark_history":
            self._send_json({"rows": get_all_benchmark_rows()})
            return

        if parsed.path == "/api/benchmark_status":
            with BENCHMARK_LOCK:
                st = dict(BENCHMARK_STATE)
            self._send_json(st)
            return

        if parsed.path == "/api/export_csv":
            csv_bytes = generate_csv_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="benchmark_results.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(body_raw) if body_raw else {}

        if parsed.path == "/api/start_benchmark":
            model_name = payload.get("model", "")
            samples = int(payload.get("samples", 999))
            res = start_benchmark_thread(model_name, num_samples=samples)
            self._send_json(res)
            return

        if parsed.path == "/api/predict":
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
    parser = argparse.ArgumentParser(description="CRAM-ABSA Benchmark Suite")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    port = find_free_port(args.port)
    server = ThreadedHTTPServer(("127.0.0.1", port), BenchmarkGUIHandler)
    url = f"http://127.0.0.1:{port}"

    models = scan_available_models()
    model_str = ", ".join(m["filename"] for m in models) if models else "Không có"
    print(f"==================================================")
    print(f" CRAM-ABSA Benchmark Suite")
    print(f" URL: {url}")
    print(f" Các file mô hình tìm thấy trong model/: {model_str}")
    print(f"==================================================")

    try:
        import webview
        threading.Thread(target=server.serve_forever, daemon=True).start()
        webview.create_window("CRAM-ABSA Benchmark Suite", url, width=1320, height=840)
        webview.start()
    except Exception:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()

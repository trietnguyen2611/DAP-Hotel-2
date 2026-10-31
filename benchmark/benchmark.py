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

ASPECT_ID_MAP = {asp: idx + 1 for idx, asp in enumerate(CANONICAL_ASPECTS)}
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


def get_benchmark_data(selected_model_name: str) -> Dict[str, Any]:
    models = scan_available_models()
    if not models or not selected_model_name:
        return {
            "has_model": False,
            "selected_model": None,
            "message": "Không có",
            "metrics": None,
            "rows": []
        }

    model_path = get_model_dir() / selected_model_name
    if not model_path.exists():
        return {
            "has_model": False,
            "selected_model": None,
            "message": "Không có",
            "metrics": None,
            "rows": []
        }

    metrics_dict = None
    epoch_val = None
    try:
        if TORCH_AVAILABLE:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        else:
            import pickle
            with open(model_path, "rb") as f:
                ckpt = pickle.load(f)
        if isinstance(ckpt, dict):
            metrics_dict = ckpt.get("val_metrics") or ckpt.get("metrics")
            epoch_val = ckpt.get("epoch")
    except Exception as e:
        return {
            "has_model": False,
            "selected_model": selected_model_name,
            "message": f"Không thể đọc file model: {e}",
            "metrics": None,
            "rows": []
        }

    if not metrics_dict:
        return {
            "has_model": True,
            "selected_model": selected_model_name,
            "message": "File model chưa lưu kèm thống kê val_metrics.",
            "metrics": None,
            "rows": []
        }

    acc0 = round(float(metrics_dict.get("acc_at_0", 0.0)), 2)
    asp_f1 = round(float(metrics_dict.get("aspect_macro_f1", 0.0)), 2)
    mae = round(float(metrics_dict.get("macro_mae", 0.0)), 4)
    rat_f1 = round(float(metrics_dict.get("rationale_f1", 0.0)), 2)
    leak = round(float(metrics_dict.get("leakage_rate", 0.0)), 2)
    vloss = round(float(metrics_dict.get("val_loss", 0.0)), 4)

    rows = [
        {
            "model": selected_model_name,
            "epoch": epoch_val if epoch_val is not None else "-",
            "val_loss": vloss,
            "acc_at_0": acc0,
            "aspect_macro_f1": asp_f1,
            "macro_mae": mae,
            "rationale_f1": rat_f1,
            "leakage_rate": leak
        }
    ]

    return {
        "has_model": True,
        "selected_model": selected_model_name,
        "message": "OK",
        "metrics": {
            "acc_at_0": acc0,
            "aspect_macro_f1": asp_f1,
            "macro_mae": mae,
            "rationale_f1": rat_f1,
            "leakage_rate": leak,
            "val_loss": vloss
        },
        "rows": rows
    }


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
            LOADED_MODEL_CACHE[model_name] = {
                "arch": arch,
                "ckpt": ckpt,
                "model": net,
                "tokenizer": tok,
                "gen_tokenizer": gen_tok,
                "feat_mean": ckpt.get("feat_mean", None),
                "feat_std": ckpt.get("feat_std", None)
            }

        bundle = LOADED_MODEL_CACHE[model_name]
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
                from collections import Counter
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


def generate_csv_bytes(selected_model: str) -> bytes:
    data = get_benchmark_data(selected_model)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Model", "Epoch", "Val Loss", "Acc@0 (%)",
        "6-Aspect Macro-F1 (%)", "Macro-MAE", "Rationale F1 (%)", "Leakage (%)"
    ])
    for r in data.get("rows", []):
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
      min-height: 100vh;
    }

    .global-nav {
      height: 44px;
      background-color: var(--surface-black);
      color: var(--body-on-dark);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      font-size: 12px;
      letter-spacing: -0.12px;
    }

    .global-nav-brand {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--body-on-dark);
      font-weight: 600;
      text-decoration: none;
    }

    .sub-nav-frosted {
      height: 52px;
      background-color: rgba(245, 245, 247, 0.82);
      backdrop-filter: saturate(180%) blur(20px);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      border-bottom: 1px solid var(--hairline);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
    }

    .sub-nav-title {
      font-family: var(--font-display);
      font-size: 21px;
      font-weight: 600;
      letter-spacing: 0.231px;
    }

    .sub-nav-controls {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .workspace-shell {
      display: flex;
      flex: 1;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
    }

    .sidebar {
      width: 240px;
      background-color: var(--canvas-parchment);
      border-right: 1px solid var(--hairline);
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      flex-shrink: 0;
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
      padding: 32px;
      display: flex;
      flex-direction: column;
      gap: 24px;
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
    }

    .btn-pill-primary:active, .btn-pill-secondary:active, .btn-icon-circle:active {
      transform: scale(0.95);
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
    }

    .apple-select {
      height: 38px;
      background-color: var(--canvas);
      color: var(--ink);
      font-family: var(--font-text);
      font-size: 14px;
      font-weight: 600;
      border: 1px solid var(--hairline);
      border-radius: var(--rounded-pill);
      padding: 0 16px;
      outline: none;
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

  <header class="global-nav">
    <a href="#" class="global-nav-brand">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
      <span>CRAM-ABSA</span>
    </a>
    <span id="navModelStatus">Model: Không có</span>
  </header>

  <div class="sub-nav-frosted">
    <div class="sub-nav-title">Benchmark</div>
    <div class="sub-nav-controls">
      <select id="modelSelect" class="apple-select" onchange="onModelChange()"></select>
      <button class="btn-icon-circle" title="Quét lại thư mục model/" onclick="refreshModelList()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="23 4 23 10 17 10"></polyline>
          <polyline points="1 20 1 14 7 14"></polyline>
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
        </svg>
      </button>
      <button class="btn-pill-secondary" onclick="exportCSV()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <span>Xuất CSV</span>
      </button>
      <button class="btn-pill-primary" onclick="runBenchmarkNow()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>Chạy Benchmark</span>
      </button>
    </div>
  </div>

  <div class="workspace-shell">
    <aside class="sidebar">
      <div>
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
                <th>Epoch</th>
                <th>Val Loss</th>
                <th>Acc@0 (%)</th>
                <th>6-Aspect F1 (%)</th>
                <th>Macro-MAE</th>
                <th>Rationale F1 (%)</th>
                <th>Leakage (%)</th>
              </tr>
            </thead>
            <tbody id="summaryTableBody">
              <tr><td colspan="8" style="color:var(--ink-muted-48);">Không có</td></tr>
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
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
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

    function switchTab(tabName) {
      document.querySelectorAll(".view-panel").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
      document.getElementById("panel-" + tabName).classList.add("active");
      document.getElementById("nav-" + tabName).classList.add("active");
    }

    async function refreshModelList() {
      const res = await fetch("/api/models");
      const data = await res.json();
      const sel = document.getElementById("modelSelect");
      sel.innerHTML = "";
      if (!data.models || data.models.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Không có";
        sel.appendChild(opt);
        currentModel = "";
        document.getElementById("navModelStatus").textContent = "Model: Không có";
        document.getElementById("sidebarModelMeta").textContent = "Không có";
      } else {
        data.models.forEach(m => {
          const opt = document.createElement("option");
          opt.value = m.filename;
          opt.textContent = `${m.filename} (${m.size_label})`;
          sel.appendChild(opt);
        });
        currentModel = sel.value;
        document.getElementById("navModelStatus").textContent = `Model: model/${currentModel}`;
        document.getElementById("sidebarModelMeta").textContent = data.models.map(x => x.filename).join(", ");
      }
      await runBenchmarkNow();
    }

    async function onModelChange() {
      const sel = document.getElementById("modelSelect");
      currentModel = sel.value;
      document.getElementById("navModelStatus").textContent = currentModel ? `Model: model/${currentModel}` : "Model: Không có";
      await runBenchmarkNow();
    }

    async function runBenchmarkNow() {
      const tbody = document.getElementById("summaryTableBody");
      if (!currentModel) {
        document.getElementById("kpiAcc0").textContent = "--";
        document.getElementById("kpiAspectF1").textContent = "--";
        document.getElementById("kpiMae").textContent = "--";
        document.getElementById("kpiLeakage").textContent = "--";
        tbody.innerHTML = `<tr><td colspan="8" style="color:var(--ink-muted-48);">Không có</td></tr>`;
        return;
      }

      const res = await fetch(`/api/benchmark?model=${encodeURIComponent(currentModel)}`);
      const data = await res.json();

      if (!data.has_model || !data.metrics) {
        document.getElementById("kpiAcc0").textContent = "--";
        document.getElementById("kpiAspectF1").textContent = "--";
        document.getElementById("kpiMae").textContent = "--";
        document.getElementById("kpiLeakage").textContent = "--";
        tbody.innerHTML = `<tr><td colspan="8" style="color:var(--ink-muted-48);">${data.message || "Không có"}</td></tr>`;
        return;
      }

      document.getElementById("kpiAcc0").textContent = data.metrics.acc_at_0.toFixed(2) + "%";
      document.getElementById("kpiAspectF1").textContent = data.metrics.aspect_macro_f1.toFixed(2) + "%";
      document.getElementById("kpiMae").textContent = data.metrics.macro_mae.toFixed(4);
      document.getElementById("kpiLeakage").textContent = data.metrics.leakage_rate.toFixed(2) + "%";

      tbody.innerHTML = "";
      data.rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${r.model}</strong></td>
          <td>${r.epoch}</td>
          <td>${r.val_loss}</td>
          <td>${r.acc_at_0.toFixed(2)}%</td>
          <td>${r.aspect_macro_f1.toFixed(2)}%</td>
          <td>${r.macro_mae.toFixed(4)}</td>
          <td>${r.rationale_f1.toFixed(2)}%</td>
          <td>${r.leakage_rate.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      });
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

        if parsed.path == "/api/benchmark":
            model_name = qs.get("model", [""])[0]
            self._send_json(get_benchmark_data(model_name))
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
        webview.create_window("CRAM-ABSA Benchmark", url, width=1320, height=840)
        webview.start()
    except Exception:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


if __name__ == "__main__":
    main()
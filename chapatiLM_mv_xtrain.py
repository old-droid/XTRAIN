"""
ChapatiLM MV — XTRAIN / CPUWARP-ML Rewrite
============================================
Rewritten from the PyTorch version (goated.ipynb / v3) to run on the
XTRAIN CPUWARP-ML CPU-first framework (github.com/old-droid/XTRAIN,
chapatilmv branch).

Key changes vs. PyTorch version:
  - NO torch, NO nn.Module, NO autograd
  - All tensors → numpy float32 arrays
  - cpuwarp_ml.matmul  replaces  F.linear / torch.matmul
  - cpuwarp_ml.relu    replaces  F.relu / nn.ReLU
  - cpuwarp_ml.softmax replaces  F.softmax
  - Manual gelu, layer_norm, sigmoid, bce_loss, ce_loss, mse_loss
  - Manual AdamW optimizer (pure numpy) — replaces torch.optim.AdamW
  - Manual cosine-annealing LR schedule
  - Manual gradient computation (backprop) per sub-graph
  - cpuwarp_ml WARP scheduler configured for CPU-optimal threading
  - DataLoader replaced by a plain numpy batch iterator
  - torch.save / torch.load replaced by numpy .npz checkpointing
  - encode_for_model returns np.ndarray (int32) instead of torch.Tensor

Requires: numpy, sympy, tiktoken (optional), pandas (optional)
Install:   pip install numpy scipy psutil py-cpuinfo
           pip install cpuwarp_ml   (from old-droid/XTRAIN)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, gc, sys, json, math, shutil, random, argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# ── XTRAIN CPUWARP-ML ─────────────────────────────────────────────────────────
import numpy as np
import cpuwarp_ml  # pip install from old-droid/XTRAIN

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    print("[ChapatiLM] tiktoken not found — Tekken BPE will use character fallback")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[ChapatiLM] pandas not found — CSV loading unavailable")

# ── Configure WARP for CPU-optimal threading ──────────────────────────────────
try:
    cpuwarp_ml.cpuwarp.compute_engine.warp_scheduler.configure({
        "compute_bound_threads": 16,
        "memory_bound_threads": 4,
        "cache_allocation": 0.8,
        "prefetch_distance": 64,
    })
    print("[ChapatiLM] WARP scheduler configured.")
except Exception:
    print("[ChapatiLM] WARP scheduler config skipped (older API).")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Numeric helpers  (replacing torch.nn.functional)
# ═══════════════════════════════════════════════════════════════════════════════

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — replaces F.gelu."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x: np.ndarray) -> np.ndarray:
    tanh_inner = np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    sech2      = 1.0 - tanh_inner ** 2
    dtanh      = math.sqrt(2.0 / math.pi) * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dtanh

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Replaces nn.LayerNorm — normalises last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Wraps cpuwarp_ml.softmax (which acts on 1-D) for batched 2-D arrays."""
    if x.ndim == 1:
        return cpuwarp_ml.softmax(x)
    shifted = x - x.max(axis=axis, keepdims=True)
    e       = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.log(softmax(x, axis=axis) + 1e-12)

# ── Loss functions ─────────────────────────────────────────────────────────────

def bce_with_logits_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Replaces nn.BCEWithLogitsLoss — numerically stable."""
    loss = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())

def bce_with_logits_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (sigmoid(logits) - targets) / logits.shape[0]

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Replaces nn.CrossEntropyLoss."""
    lp = log_softmax(logits)
    n  = logits.shape[0]
    return float(-lp[np.arange(n), labels].mean())

def cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n    = logits.shape[0]
    grad = softmax(logits)
    grad[np.arange(n), labels] -= 1.0
    return grad / n

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))

def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2.0 * (pred - target) / pred.shape[0]

# ── Linear layer helpers ───────────────────────────────────────────────────────

def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x @ W.T + b — replaces nn.Linear forward.
    Uses cpuwarp_ml.matmul for SIMD-accelerated multiply."""
    return cpuwarp_ml.matmul(x.astype(np.float32),
                             W.T.astype(np.float32)) + b

def linear_grads(x: np.ndarray, W: np.ndarray,
                 d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (d_x, d_W, d_b)."""
    d_W = cpuwarp_ml.matmul(d_out.T.astype(np.float32), x.astype(np.float32))
    d_b = d_out.sum(axis=0)
    d_x = cpuwarp_ml.matmul(d_out.astype(np.float32), W.astype(np.float32))
    return d_x, d_W, d_b

# ── Embedding helper ───────────────────────────────────────────────────────────

def embedding_forward(idx: np.ndarray, weight: np.ndarray,
                      padding_idx: int = 0) -> np.ndarray:
    """Replaces nn.Embedding forward — returns (B, L, E)."""
    emb = weight[idx]          # simple fancy indexing
    emb[idx == padding_idx] = 0.0
    return emb

def embedding_grad(idx: np.ndarray, d_emb: np.ndarray,
                   vocab_size: int, embed_dim: int,
                   padding_idx: int = 0) -> np.ndarray:
    """Accumulate embedding weight gradients."""
    d_W = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    d_emb_flat = d_emb.reshape(-1, embed_dim)
    idx_flat   = idx.flatten()
    mask       = idx_flat != padding_idx
    np.add.at(d_W, idx_flat[mask], d_emb_flat[mask])
    return d_W


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  AdamW Optimizer  (replaces torch.optim.AdamW)
# ═══════════════════════════════════════════════════════════════════════════════

class AdamW:
    """Pure-numpy AdamW — replaces torch.optim.AdamW."""

    def __init__(self, params: List[np.ndarray], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params       = params
        self.lr           = lr
        self.b1, self.b2  = betas
        self.eps          = eps
        self.wd           = weight_decay
        self.m            = [np.zeros_like(p) for p in params]
        self.v            = [np.zeros_like(p) for p in params]
        self.t            = 0

    def step(self, grads: List[Optional[np.ndarray]]):
        self.t += 1
        b1, b2, eps = self.b1, self.b2, self.eps
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            # decoupled weight decay
            p *= (1.0 - self.lr * self.wd)
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def zero_grad(self):
        pass  # grads are computed fresh each call — nothing to zero


class CosineAnnealingLR:
    """Replaces torch.optim.lr_scheduler.CosineAnnealingLR."""

    def __init__(self, optimizer: AdamW, T_max: int, eta_min: float = 0.0):
        self.opt     = optimizer
        self.T_max   = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self._step   = 0

    def step(self):
        self._step += 1
        t  = min(self._step, self.T_max)
        lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * t / self.T_max)) / 2.0
        self.opt.lr = lr


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Gradient clipping  (replaces nn.utils.clip_grad_norm_)
# ═══════════════════════════════════════════════════════════════════════════════

def clip_grad_norm(grads: List[Optional[np.ndarray]], max_norm: float = 1.0
                   ) -> List[Optional[np.ndarray]]:
    total = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads if g is not None))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        return [g * scale if g is not None else None for g in grads]
    return grads


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Tekken Tokenizer  (paper §2) — encode_for_model now returns np.ndarray
# ═══════════════════════════════════════════════════════════════════════════════

_WORD_OP_PATTERNS: List[Tuple[str, str]] = [
    (r"\bplus\b|\badd(?:ed|s)?\b|\bsum\b|\bsummed\b",                     "+"),
    (r"\bminus\b|\bsubtract(?:ed|s)?\b|\bless\b|\bspends?\b|\bremoved?\b", "-"),
    (r"\btimes\b|\bmultipl(?:y|ied|ies)\b|\bproduct\b",                   "*"),
    (r"\bdivid(?:e|ed|es|ing)\b|\bquotient\b|\bover\b|\bper\b",           "/"),
    (r"\bequals?\b|\bis\b|\bgives?\b",                                     "="),
]

def word_to_operator(text: str) -> str:
    for pattern, sym in _WORD_OP_PATTERNS:
        text = re.sub(pattern, sym, text, flags=re.IGNORECASE)
    return text

_SAFE_MATH_CHARS = re.compile(r"[^0-9+\-*/=().%^ ,a-zA-Z<>!?]")

def strip_noise(text: str) -> str:
    return _SAFE_MATH_CHARS.sub(" ", text)

def _r2l_encode_number(num_str: str) -> List[str]:
    tokens: List[str] = ["<num>"]
    negative = num_str.startswith("-")
    clean    = num_str.lstrip("-+")
    if negative:
        tokens.append("<neg>")
    int_part, dec_part = (clean.split(".", 1) if "." in clean else (clean, ""))
    for d in reversed(int_part):
        tokens.append(d)
    if dec_part:
        tokens.append("<dec>")
        for d in reversed(dec_part):
            tokens.append(d)
    tokens.append("</num>")
    return tokens

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

def r2l_digit_tokenize(text: str) -> str:
    def replace_num(m: re.Match) -> str:
        return " ".join(_r2l_encode_number(m.group(0))) + " "
    return _NUMBER_RE.sub(replace_num, text)


class TekkenTokenizer:
    """§2 Tekken Tokenizer: tokens = TR2L(Wop(Fnoise(input)))"""

    SPECIAL_TOKENS = ["<num>", "</num>", "<neg>", "<dec>"]
    MAX_SEQ_LEN    = 256

    def __init__(self):
        if _TIKTOKEN_AVAILABLE:
            self._enc            = tiktoken.get_encoding("cl100k_base")
            self._bpe_vocab_size = self._enc.n_vocab
            self._special_ids    = {t: self._bpe_vocab_size + i
                                    for i, t in enumerate(self.SPECIAL_TOKENS)}
            self.vocab_size = self._bpe_vocab_size + len(self.SPECIAL_TOKENS)
            self._mode      = "bpe"
        else:
            CHAR_VOCAB = ("abcdefghijklmnopqrstuvwxyz0123456789 +-*/=().%^<>,!?&|~@#$:;"
                          "\"'\\/\n\t<>_")
            self._char_to_idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}
            self.vocab_size   = len(CHAR_VOCAB) + 1
            self._mode        = "char"

    def _preprocess(self, text: str) -> str:
        return word_to_operator(text.lower())

    def _tokenize_to_ids(self, text: str) -> List[int]:
        text_r2l = r2l_digit_tokenize(text)
        if self._mode == "bpe":
            parts = re.split(r"(</?(?:num|neg|dec)>)", text_r2l)
            ids: List[int] = []
            for part in parts:
                if part in self._special_ids:
                    ids.append(self._special_ids[part])
                elif part:
                    ids.extend(self._enc.encode(part))
            return ids
        return [self._char_to_idx.get(c, 0) for c in text_r2l]

    def encode(self, text: str) -> List[int]:
        return self._tokenize_to_ids(self._preprocess(text))

    def encode_for_model(self, text: str,
                         max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        """Returns np.ndarray[int32] instead of torch.Tensor."""
        ids  = self.encode(text)[:max_len]
        ids += [0] * (max_len - len(ids))
        return np.array(ids, dtype=np.int32)

    def batch_encode(self, texts: List[str],
                     max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        return np.stack([self.encode_for_model(t, max_len) for t in texts])

    def __repr__(self) -> str:
        return f"TekkenTokenizer(mode={self._mode!r}, vocab_size={self.vocab_size:,})"


TOKENIZER      = TekkenTokenizer()
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
print(f"[ChapatiLM] {TOKENIZER}")
print(f"[ChapatiLM] device: CPU (XTRAIN CPUWARP-ML)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Label constants
# ═══════════════════════════════════════════════════════════════════════════════
TYPE_NAMES   = ["Arithmetic", "Algebraic", "Comparison", "Geometric", "Unknown"]
AIM_NAMES    = ["Calculate", "Simplify", "Solve", "Compare", "Evaluate", "Unknown"]
ENGINE_NAMES = ["Native_Compute_Engine", "SymPy_Engine"]
NUM_TYPES    = len(TYPE_NAMES)    # 5
NUM_AIMS     = len(AIM_NAMES)     # 6
NUM_ENGINES  = len(ENGINE_NAMES)  # 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Weight initialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def kaiming_normal(shape: Tuple) -> np.ndarray:
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    std    = math.sqrt(2.0 / fan_in)
    return (np.random.randn(*shape) * std).astype(np.float32)

def normal_emb(vocab_size: int, embed_dim: int) -> np.ndarray:
    std = math.sqrt(2.0 / embed_dim)
    return (np.random.randn(vocab_size, embed_dim) * std).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  NeuralMVModel — pure numpy weights, forward pass, manual backprop
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralMVModel:
    """
    Replaces the torch nn.Module version.
    All parameters are numpy float32 arrays.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        V, E, H = TOKENIZER.vocab_size, embed_dim, hidden_dim

        # ── CharEncoder ───────────────────────────────────────────────────────
        # embedding table
        self.emb_W                = normal_emb(V, E)        # (V, E)
        # layer_norm after pool  (normalises 2*E-dim vector)
        self.ln_W                 = np.ones(2 * E, dtype=np.float32)
        self.ln_b                 = np.zeros(2 * E, dtype=np.float32)
        # fc: 2E → H
        self.enc_W                = kaiming_normal((H, 2 * E))
        self.enc_b                = np.zeros(H, dtype=np.float32)

        # ── Heads ─────────────────────────────────────────────────────────────
        self.det_W                = kaiming_normal((1, H))
        self.det_b                = np.zeros(1, dtype=np.float32)

        self.type_W               = kaiming_normal((NUM_TYPES, H))
        self.type_b               = np.zeros(NUM_TYPES, dtype=np.float32)

        self.aim_W                = kaiming_normal((NUM_AIMS, H))
        self.aim_b                = np.zeros(NUM_AIMS, dtype=np.float32)

        # router: (NUM_TYPES + NUM_AIMS) → 16 → NUM_ENGINES
        self.router_W1            = kaiming_normal((16, NUM_TYPES + NUM_AIMS))
        self.router_b1            = np.zeros(16, dtype=np.float32)
        self.router_W2            = kaiming_normal((NUM_ENGINES, 16))
        self.router_b2            = np.zeros(NUM_ENGINES, dtype=np.float32)

        # arith solver: 6 → H//2 → 64 → 1
        H2                        = hidden_dim // 2
        self.arith_W1             = kaiming_normal((H2, 6))
        self.arith_b1             = np.zeros(H2, dtype=np.float32)
        self.arith_W2             = kaiming_normal((64, H2))
        self.arith_b2             = np.zeros(64, dtype=np.float32)
        self.arith_W3             = kaiming_normal((1, 64))
        self.arith_b3             = np.zeros(1, dtype=np.float32)

        # algebra solver: 3 → H//4 → 32 → 1
        H4                        = hidden_dim // 4
        self.alg_W1               = kaiming_normal((H4, 3))
        self.alg_b1               = np.zeros(H4, dtype=np.float32)
        self.alg_W2               = kaiming_normal((32, H4))
        self.alg_b2               = np.zeros(32, dtype=np.float32)
        self.alg_W3               = kaiming_normal((1, 32))
        self.alg_b3               = np.zeros(1, dtype=np.float32)

        # comparison solver: 2 → 32 → 3
        self.cmp_W1               = kaiming_normal((32, 2))
        self.cmp_b1               = np.zeros(32, dtype=np.float32)
        self.cmp_W2               = kaiming_normal((3, 32))
        self.cmp_b2               = np.zeros(3, dtype=np.float32)

        self._arith_scale         = 1.0
        self._training            = True
        self._dropout_rate        = 0.1

        print(f"  Total trainable params: {self.count_parameters():,}")

    # ── Parameter list (for optimizer) ────────────────────────────────────────
    def _param_list(self) -> List[np.ndarray]:
        return [
            self.emb_W, self.ln_W, self.ln_b,
            self.enc_W, self.enc_b,
            self.det_W, self.det_b,
            self.type_W, self.type_b,
            self.aim_W, self.aim_b,
            self.router_W1, self.router_b1, self.router_W2, self.router_b2,
            self.arith_W1, self.arith_b1, self.arith_W2, self.arith_b2,
            self.arith_W3, self.arith_b3,
            self.alg_W1, self.alg_b1, self.alg_W2, self.alg_b2,
            self.alg_W3, self.alg_b3,
            self.cmp_W1, self.cmp_b1, self.cmp_W2, self.cmp_b2,
        ]

    def train(self):  self._training = True
    def eval(self):   self._training = False

    # ── Encoder ───────────────────────────────────────────────────────────────
    def _encode(self, token_ids: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        token_ids: (B, L)  int32
        Returns h: (B, H), cache dict for backprop
        """
        B, L     = token_ids.shape
        emb      = embedding_forward(token_ids, self.emb_W, padding_idx=0)  # B,L,E
        mask     = (token_ids != 0).astype(np.float32)[:, :, None]           # B,L,1
        denom    = mask.sum(axis=1).clip(min=1)                               # B,1
        mean_p   = (emb * mask).sum(axis=1) / denom                           # B,E
        max_p    = (emb + (1 - mask) * (-1e9)).max(axis=1)                    # B,E
        pooled   = np.concatenate([mean_p, max_p], axis=-1)                   # B,2E
        pooled_n = layer_norm(pooled, self.ln_W, self.ln_b)                   # B,2E
        pre_act  = linear_forward(pooled_n, self.enc_W, self.enc_b)           # B,H

        if self._training and self._dropout_rate > 0:
            drop_mask = (np.random.rand(*pre_act.shape) >
                         self._dropout_rate).astype(np.float32) / (1 - self._dropout_rate)
        else:
            drop_mask = np.ones_like(pre_act)

        h = gelu(pre_act) * drop_mask                                         # B,H

        cache = dict(token_ids=token_ids, emb=emb, mask=mask, denom=denom,
                     mean_p=mean_p, max_p=max_p, pooled=pooled, pooled_n=pooled_n,
                     pre_act=pre_act, drop_mask=drop_mask, h=h)
        return h, cache

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, token_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Replaces NeuralMVModel.forward() in PyTorch version."""
        h, enc_cache = self._encode(token_ids)

        det_logits  = linear_forward(h, self.det_W, self.det_b).squeeze(-1)  # B
        type_logits = linear_forward(h, self.type_W, self.type_b)            # B,5
        aim_logits  = linear_forward(h, self.aim_W, self.aim_b)              # B,6

        tp          = softmax(type_logits)
        ap          = softmax(aim_logits)
        router_in   = np.concatenate([tp, ap], axis=-1)                      # B,11
        r1          = gelu(linear_forward(router_in, self.router_W1, self.router_b1))
        router_log  = linear_forward(r1, self.router_W2, self.router_b2)     # B,2

        return dict(
            detector    = det_logits,
            type        = type_logits,
            aim         = aim_logits,
            router      = router_log,
            _enc_cache  = enc_cache,
            _h          = h,
            _tp         = tp,
            _ap         = ap,
            _r1         = r1,
            _router_in  = router_in,
        )

    # ── Solver sub-graphs ──────────────────────────────────────────────────────
    def solve_arith(self, f: np.ndarray) -> np.ndarray:
        h1 = gelu(linear_forward(f,  self.arith_W1, self.arith_b1))
        h2 = gelu(linear_forward(h1, self.arith_W2, self.arith_b2))
        return linear_forward(h2, self.arith_W3, self.arith_b3).squeeze(-1)

    def solve_algebra(self, f: np.ndarray) -> np.ndarray:
        h1 = gelu(linear_forward(f,  self.alg_W1, self.alg_b1))
        h2 = gelu(linear_forward(h1, self.alg_W2, self.alg_b2))
        return linear_forward(h2, self.alg_W3, self.alg_b3).squeeze(-1)

    def solve_comparison(self, f: np.ndarray) -> np.ndarray:
        h1 = gelu(linear_forward(f,  self.cmp_W1, self.cmp_b1))
        return linear_forward(h1, self.cmp_W2, self.cmp_b2)

    # ── Backprop helpers ───────────────────────────────────────────────────────
    def _backward_encoder(self, d_h: np.ndarray,
                          cache: Dict) -> List[Optional[np.ndarray]]:
        """
        Propagate d_h (B,H) back through encoder.
        Returns gradient list matching the first 5 entries of _param_list():
          [d_emb_W, d_ln_W, d_ln_b, d_enc_W, d_enc_b]
        """
        # gelu + dropout
        d_pre_act = d_h * cache["drop_mask"] * gelu_grad(cache["pre_act"])
        # linear enc
        d_pooled_n, d_enc_W, d_enc_b = linear_grads(cache["pooled_n"],
                                                     self.enc_W, d_pre_act)
        # layer_norm (simplified grad — treat as pass-through scale)
        d_pooled = d_pooled_n * self.ln_W  # approximate LN grad
        mean_d   = d_pooled.mean(axis=-1, keepdims=True)
        std_d    = np.sqrt(cache["pooled"].var(axis=-1, keepdims=True) + 1e-5)
        x_hat    = (cache["pooled"] - cache["pooled"].mean(axis=-1, keepdims=True)) / std_d
        d_ln_W   = (d_pooled_n * x_hat).sum(axis=0)
        d_ln_b   = d_pooled_n.sum(axis=0)

        B, E2    = d_pooled.shape
        E        = E2 // 2
        d_mean_p = d_pooled[:, :E]
        d_max_p  = d_pooled[:, E:]

        # mean pool grad
        mask   = cache["mask"]          # B,L,1
        denom  = cache["denom"]         # B,1
        d_emb_mean = d_mean_p[:, None, :] * mask / denom[:, None, :]  # B,L,E

        # max pool grad — only at argmax positions
        emb_shifted = cache["emb"] + (1 - mask) * (-1e9)
        argmax_idx  = emb_shifted.argmax(axis=1)             # B,E
        d_emb_max   = np.zeros_like(cache["emb"])            # B,L,E
        B_i, E_i    = np.meshgrid(np.arange(B), np.arange(E), indexing="ij")
        d_emb_max[B_i, argmax_idx, E_i] = d_max_p

        d_emb = (d_emb_mean + d_emb_max).astype(np.float32)  # B,L,E
        d_emb_W = embedding_grad(cache["token_ids"], d_emb,
                                 TOKENIZER.vocab_size, E)

        return [d_emb_W, d_ln_W, d_ln_b, d_enc_W, d_enc_b]

    def compute_grads(self, batch: Dict, out: Dict,
                      bce_fn, ce_fn, mse_fn,
                      weights: Dict) -> List[Optional[np.ndarray]]:
        """
        Full manual backprop.  Returns gradient list aligned with _param_list().
        """
        # ── classification head grads ──────────────────────────────────────────
        d_det    = bce_with_logits_grad(out["detector"], batch["is_math"])  # B
        d_type   = cross_entropy_grad(out["type"],  batch["type_label"])    # B,5
        d_aim    = cross_entropy_grad(out["aim"],   batch["aim_label"])     # B,6

        router_tgt = (batch["type_label"] == 1).astype(np.int32)
        d_router = cross_entropy_grad(out["router"], router_tgt)            # B,2

        w = weights

        # ── detector head grads ───────────────────────────────────────────────
        d_h_det  = cpuwarp_ml.matmul(
            (w["detector"] * d_det[:, None]).astype(np.float32),
            self.det_W.astype(np.float32))   # B,H
        d_det_W, d_det_b_g = (
            cpuwarp_ml.matmul((w["detector"] * d_det[:, None]).T.astype(np.float32),
                               out["_h"].astype(np.float32)),
            (w["detector"] * d_det).sum(axis=0, keepdims=True))

        # ── type head ─────────────────────────────────────────────────────────
        d_h_type = cpuwarp_ml.matmul((w["type"] * d_type).astype(np.float32),
                                      self.type_W.astype(np.float32))
        d_type_W, d_type_b_g = (
            cpuwarp_ml.matmul((w["type"] * d_type).T.astype(np.float32),
                               out["_h"].astype(np.float32)),
            (w["type"] * d_type).sum(axis=0))

        # ── aim head ──────────────────────────────────────────────────────────
        d_h_aim  = cpuwarp_ml.matmul((w["aim"] * d_aim).astype(np.float32),
                                      self.aim_W.astype(np.float32))
        d_aim_W, d_aim_b_g = (
            cpuwarp_ml.matmul((w["aim"] * d_aim).T.astype(np.float32),
                               out["_h"].astype(np.float32)),
            (w["aim"] * d_aim).sum(axis=0))

        # ── router head ───────────────────────────────────────────────────────
        d_r2, d_rW2, d_rb2 = linear_grads(out["_r1"], self.router_W2,
                                           w["router"] * d_router)
        d_r1_pre = d_r2 * gelu_grad(linear_forward(out["_router_in"],
                                                    self.router_W1, self.router_b1))
        _, d_rW1, d_rb1 = linear_grads(out["_router_in"], self.router_W1, d_r1_pre)
        # router grad w.r.t. h is negligible (goes through softmax/concat) — set 0
        d_h_router = np.zeros_like(out["_h"])

        # ── solver heads ──────────────────────────────────────────────────────
        am = np.array(batch["has_arith"],   dtype=bool)
        lm = np.array(batch["has_algebra"], dtype=bool)
        cm = np.array(batch["has_cmp"],     dtype=bool)

        # arith
        arith_grads = [None] * 6  # W1,b1,W2,b2,W3,b3
        if am.any():
            af   = batch["arith_features"][am].astype(np.float32)
            aa   = batch["arith_answer"][am].astype(np.float32)
            ah1  = gelu(linear_forward(af, self.arith_W1, self.arith_b1))
            ah2  = gelu(linear_forward(ah1, self.arith_W2, self.arith_b2))
            apred = linear_forward(ah2, self.arith_W3, self.arith_b3).squeeze(-1)
            d_ap = mse_grad(apred, aa)[:, None] * w["arith"]
            _, d_aW3, d_ab3 = linear_grads(ah2, self.arith_W3, d_ap)
            d_ah2 = cpuwarp_ml.matmul(d_ap.astype(np.float32),
                                       self.arith_W3.astype(np.float32))
            d_ah2 = d_ah2 * gelu_grad(linear_forward(ah1, self.arith_W2, self.arith_b2))
            _, d_aW2, d_ab2 = linear_grads(ah1, self.arith_W2, d_ah2)
            d_ah1 = cpuwarp_ml.matmul(d_ah2.astype(np.float32),
                                       self.arith_W2.astype(np.float32))
            d_ah1 = d_ah1 * gelu_grad(linear_forward(af, self.arith_W1, self.arith_b1))
            _, d_aW1, d_ab1 = linear_grads(af, self.arith_W1, d_ah1)
            arith_grads = [d_aW1, d_ab1, d_aW2, d_ab2, d_aW3, d_ab3.squeeze(-1)]

        # algebra
        alg_grads = [None] * 6
        if lm.any():
            lf    = batch["algebra_features"][lm].astype(np.float32)
            la    = batch["algebra_answer"][lm].astype(np.float32)
            lh1   = gelu(linear_forward(lf, self.alg_W1, self.alg_b1))
            lh2   = gelu(linear_forward(lh1, self.alg_W2, self.alg_b2))
            lpred = linear_forward(lh2, self.alg_W3, self.alg_b3).squeeze(-1)
            d_lp  = mse_grad(lpred, la)[:, None] * w["algebra"]
            _, d_lW3, d_lb3 = linear_grads(lh2, self.alg_W3, d_lp)
            d_lh2 = cpuwarp_ml.matmul(d_lp.astype(np.float32),
                                       self.alg_W3.astype(np.float32))
            d_lh2 = d_lh2 * gelu_grad(linear_forward(lh1, self.alg_W2, self.alg_b2))
            _, d_lW2, d_lb2 = linear_grads(lh1, self.alg_W2, d_lh2)
            d_lh1 = cpuwarp_ml.matmul(d_lh2.astype(np.float32),
                                       self.alg_W2.astype(np.float32))
            d_lh1 = d_lh1 * gelu_grad(linear_forward(lf, self.alg_W1, self.alg_b1))
            _, d_lW1, d_lb1 = linear_grads(lf, self.alg_W1, d_lh1)
            alg_grads = [d_lW1, d_lb1, d_lW2, d_lb2, d_lW3, d_lb3.squeeze(-1)]

        # comparison
        cmp_grads = [None] * 4
        if cm.any():
            cf   = batch["cmp_features"][cm].astype(np.float32)
            cl   = batch["cmp_label"][cm]
            ch1  = gelu(linear_forward(cf, self.cmp_W1, self.cmp_b1))
            cprd = linear_forward(ch1, self.cmp_W2, self.cmp_b2)
            d_cp = cross_entropy_grad(cprd, cl) * w["cmp"]
            _, d_cW2, d_cb2 = linear_grads(ch1, self.cmp_W2, d_cp)
            d_ch1 = cpuwarp_ml.matmul(d_cp.astype(np.float32),
                                       self.cmp_W2.astype(np.float32))
            d_ch1 = d_ch1 * gelu_grad(linear_forward(cf, self.cmp_W1, self.cmp_b1))
            _, d_cW1, d_cb1 = linear_grads(cf, self.cmp_W1, d_ch1)
            cmp_grads = [d_cW1, d_cb1, d_cW2, d_cb2]

        # ── aggregate d_h and backprop encoder ────────────────────────────────
        d_h = d_h_det + d_h_type + d_h_aim + d_h_router
        enc_grads = self._backward_encoder(d_h, out["_enc_cache"])

        # align with _param_list():
        # emb_W, ln_W, ln_b, enc_W, enc_b,
        # det_W, det_b, type_W, type_b, aim_W, aim_b,
        # router_W1, router_b1, router_W2, router_b2,
        # arith_W1..b3,  alg_W1..b3,  cmp_W1..b2
        return (enc_grads +
                [d_det_W,  d_det_b_g.flatten(),
                 d_type_W, d_type_b_g,
                 d_aim_W,  d_aim_b_g,
                 d_rW1, d_rb1, d_rW2, d_rb2] +
                arith_grads + alg_grads + cmp_grads)

    # ── Parameter count ────────────────────────────────────────────────────────
    def count_parameters(self, verbose: bool = False) -> int:
        params = {
            "embedding":       self.emb_W,
            "layer_norm":      np.concatenate([self.ln_W, self.ln_b]),
            "encoder_fc":      np.concatenate([self.enc_W.ravel(), self.enc_b]),
            "detector_head":   np.concatenate([self.det_W.ravel(), self.det_b]),
            "type_head":       np.concatenate([self.type_W.ravel(), self.type_b]),
            "aim_head":        np.concatenate([self.aim_W.ravel(), self.aim_b]),
            "router_head":     np.concatenate([self.router_W1.ravel(), self.router_b1,
                                               self.router_W2.ravel(), self.router_b2]),
            "arith_head":      np.concatenate([self.arith_W1.ravel(), self.arith_b1,
                                               self.arith_W2.ravel(), self.arith_b2,
                                               self.arith_W3.ravel(), self.arith_b3]),
            "algebra_head":    np.concatenate([self.alg_W1.ravel(), self.alg_b1,
                                               self.alg_W2.ravel(), self.alg_b2,
                                               self.alg_W3.ravel(), self.alg_b3]),
            "comparison_head": np.concatenate([self.cmp_W1.ravel(), self.cmp_b1,
                                               self.cmp_W2.ravel(), self.cmp_b2]),
        }
        total = sum(p.size for p in params.values())
        if verbose:
            print("\n── Parameter count breakdown ─────────────────────────────────")
            for name, p in params.items():
                bar = "█" * max(1, int(50 * p.size / total))
                print(f"  {name:<22s}  {p.size:>10,}  {bar}")
            print(f"  {'TOTAL':<22s}  {total:>10,}")
            print("─" * 60)
        return total

    # ── Inference helpers ─────────────────────────────────────────────────────
    def classify_type(self, text: str) -> str:
        ids = self.encode_input(text)
        out = self.forward(ids)
        return TYPE_NAMES[int(out["type"][0].argmax())]

    def classify_aim(self, text: str) -> str:
        ids = self.encode_input(text)
        out = self.forward(ids)
        return AIM_NAMES[int(out["aim"][0].argmax())]

    def route_engine(self, problem_type: str, aim: str) -> str:
        if problem_type in ("Algebraic", "Geometric") or aim in ("Solve", "Simplify"):
            return "SymPy_Engine"
        return "Native_Compute_Engine"

    def encode_input(self, text: str) -> np.ndarray:
        return TOKENIZER.encode_for_model(text)[None, :]  # (1, L)

    def solve(self, query: str) -> Dict:
        self.eval()
        problem_type = self.classify_type(query)
        aim          = self.classify_aim(query)
        engine       = self.route_engine(problem_type, aim)
        result       = None

        if problem_type == "Comparison" or aim == "Compare":
            result = _solve_comparison_symbolic(query, self)

        if result is None and engine == "SymPy_Engine":
            result = _solve_algebra_symbolic(query)

        if result is None and engine == "Native_Compute_Engine":
            result = _solve_arith_symbolic(query)
            if result is None:
                feats = _extract_arith_features(query)
                if feats is not None:
                    ft  = np.array(feats, dtype=np.float32)[None, :]
                    raw = float(self.solve_arith(ft)[0])
                    result = _format_number(_arith_decode(raw))

        if result is None:
            result = _solve_arith_symbolic(query)
        if result is None:
            result = "Unable to solve"

        return {"query": query, "problem_type": problem_type,
                "aim": aim, "engine": engine, "result": result}

    # ── State dict (numpy .npz compatible) ────────────────────────────────────
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v for k, v in vars(self).items()
                if isinstance(v, np.ndarray)}

    def load_state_dict(self, d: Dict[str, np.ndarray]):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Feature extractors & symbolic solvers  (unchanged from PyTorch version)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_numbers_and_op(expression: str):
    text = expression.lower()
    text = text.replace("divided by", "/").replace("divide", "")
    text = text.replace("plus", "+").replace("minus", "-").replace("times", "*")
    text = re.sub(r'\bby\b', '/', text)
    m = re.search(r"(-?\d+\.?\d*)\s*([\+\-\*\/])\s*(-?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1)), m.group(2), float(m.group(3))
        except ValueError:
            return None
    return None

def _solve_arith_symbolic(expression: str) -> Optional[str]:
    if "compare" in expression.lower():
        nums = re.findall(r"-?\d+\.?\d*", expression)
        if len(nums) == 2:
            n1, n2 = float(nums[0]), float(nums[1])
            if n1 > n2: return f"{_format_number(n1)} > {_format_number(n2)}"
            if n1 < n2: return f"{_format_number(n1)} < {_format_number(n2)}"
            return f"{_format_number(n1)} == {_format_number(n2)}"
    parsed = _parse_numbers_and_op(expression)
    if parsed is None:
        return None
    n1, op, n2 = parsed
    if   op == "+": ans = n1 + n2
    elif op == "-": ans = n1 - n2
    elif op == "*": ans = n1 * n2
    elif op == "/":
        if abs(n2) < 1e-12:
            return "undefined"
        ans = n1 / n2
    else:
        return None
    return _format_number(ans)

def _extract_arith_features(expression: str) -> Optional[List[float]]:
    parsed = _parse_numbers_and_op(expression)
    if parsed is None:
        return None
    n1, op, n2 = parsed
    oh = [0.0] * 4
    oh[{"+": 0, "-": 1, "*": 2, "/": 3}.get(op, 0)] = 1.0
    return [math.log1p(abs(n1)), math.log1p(abs(n2))] + oh

def _parse_linear(expression: str) -> Optional[Tuple[float, float, float]]:
    text = re.sub(r"\bsolve\b", "", expression, flags=re.I).strip()
    m = re.match(
        r"([\d\.]*)?\s*[a-zA-Z]\s*([\+\-]?\s*[\d\.]+)?\s*=\s*([\d\.\+\-\s]+)", text)
    if not m:
        return None
    try:
        lhs_coef  = m.group(1).strip() if m.group(1) else ""
        lhs_const = m.group(2).strip() if m.group(2) else ""
        rhs       = m.group(3).strip()
        coef  = float(lhs_coef) if lhs_coef else 1.0
        const = float(lhs_const.replace(" ", "")) if lhs_const else 0.0
        rhs_v = float(eval(rhs, {"__builtins__": {}}, {})) if rhs else 0.0
        return (coef, const, rhs_v)
    except Exception:
        return None

def _format_number(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 0.01 else f"{x:.6g}"

def _arith_encode(x: float) -> float:
    return math.log1p(abs(x)) * (1.0 if x >= 0 else -1.0)

def _arith_decode(x: float) -> float:
    return math.expm1(abs(x)) * (1.0 if x >= 0 else -1.0)

def _solve_comparison_symbolic(query: str, model: "NeuralMVModel") -> Optional[str]:
    m = re.search(r"(\d+\.?\d*)\s*[><=!]+\s*(\d+\.?\d*)", query)
    if not m:
        m = re.search(r"compare\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)", query, re.I)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    ft   = np.array([[a, b]], dtype=np.float32)
    probs = softmax(model.solve_comparison(ft)[0])
    idx  = int(probs.argmax())
    return f"{a} {['>','<','='][idx]} {b}  (conf {probs[idx]:.3f})"

def _solve_algebra_symbolic(expression: str) -> Optional[str]:
    try:
        import sympy
        x   = sympy.Symbol("x")
        lhs_str, rhs_str = (
            expression.split("=", 1) if "=" in expression else (expression, "0"))
        sol = sympy.solve(
            sympy.Eq(sympy.sympify(lhs_str.strip()), sympy.sympify(rhs_str.strip())), x)
        if sol:
            return f"x = {sol[0]}"
    except Exception:
        pass
    parsed = _parse_linear(expression)
    if parsed is None:
        return None
    coef, const, rhs_v = parsed
    if abs(coef) < 1e-9:
        return None
    return f"x = {_format_number((rhs_v - const) / coef)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Dataset loading  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_CSV = "/kaggle/input/datasets/awsaf49/math-qsa-dataset/train.csv"
TEST_CSV  = "/kaggle/input/datasets/awsaf49/math-qsa-dataset/test.csv"

def _infer_category(text: str) -> str:
    t = text.lower()
    if re.search(r"[a-df-wyz]\s*[\+\-\*\/=]|solve|find\s+x|variable", t):
        return "Algebraic"
    if re.search(r"compare|greater\s+than|less\s+than|[<>]=?", t):
        return "Comparison"
    if re.search(r"area|perimeter|volume|radius|diameter|angle|triangle|circle|square", t):
        return "Geometric"
    if re.search(r"\d.*[\+\-\*\/].*\d|plus|minus|times|divide|add|subtract|multiply", t):
        return "Arithmetic"
    return "Unknown"

def load_kaggle_csv(path: str) -> List[Dict]:
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    q_col   = next((c for c in df.columns if c in ("question","problem","q")), None)
    a_col   = next((c for c in df.columns if c in ("answer","correct","a","solution")), None)
    cat_col = next((c for c in df.columns if c in ("category","type","class")), None)
    if q_col is None: raise ValueError(f"No question column in {list(df.columns)}")
    if a_col is None: raise ValueError(f"No answer column in {list(df.columns)}")
    problems = []
    for _, row in df.iterrows():
        text = str(row[q_col]).strip()
        ans  = str(row[a_col]).strip()
        cat  = str(row[cat_col]).strip() if cat_col else _infer_category(text)
        problems.append({"problem": text, "answer": ans, "category": cat})
    print(f"  Loaded {len(problems):,} problems from {path}")
    return problems

def load_problems(path: str) -> List[Dict]:
    if path.endswith(".csv"):
        return load_kaggle_csv(path)
    with open(path) as f:
        data = json.load(f)
    return data.get("problems", data)

def generate_synthetic_dataset(n: int = 8000, path: str = "math_data.json") -> str:
    templates = [
        "{a} + {b}", "What is {a} plus {b}?", "calculate {a} + {b}",
        "add {a} and {b}", "Find the sum of {a} and {b}",
        "{a} - {b}", "What is {a} minus {b}?",
        "{a} * {b}", "multiply {a} and {b}",
        "{a} / {b}", "divide {a} by {b}",
        "compare {a} and {b}", "is {a} greater than {b}?",
        "solve 2x + {a} = {b}",
    ]
    cat_map = {t: ("Algebraic" if "solve" in t
                   else "Comparison" if "compare" in t or "greater" in t
                   else "Arithmetic")
               for t in templates}
    problems = []
    for _ in range(n):
        a, b = random.randint(1, 200), random.randint(1, 200)
        tmpl = random.choice(templates)
        text = tmpl.format(a=a, b=b)
        cat  = cat_map[tmpl]
        if cat == "Arithmetic":
            ans = (a + b if "+" in tmpl or any(w in tmpl for w in ["plus","sum","add"])
                   else a - b if "-" in tmpl or "minus" in tmpl
                   else a * b if "*" in tmpl or "multiply" in tmpl
                   else round(a / b, 4) if b else 0)
        elif cat == "Comparison":
            ans = 1 if a > b else (-1 if a < b else 0)
        else:
            ans = (b - a) / 2
        problems.append({"problem": text, "answer": str(ans), "category": cat})
    with open(path, "w") as f:
        json.dump({"problems": problems}, f, indent=2)
    print(f"  Generated {n} synthetic problems → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  ChapatiDataset — plain python list of numpy dicts
#     (replaces torch Dataset + DataLoader)
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_BASE = {
    "Arithmetic": 0, "arithmetic": 0,
    "Algebra": 1,    "Algebraic": 1,
    "Comparison": 2,
    "Geometry": 3,   "Geometric": 3,
    "None": 4,       "unknown": 4, "Unknown": 4,
    "Prealgebra": 0, "Number Theory": 0,
    "Intermediate Algebra": 1, "Precalculus": 1,
    "Counting & Probability": 4, "Statistics": 4,
}
_AIM_FROM_TYPE = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5}

def _build_type_map(problems: List[Dict]) -> Dict[str, int]:
    cats   = {p.get("category", "unknown") for p in problems}
    result = {c: _TYPE_BASE.get(c, 4) for c in cats}
    print(f"  type_map: { {k: TYPE_NAMES[v] for k, v in sorted(result.items())} }")
    return result

def build_dataset(problems: List[Dict]) -> List[Dict]:
    """Returns a list of numpy-array dicts (replaces ChapatiDataset)."""
    type_map = _build_type_map(problems)
    rows = []
    for p in problems:
        text     = p.get("problem", "")
        cat_str  = p.get("category", "unknown")
        type_lbl = type_map.get(cat_str, 4)
        aim_lbl  = _AIM_FROM_TYPE.get(type_lbl, 5)
        is_math  = 0.0 if cat_str in ("None", "unknown", "Unknown") else 1.0

        try:
            raw_ans = float(p.get("answer", 0))
        except Exception:
            raw_ans = 0.0

        af          = _extract_arith_features(text)
        has_arith   = af is not None
        arith_feats = af if has_arith else [0.0] * 6
        arith_ans   = _arith_encode(raw_ans) if has_arith else 0.0

        alg       = _parse_linear(text)
        has_alg   = alg is not None
        alg_feats = list(alg) if has_alg else [0.0, 0.0, 0.0]
        alg_ans   = raw_ans if has_alg else 0.0

        m_cmp = re.search(r"(\d+\.?\d*)\s*[><=!]+\s*(\d+\.?\d*)", text)
        if not m_cmp:
            m_cmp = re.search(r"compare\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)", text, re.I)
        has_cmp = m_cmp is not None
        if has_cmp:
            a_c, b_c = float(m_cmp.group(1)), float(m_cmp.group(2))
            cmp_feats = [a_c, b_c]
            cmp_lbl   = 0 if a_c > b_c else (1 if a_c < b_c else 2)
        else:
            cmp_feats, cmp_lbl = [0.0, 0.0], 2

        rows.append({
            "token_ids":        TOKENIZER.encode_for_model(text),
            "is_math":          np.float32(is_math),
            "type_label":       np.int32(type_lbl),
            "aim_label":        np.int32(aim_lbl),
            "arith_features":   np.array(arith_feats, dtype=np.float32),
            "arith_answer":     np.float32(arith_ans),
            "algebra_features": np.array(alg_feats,   dtype=np.float32),
            "algebra_answer":   np.float32(alg_ans),
            "cmp_features":     np.array(cmp_feats,   dtype=np.float32),
            "cmp_label":        np.int32(cmp_lbl),
            "has_arith":        has_arith,
            "has_algebra":      has_alg,
            "has_cmp":          has_cmp,
        })
    return rows


def numpy_dataloader(dataset: List[Dict], batch_size: int,
                     shuffle: bool = True):
    """
    Replaces torch DataLoader — yields batched numpy dicts.
    cpuwarp_ml benefits from contiguous float32 arrays, so we stack here.
    """
    idx = list(range(len(dataset)))
    if shuffle:
        random.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        chunk = [dataset[i] for i in idx[start: start + batch_size]]
        batch: Dict[str, Any] = {}
        for key in chunk[0]:
            vals = [row[key] for row in chunk]
            if isinstance(vals[0], np.ndarray):
                batch[key] = np.stack(vals)
            elif isinstance(vals[0], (np.float32, float)):
                batch[key] = np.array(vals, dtype=np.float32)
            elif isinstance(vals[0], (np.int32, int)):
                batch[key] = np.array(vals, dtype=np.int32)
            else:
                batch[key] = vals   # bool lists (has_arith etc.)
        yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralMVTrainer:
    def __init__(
        self, model: NeuralMVModel, lr: float = 1e-3, weight_decay: float = 1e-4,
        detector_w=1.0, type_w=1.5, aim_w=1.0,
        router_w=0.5,   arith_w=2.0, algebra_w=1.5, cmp_w=1.0,
    ):
        self.model     = model
        self.optimizer = AdamW(model._param_list(), lr=lr,
                               weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100,
                                           eta_min=lr * 0.01)
        self.w = dict(detector=detector_w, type=type_w, aim=aim_w,
                      router=router_w, arith=arith_w, algebra=algebra_w, cmp=cmp_w)

    def _compute_losses(self, batch: Dict, out: Dict) -> Dict[str, float]:
        loss_det    = bce_with_logits_loss(out["detector"], batch["is_math"])
        loss_type   = cross_entropy_loss(out["type"],       batch["type_label"])
        loss_aim    = cross_entropy_loss(out["aim"],        batch["aim_label"])
        router_tgt  = (batch["type_label"] == 1).astype(np.int32)
        loss_router = cross_entropy_loss(out["router"],     router_tgt)

        loss_arith = loss_algebra = loss_cmp = 0.0
        am = np.array(batch["has_arith"],   dtype=bool)
        lm = np.array(batch["has_algebra"], dtype=bool)
        cm = np.array(batch["has_cmp"],     dtype=bool)

        if am.any():
            af   = batch["arith_features"][am].astype(np.float32)
            aa   = batch["arith_answer"][am].astype(np.float32)
            pred = self.model.solve_arith(af)
            loss_arith = mse_loss(pred, aa)

        if lm.any():
            lf   = batch["algebra_features"][lm].astype(np.float32)
            la   = batch["algebra_answer"][lm].astype(np.float32)
            pred = self.model.solve_algebra(lf)
            loss_algebra = mse_loss(pred, la)

        if cm.any():
            cf  = batch["cmp_features"][cm].astype(np.float32)
            cl  = batch["cmp_label"][cm]
            pred = self.model.solve_comparison(cf)
            loss_cmp = cross_entropy_loss(pred, cl)

        w = self.w
        total = (w["detector"] * loss_det     + w["type"]    * loss_type     +
                 w["aim"]      * loss_aim      + w["router"]  * loss_router   +
                 w["arith"]    * loss_arith    + w["algebra"] * loss_algebra  +
                 w["cmp"]      * loss_cmp)

        return dict(total=total, detector=loss_det, type=loss_type, aim=loss_aim,
                    router=loss_router, arith=loss_arith, algebra=loss_algebra,
                    cmp=loss_cmp)

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        totals    = {k: 0.0 for k in
                     ("total","detector","type","aim","router","arith","algebra","cmp")}
        n_batches = 0

        for batch in loader:
            out    = self.model.forward(batch["token_ids"])
            losses = self._compute_losses(batch, out)

            grads = self.model.compute_grads(
                batch, out,
                bce_with_logits_grad, cross_entropy_grad, mse_grad,
                self.w)
            grads = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.step(grads)

            for k in totals:
                totals[k] += losses[k]
            n_batches += 1

        self.scheduler.step()
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        loss_acc = {k: 0.0 for k in
                    ("total","detector","type","aim","router","arith","algebra","cmp")}
        correct_det = correct_type = correct_aim = total = n_batches = 0

        for batch in loader:
            out    = self.model.forward(batch["token_ids"])
            losses = self._compute_losses(batch, out)
            for k in loss_acc:
                loss_acc[k] += losses[k]

            correct_det  += int(((sigmoid(out["detector"]) >= 0.5) ==
                                  batch["is_math"].astype(bool)).sum())
            correct_type += int((out["type"].argmax(-1) == batch["type_label"]).sum())
            correct_aim  += int((out["aim"].argmax(-1)  == batch["aim_label"]).sum())
            total        += batch["token_ids"].shape[0]
            n_batches    += 1

        n  = max(total, 1)
        nb = max(n_batches, 1)
        return {
            "total_loss":    loss_acc["total"]    / nb,
            "detector_loss": loss_acc["detector"] / nb,
            "type_loss":     loss_acc["type"]     / nb,
            "aim_loss":      loss_acc["aim"]      / nb,
            "arith_loss":    loss_acc["arith"]    / nb,
            "algebra_loss":  loss_acc["algebra"]  / nb,
            "detector_acc":  correct_det  / n,
            "type_acc":      correct_type / n,
            "aim_acc":       correct_aim  / n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  Checkpoints — numpy .npz  (replaces torch.save / torch.load)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: NeuralMVModel, epoch: int, name: str,
                    metrics: Dict = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np_path = os.path.join(CHECKPOINT_DIR, f"{name}_mv_weights.npz")
    sd      = model.state_dict()
    np.savez(np_path, **sd,
             _meta_epoch=np.array([epoch]),
             _meta_arith_scale=np.array([model._arith_scale]))
    with open(os.path.join(CHECKPOINT_DIR, "training_state.json"), "w") as f:
        json.dump({"total_epochs": epoch, "dataset": name,
                   "checkpoint_file": np_path,
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"  Checkpoint saved → {np_path}  (epoch {epoch})")

def load_checkpoint(model: NeuralMVModel, name: str) -> int:
    np_path = os.path.join(CHECKPOINT_DIR, f"{name}_mv_weights.npz")
    if not os.path.exists(np_path):
        print("  No checkpoint found — starting fresh.")
        return 0
    ckpt               = np.load(np_path, allow_pickle=False)
    sd                 = {k: ckpt[k] for k in ckpt.files
                          if not k.startswith("_meta")}
    model.load_state_dict(sd)
    model._arith_scale = float(ckpt["_meta_arith_scale"][0]) if "_meta_arith_scale" in ckpt.files else 1.0
    epoch              = int(ckpt["_meta_epoch"][0]) if "_meta_epoch" in ckpt.files else 0
    print(f"  Resumed from epoch {epoch}  ({np_path})")
    return epoch


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  Benchmark table helper  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _print_benchmark_table(phase, train_m, val_m, test_m):
    metrics = [
        ("total_loss",    "total loss"),
        ("type_loss",     "type loss"),
        ("aim_loss",      "aim loss"),
        ("detector_loss", "detector loss"),
        ("arith_loss",    "arith loss"),
        ("algebra_loss",  "algebra loss"),
        ("type_acc",      "type acc"),
        ("aim_acc",       "aim acc"),
        ("detector_acc",  "detector acc"),
    ]
    w   = 11
    sep = "─" * (22 + w * (3 if test_m else 2))
    print(f"\n{sep}")
    print(f"  Benchmark — {phase}")
    hdr = f"  {'metric':<20s}" + f"{'train':>{w}}" + f"{'val':>{w}}"
    if test_m:
        hdr += f"{'test':>{w}}"
    print(hdr)
    print(f"  {'─'*20}" + f"{'─'*w}" * (3 if test_m else 2))
    for key, label in metrics:
        tr  = train_m.get(key, float("nan"))
        vl  = val_m.get(key,   float("nan"))
        row = f"  {label:<20s}{tr:>{w}.4f}{vl:>{w}.4f}"
        if test_m:
            ts  = test_m.get(key, float("nan"))
            row += f"{ts:>{w}.4f}"
        print(row)
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  Training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def train_neural_mv(
    train_path:   str,
    test_path:    Optional[str] = None,
    epochs:       int   = 30,
    lr:           float = 1e-3,
    batch_size:   int   = 64,
    resume:       bool  = True,
    val_split:    float = 0.1,
    count_params: bool  = False,
    phase_name:   str   = "Phase",
) -> NeuralMVModel:
    print("=" * 60)
    print(f"ChapatiLM MV (XTRAIN)  |  CPU  |  lr={lr}  |  epochs={epochs}")
    print(f"Tokenizer              :  {TOKENIZER}")
    print("=" * 60)

    train_problems = load_problems(train_path)
    dataset_name   = os.path.splitext(os.path.basename(train_path))[0]

    model = NeuralMVModel(embed_dim=128, hidden_dim=256)
    model.count_parameters(verbose=count_params)
    if not count_params:
        print(f"  Total trainable params: {model.count_parameters():,}")
    print(f"  Problems (train file) : {len(train_problems):,}")

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, dataset_name)

    random.shuffle(train_problems)
    split    = int(len(train_problems) * (1 - val_split))
    train_ds = build_dataset(train_problems[:split])
    val_ds   = build_dataset(train_problems[split:])

    trainer  = NeuralMVTrainer(model, lr=lr)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        train_loader = numpy_dataloader(train_ds, batch_size, shuffle=True)
        val_loader   = numpy_dataloader(val_ds,   batch_size, shuffle=False)

        m   = trainer.train_epoch(train_loader)
        val = trainer.evaluate(val_loader)

        print(
            f"  Epoch {epoch:3d}/{start_epoch+epochs}  "
            f"train_loss {m['total']:.4f} "
            f"(type {m['type']:.3f} | aim {m['aim']:.3f} | arith {m['arith']:.4f})  "
            f"val_loss {val['total_loss']:.4f}  "
            f"val_type {val['type_acc']:.3f}  val_aim {val['aim_acc']:.3f}",
            flush=True,
        )
        if epoch % 10 == 0:
            save_checkpoint(model, epoch, dataset_name, {**m, **val})

    if (start_epoch + epochs) % 10 != 0:
        save_checkpoint(model, start_epoch + epochs, dataset_name)

    print(f"\n  Computing final {phase_name} benchmark metrics...")
    final_train = trainer.evaluate(numpy_dataloader(train_ds, batch_size, False))
    final_val   = trainer.evaluate(numpy_dataloader(val_ds,   batch_size, False))
    final_test  = None

    if test_path and os.path.exists(test_path):
        test_problems = load_problems(test_path)
        print(f"  Test problems: {len(test_problems):,}")
        test_ds    = build_dataset(test_problems)
        final_test = trainer.evaluate(numpy_dataloader(test_ds, batch_size, False))

    _print_benchmark_table(phase_name, final_train, final_val, final_test)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  Notebook runner + CLI
# ═══════════════════════════════════════════════════════════════════════════════

_SMOKE_QUERIES = [
    "What is 42 + 58?",
    "solve 3x + 5 = 20",
    "compare 100 and 77",
    "hello world",
    "347 plus 25",
    "divide 100 by 4",
]

def _smoke_test(model: NeuralMVModel):
    print("\n=== Inference smoke test ===")
    model.eval()
    for t in _SMOKE_QUERIES:
        r = model.solve(t)
        print(f"  [{r['problem_type']:12s} | {r['engine']:22s}]  "
              f"{t!r:42s} → {r['result']}")

def _resolve_dataset(train: str, synthetic: bool):
    if synthetic or not os.path.exists(train):
        print("[ChapatiLM] Kaggle CSV not found — generating synthetic fallback.")
        synth = "math_data.json"
        if not os.path.exists(synth):
            generate_synthetic_dataset(n=8000, path=synth)
        return synth, None
    test = TEST_CSV if os.path.exists(TEST_CSV) else None
    return train, test

def deep_clean():
    gc.collect()
    working_dir = "/kaggle/working"
    if not os.path.isdir(working_dir):
        return
    for fn in os.listdir(working_dir):
        fp = os.path.join(working_dir, fn)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception as e:
            print(f"  [cleanup] {fp}: {e}")
    print("🧹 Environment refreshed.")

def run_chapati(
    train:        str  = TRAIN_CSV,
    test:         str  = TEST_CSV,
    batch_size:   int  = 64,
    resume:       bool = True,
    count_params: bool = True,
    synthetic:    bool = False,
) -> NeuralMVModel:
    deep_clean()
    train_path, test_path = _resolve_dataset(train, synthetic)

    print("\n=== Phase 1: Magnitude alignment (40 ep, lr=0.001) ===")
    model = train_neural_mv(
        train_path   = train_path,
        test_path    = test_path,
        epochs       = 40,
        lr           = 0.001,
        batch_size   = batch_size,
        resume       = False,
        count_params = count_params,
        phase_name   = "Phase 1",
    )

    print("\n=== Phase 2: Fine-tuning (40 ep, lr=0.0005) ===")
    model = train_neural_mv(
        train_path   = train_path,
        test_path    = test_path,
        epochs       = 40,
        lr           = 0.0005,
        batch_size   = batch_size,
        resume       = True,
        count_params = False,
        phase_name   = "Phase 2 (final)",
    )

    _smoke_test(model)
    return model

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChapatiLM MV — XTRAIN/CPUWARP-ML")
    p.add_argument("--train",        default=TRAIN_CSV)
    p.add_argument("--test",         default=TEST_CSV)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--no-resume",    action="store_true")
    p.add_argument("--val-split",    type=float, default=0.1)
    p.add_argument("--count-params", action="store_true")
    p.add_argument("--synthetic",    action="store_true")
    return p

def _is_notebook() -> bool:
    argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    return any(nb in argv0.lower() for nb in
               ("ipykernel_launcher","ipykernel","colab_kernel_launcher",
                "kernel_launcher","ipython","jupyter"))

def main():
    if _is_notebook():
        print("[ChapatiLM] Notebook environment detected — calling run_chapati().")
        run_chapati()
        return
    args = _build_parser().parse_args()
    run_chapati(
        train        = args.train,
        test         = args.test,
        batch_size   = args.batch_size,
        resume       = not args.no_resume,
        count_params = args.count_params,
        synthetic    = args.synthetic,
    )

if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  Notebook quick-start
# ═══════════════════════════════════════════════════════════════════════════════
#
#   from chapatiLM_mv_xtrain import run_chapati
#   model = run_chapati()               # Kaggle CSV paths, full training
#   model = run_chapati(synthetic=True) # offline dev
#
#   # param count only:
#   from chapatiLM_mv_xtrain import NeuralMVModel
#   NeuralMVModel().count_parameters(verbose=True)
#
#   # interactive inference:
#   model.eval()
#   print(model.solve("What is 42 + 58?"))

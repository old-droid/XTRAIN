"""
ChapatiLM MV Final: Clean Math Vision Pipeline
===============================================
ONLY components:
- TekkenTokenizer (BPE + R2L digit tokenization)
- Neural MV Components (all learnable, checkpointable)
- Neural Orchestration System (reasoning/scoring)
- ScavengerDataset (auto-finds 8000 math JSON)
- Training pipeline
"""

import sys
import os
import re
import json
import math
import random
import time
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import cpuwarp_ml
import re


# ============================================================
# TekkenTokenizer: BPE + R2L digit tokenization
# ============================================================
class TekkenTokenizer:
    def __init__(self, vocab_size: int = 130000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
            "<sep>": 4, "<cls>": 5, "<mask>": 6,
            "<audio>": 7, "<control>": 8, "<tool>": 9,
            "<image>": 10, "<video>": 11,
            "<system>": 12, "<user>": 13, "<assistant>": 14,
        }
        self.vocab = self._build_vocabulary()
        self.merges = self._build_merges()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_lookup = {merge: idx for idx, merge in enumerate(self.merges)}
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s a-zA-Z0-9]+|\s+"
        )
        print(f"Tekken Tokenizer: {len(self.vocab)} tokens, {len(self.merges)} merges")

    def _build_vocabulary(self) -> Dict[str, int]:
        vocab = dict(self.special_tokens)
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        extended = ["\u20ac","\u00a3","\u00a5","\u00a9","\u00ae","\u2122","\u00b0","\u00b1","\u00b5","\u00b7","\u00a7","\u00b6","\u2020","\u2021","\u2022","\u2026","\u2013","\u2014","\u2764","\ud83d\udd25","\ud83d\ude80","\ud83d\udca1","\ud83d\udcca","\ud83d\udd27","\ud83d\udcbb","\ud83d\udcf1","\ud83c\udf0d","\ud83d\udd12","\ud83d\udd11","\ud83d\udcc8","\ud83d\udcc9","\ud83d\udcb0"]
        for c in extended:
            if c not in vocab:
                vocab[c] = len(vocab)
        common_words = ["the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he","as","you","do","at","this","but","his","by","from","they","we","say","her","she","or","an","will","my","one","all","would","there","their","what"]
        for w in common_words:
            if w not in vocab:
                vocab[w] = len(vocab)
        subwords = ["ing","ed","s","es","ly","tion","ment","ness","ful","less","un","re","pre","dis","able","ible","al","ive","ize","ate","ify","hood","ship","dom"]
        for sw in subwords:
            if sw not in vocab:
                vocab[sw] = len(vocab)
        byte_pairs = ["th","he","in","er","an","re","on","at","en","nd","ti","es","or","te","of","ed","is","it","al","ar","st","to","ha","ng","se","ou","io","le","ve","co","me","de","hi","ri","ro","ic","ne","ea","ra","ce","li","ch","ll","be","ma","si","om","ur","ad","id"]
        for bp in byte_pairs:
            if bp not in vocab:
                vocab[bp] = len(vocab)
        import string
        for c1 in string.ascii_lowercase:
            for c2 in string.ascii_lowercase:
                if len(vocab) >= self.vocab_size:
                    break
                pair = c1 + c2
                if pair not in vocab:
                    vocab[pair] = len(vocab)
            if len(vocab) >= self.vocab_size:
                break
        return vocab

    def _build_merges(self) -> List[Tuple[str, str]]:
        return [
            ("t","h"),("h","e"),("e"," "),(" ","t"),("t","o"),("o"," "),
            (" ","a"),("a","n"),("n","d"),("d"," "),(" ","i"),("i","n"),
            ("n"," "),(" ","s"),("s"," "),(" ","f"),("f","o"),("o","r"),
            ("r"," "),(" ","w"),("w","i"),("i","t"),("t","h"),("h"," "),
            (" ","b"),("b","e"),("e"," "),(" ","y"),("y","o"),("o","u"),
            ("u"," "),(" ","c"),("c","a"),("a","n"),("n"," "),(" ","d"),
            ("d","o"),("o"," "),(" ","h"),("h","a"),("a","v"),("v","e"),
            ("e"," "),(" ","i"),("i","t"),("t"," "),(" ","t"),("t","h"),
            ("h","a"),("a","t"),("t"," "),(" ","b"),("b","y"),("y"," "),
            (" ","o"),("o","f"),("f"," "),(" ","t"),("t","h"),("h","i"),
            ("i","s"),("s"," "),(" ","a"),("a","s"),("s"," "),(" ","w"),
            ("w","e"),("e","r"),("r","e"),("e"," "),(" ","t"),("t","o"),
            ("o"," "),(" ","b"),("b","e"),("e"," "),(" ","o"),("o","r"),
            ("r"," "),(" ","n"),("n","o"),("o","t"),("t"," "),(" ","w"),
            ("w","h"),("h","i"),("i","c"),("c","h"),("h"," "),(" ","a"),
            ("a","r"),("r","e"),("e"," "),(" ","t"),("t","h"),("h","e"),
            ("e","y"),("y"," "),(" ","w"),("w","e"),("e","r"),("r","e"),
            ("e"," "),(" ","t"),("t","h"),("h","e"),("e","m"),("m"," "),
            (" ","a"),("a","n"),("n","d"),("d"," "),(" ","t"),("t","h"),
            ("h","e"),("e","i"),("i","r"),("r"," "),(" ","o"),("o","f"),
            ("f"," "),(" ","t"),("t","h"),("h","e"),("e"," "),
            ("in","g"),("ed"," "),("ly"," "),("ti","o"),("al"," "),
            ("men","t"),("nes","s"),("ful"," "),("les","s"),
            ("=","="),("!","="),("<","="),(">","="),("+","="),("-","="),
            ("*","="),("/","="),("&","&"),("|","|"),("+","+"),("-","-"),
            ("<","<"),(">",">"),("(",")"),("[","]"),("{","}"),
        ]

    def _get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        pairs = []
        prev = word[0]
        for c in word[1:]:
            pairs.append((prev, c))
            prev = c
        return pairs

    def _bpe(self, token: str) -> List[str]:
        if token in self.vocab:
            return [token]
        word = list(token)
        while len(word) > 1:
            pairs = self._get_pairs(word)
            best_pair, best_pri = None, -1
            for p in pairs:
                if p in self.merge_lookup and self.merge_lookup[p] > best_pri:
                    best_pri = self.merge_lookup[p]
                    best_pair = p
            if best_pair is None:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    merged = word[i] + word[i+1]
                    if merged in self.vocab:
                        new_word.append(merged)
                    else:
                        new_word.extend([word[i], word[i+1]])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == len(new_word):
                break
        return word

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for m in self.pattern.finditer(text):
            t = m.group()
            if t.strip():
                tokens.extend(self._bpe(t))
        if tokens:
            tokens = [self.inverse_vocab[2]] + tokens + [self.inverse_vocab[3]]
        return tokens

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(t, self.special_tokens["<unk>"]) for t in self.tokenize(text)]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, "<unk>") for tid in token_ids]
        text = "".join(tokens)
        for st in self.special_tokens:
            text = text.replace(st, "")
        return text

    def tokenize_numbers_r2l(self, text: str) -> List[str]:
        tokens = []
        num_pat = re.compile(r'\d+\.?\d*')
        last = 0
        for m in num_pat.finditer(text):
            if m.start() > last:
                tokens.extend(self.tokenize(text[last:m.start()]))
            ns = m.group()
            if '.' in ns:
                ip, dp = ns.split('.')
                tokens.extend(['<num>'] + list(reversed(ip)) + ['<dec>'] + list(reversed(dp)) + ['</num>'])
            else:
                tokens.extend(['<num>'] + list(reversed(ns)) + ['</num>'])
            last = m.end()
        if last < len(text):
            tokens.extend(self.tokenize(text[last:]))
        return tokens

    def get_vocab_size(self) -> int:
        return len(self.vocab)


# ============================================================
# Neural MV Components (all learnable, checkpointable)
# ============================================================
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 +-*/=().%^<>,!?&|~@#$:;\"'\\/\n\t"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)
MAX_SEQ_LEN = 256


def text_to_char_ids(text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    text = text.lower()[:max_len]
    ids = np.zeros(max_len, dtype=np.int32)
    for i, c in enumerate(text):
        if c in CHAR_TO_IDX:
            ids[i] = CHAR_TO_IDX[c]
    return ids


def char_ids_to_embedding(char_ids: np.ndarray, embed_matrix: np.ndarray) -> np.ndarray:
    return embed_matrix[char_ids]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# --- 1. NeuralMathDetector ---
class NeuralMathDetector:
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc1_w = np.random.randn(embed_dim, 32).astype(np.float32) * 0.1
        self.fc1_b = np.zeros(32, dtype=np.float32)
        self.fc2_w = np.random.randn(32, 1).astype(np.float32) * 0.1
        self.fc2_b = np.zeros(1, dtype=np.float32)

    def forward(self, text: str) -> float:
        char_ids = text_to_char_ids(text)
        embeds = char_ids_to_embedding(char_ids, self.char_embedding)
        pooled = embeds.mean(axis=0)
        h = gelu(cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b)
        return float(cpuwarp_ml.softmax(cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b)[0])

    def is_math_query(self, text: str, threshold: float = 0.5) -> bool:
        return self.forward(text) >= threshold

    def get_weights(self) -> Dict:
        return {"char_embedding": self.char_embedding.copy(), "fc1_w": self.fc1_w.copy(),
                "fc1_b": self.fc1_b.copy(), "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy()}

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()


# --- 2. NeuralTypeClassifier ---
class NeuralTypeClassifier:
    TYPE_NAMES = ["Arithmetic", "Algebraic", "Comparison", "Geometric", "Unknown"]
    NUM_TYPES = 5

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc1_w = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.1
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, self.NUM_TYPES).astype(np.float32) * 0.1
        self.fc2_b = np.zeros(self.NUM_TYPES, dtype=np.float32)

    def forward(self, text: str) -> np.ndarray:
        char_ids = text_to_char_ids(text)
        embeds = char_ids_to_embedding(char_ids, self.char_embedding)
        pooled = embeds.mean(axis=0)
        h = gelu(cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b)
        logits = cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b
        return cpuwarp_ml.softmax(logits)

    def classify(self, text: str) -> str:
        return self.TYPE_NAMES[int(np.argmax(self.forward(text)))]

    def get_probs(self, text: str) -> Dict[str, float]:
        return {n: float(self.forward(text)[i]) for i, n in enumerate(self.TYPE_NAMES)}

    def get_weights(self) -> Dict:
        return {"char_embedding": self.char_embedding.copy(), "fc1_w": self.fc1_w.copy(),
                "fc1_b": self.fc1_b.copy(), "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy()}

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()


# --- 3. NeuralAimClassifier ---
class NeuralAimClassifier:
    AIM_NAMES = ["Calculate", "Simplify", "Solve", "Compare", "Evaluate", "Unknown"]
    NUM_AIMS = 6

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc1_w = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.1
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, self.NUM_AIMS).astype(np.float32) * 0.1
        self.fc2_b = np.zeros(self.NUM_AIMS, dtype=np.float32)

    def forward(self, text: str) -> np.ndarray:
        char_ids = text_to_char_ids(text)
        embeds = char_ids_to_embedding(char_ids, self.char_embedding)
        pooled = embeds.mean(axis=0)
        h = gelu(cpuwarp_ml.matmul(pooled, self.fc1_w) + self.fc1_b)
        logits = cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b
        return cpuwarp_ml.softmax(logits)

    def identify(self, text: str) -> str:
        return self.AIM_NAMES[int(np.argmax(self.forward(text)))]

    def get_probs(self, text: str) -> Dict[str, float]:
        return {n: float(self.forward(text)[i]) for i, n in enumerate(self.AIM_NAMES)}

    def get_weights(self) -> Dict:
        return {"char_embedding": self.char_embedding.copy(), "fc1_w": self.fc1_w.copy(),
                "fc1_b": self.fc1_b.copy(), "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy()}

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()


# --- 4. NeuralSymbolicRouter ---
class NeuralSymbolicRouter:
    ENGINE_NAMES = ["Native_Compute_Engine", "SymPy_Engine"]

    def __init__(self):
        self.fc_w = np.random.randn(11, 16).astype(np.float32) * 0.1
        self.fc_b = np.zeros(16, dtype=np.float32)
        self.out_w = np.random.randn(16, 2).astype(np.float32) * 0.1
        self.out_b = np.zeros(2, dtype=np.float32)

    def forward(self, type_probs: np.ndarray, aim_probs: np.ndarray) -> np.ndarray:
        x = np.concatenate([type_probs, aim_probs])
        h = gelu(cpuwarp_ml.matmul(x, self.fc_w) + self.fc_b)
        logits = cpuwarp_ml.matmul(h, self.out_w) + self.out_b
        e = np.exp(logits - np.max(logits))
        return e / (e.sum() + 1e-10)

    def route(self, type_probs: np.ndarray, aim_probs: np.ndarray) -> str:
        return self.ENGINE_NAMES[int(np.argmax(self.forward(type_probs, aim_probs)))]

    def get_weights(self) -> Dict:
        return {"fc_w": self.fc_w.copy(), "fc_b": self.fc_b.copy(),
                "out_w": self.out_w.copy(), "out_b": self.out_b.copy()}

    def load_weights(self, w: Dict):
        self.fc_w = w["fc_w"].copy()
        self.fc_b = w["fc_b"].copy()
        self.out_w = w["out_w"].copy()
        self.out_b = w["out_b"].copy()


# --- 5. NeuralMathFilter ---
class NeuralMathFilter:
    def __init__(self, embed_dim: int = 32):
        self.char_embedding = np.random.randn(CHAR_VOCAB_SIZE, embed_dim).astype(np.float32) * 0.02
        self.fc_w = np.random.randn(embed_dim, 1).astype(np.float32) * 0.1
        self.fc_b = np.zeros(1, dtype=np.float32)

    def filter(self, text: str, threshold: float = 0.5) -> str:
        text = text.lower()
        result = []
        for c in text:
            if c in CHAR_TO_IDX:
                emb = self.char_embedding[CHAR_TO_IDX[c]]
                score = float(cpuwarp_ml.softmax(cpuwarp_ml.matmul(emb, self.fc_w) + self.fc_b)[0])
                if score >= threshold or c == ' ':
                    result.append(c)
            elif c == ' ':
                result.append(c)
        return re.sub(r'\s+', ' ', ''.join(result)).strip()

    def get_weights(self) -> Dict:
        return {"char_embedding": self.char_embedding.copy(),
                "fc_w": self.fc_w.copy(), "fc_b": self.fc_b.copy()}

    def load_weights(self, w: Dict):
        self.char_embedding = w["char_embedding"].copy()
        self.fc_w = w["fc_w"].copy()
        self.fc_b = w["fc_b"].copy()


# --- 6. NeuralOperatorMapper ---
class NeuralOperatorMapper:
    OPERATORS = ["+", "-", "*", "/", "=", "^", "%", ">", "<", "**"]
    KNOWN_WORDS = ["plus","add","sum","added","minus","subtract","difference","less","spends",
                    "times","multiply","product","multiplied","divided","divide","quotient","over",
                    "power","raised","squared","cubed","mod","modulo","remainder",
                    "equals","equal","is","gives","greater","gt","lt"]

    def __init__(self, embed_dim: int = 32):
        self.embed_dim = embed_dim
        self.word_embeddings = {}
        for w in self.KNOWN_WORDS:
            self.word_embeddings[w] = np.random.randn(embed_dim).astype(np.float32) * 0.1
        self.op_embeddings = np.random.randn(len(self.OPERATORS), embed_dim).astype(np.float32) * 0.1
        self.proj_w = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
        self.proj_b = np.zeros(embed_dim, dtype=np.float32)

    def _word_to_vec(self, word: str) -> np.ndarray:
        word = word.lower().strip()
        if word in self.word_embeddings:
            return self.word_embeddings[word]
        import hashlib
        idx = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % CHAR_VOCAB_SIZE
        return np.random.RandomState(idx).randn(self.embed_dim).astype(np.float32) * 0.1

    def map_word(self, word: str) -> str:
        word = word.lower().strip()
        if word in "+-*/=^%><":
            return word
        w_vec = self._word_to_vec(word)
        projected = gelu(cpuwarp_ml.matmul(w_vec, self.proj_w) + self.proj_b)
        return self.OPERATORS[int(np.argmax(cpuwarp_ml.matmul(projected, self.op_embeddings.T)))]

    def apply_to_text(self, text: str) -> str:
        words = re.findall(r'[a-zA-Z]+|[^\s]+', text.lower())
        result = []
        for w in words:
            if re.match(r'^[a-z]+$', w):
                result.append(self.map_word(w))
            else:
                result.append(w)
        return re.sub(r'\s+', ' ', ' '.join(result)).strip()

    def get_weights(self) -> Dict:
        return {"op_embeddings": self.op_embeddings.copy(),
                "proj_w": self.proj_w.copy(), "proj_b": self.proj_b.copy()}

    def load_weights(self, w: Dict):
        self.op_embeddings = w["op_embeddings"].copy()
        self.proj_w = w["proj_w"].copy()
        self.proj_b = w["proj_b"].copy()


# --- 7. NeuralArithmeticSolver ---
class NeuralArithmeticSolver:
    def __init__(self, hidden_dim: int = 128):
        self.fc1_w = np.random.randn(6, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 64).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(64, dtype=np.float32)
        self.fc3_w = np.random.randn(64, 1).astype(np.float32) * 0.01
        self.fc3_b = np.zeros(1, dtype=np.float32)

    def _extract_features(self, expression: str) -> Optional[np.ndarray]:
        cleaned = re.sub(r'\s+', '', expression).replace('^', '**')
        match = re.match(r'^(-?\d+\.?\d*)([\+\-\*/\*\*])(-?\d+\.?\d*)$', cleaned)
        if not match:
            return None
        n1_str, op, n2_str = match.groups()
        try:
            n1, n2 = float(n1_str), float(n2_str)
        except ValueError:
            return None
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3, '**': 4}
        oh = np.zeros(5); oh[op_map.get(op, 0)] = 1.0
        return np.array([n1, n2, oh[0], oh[1], oh[2], oh[3]], dtype=np.float32)

    def predict(self, expression: str) -> Optional[float]:
        features = self._extract_features(expression)
        if features is None:
            return None
        h = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        h = gelu(cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b)
        return float((cpuwarp_ml.matmul(h, self.fc3_w) + self.fc3_b)[0])

    def solve(self, expression: str) -> Optional[str]:
        result = self.predict(expression)
        if result is None:
            return None
        return str(int(result)) if result == int(result) else f"{result:.6g}"

    def get_weights(self) -> Dict:
        return {"fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
                "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
                "fc3_w": self.fc3_w.copy(), "fc3_b": self.fc3_b.copy()}

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()
        self.fc3_w = w["fc3_w"].copy()
        self.fc3_b = w["fc3_b"].copy()


# --- 8. NeuralAlgebraicSolver ---
class NeuralAlgebraicSolver:
    def __init__(self, hidden_dim: int = 64):
        self.fc1_w = np.random.randn(3, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 32).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(32, dtype=np.float32)
        self.fc3_w = np.random.randn(32, 1).astype(np.float32) * 0.01
        self.fc3_b = np.zeros(1, dtype=np.float32)

    def _parse_linear(self, expression: str) -> Optional[Tuple[float, float, float]]:
        match = re.match(r'([\d\.\+\-\*/\*\s]*)\s*([a-zA-Z])\s*([\+\-\d\.\*/\s]*)\s*=\s*([\d\.\+\-\*/\s]*)', expression)
        if not match:
            return None
        try:
            lhs_coef, var, lhs_const, rhs = match.groups()
            lhs_coef = lhs_coef.strip()
            if not lhs_coef or lhs_coef in ('+', '-'):
                coef = 1.0 if not lhs_coef or lhs_coef == '+' else -1.0
            else:
                coef = float(eval(lhs_coef, {"__builtins__": {}}, {}))
            const = float(eval(lhs_const.strip(), {"__builtins__": {}}, {})) if lhs_const.strip() else 0.0
            rhs_val = float(eval(rhs.strip(), {"__builtins__": {}}, {}))
            return (coef, const, rhs_val)
        except Exception:
            return None

    def predict(self, expression: str) -> Optional[float]:
        parsed = self._parse_linear(expression)
        if parsed is None:
            return None
        a, b, c = parsed
        features = np.array([a, b, c], dtype=np.float32)
        h = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        h = gelu(cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b)
        return float((cpuwarp_ml.matmul(h, self.fc3_w) + self.fc3_b)[0])

    def solve(self, expression: str) -> Optional[str]:
        result = self.predict(expression)
        if result is None:
            return None
        return f"x = {int(result)}" if result == int(result) else f"x = {result:.6g}"

    def get_weights(self) -> Dict:
        return {"fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
                "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy(),
                "fc3_w": self.fc3_w.copy(), "fc3_b": self.fc3_b.copy()}

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()
        self.fc3_w = w["fc3_w"].copy()
        self.fc3_b = w["fc3_b"].copy()


# --- 9. NeuralComparisonSolver ---
class NeuralComparisonSolver:
    def __init__(self, hidden_dim: int = 32):
        self.fc1_w = np.random.randn(2, hidden_dim).astype(np.float32) * 0.01
        self.fc1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.fc2_w = np.random.randn(hidden_dim, 3).astype(np.float32) * 0.01
        self.fc2_b = np.zeros(3, dtype=np.float32)

    def predict(self, a: float, b: float) -> np.ndarray:
        features = np.array([a, b], dtype=np.float32)
        h = gelu(cpuwarp_ml.matmul(features, self.fc1_w) + self.fc1_b)
        logits = cpuwarp_ml.matmul(h, self.fc2_w) + self.fc2_b
        e = np.exp(logits - np.max(logits))
        return e / (e.sum() + 1e-10)

    def solve(self, expression: str) -> Optional[str]:
        match = re.search(r'(\d+\.?\d*)\s*([><=!]+)\s*(\d+\.?\d*)', expression)
        if match:
            left, op, right = float(match.group(1)), match.group(2), float(match.group(3))
            probs = self.predict(left, right)
            ops = [">", "<", "="]
            return f"{left} {ops[int(np.argmax(probs))]} {right} ({probs.max():.3f})"
        match = re.search(r'compare\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)', expression, re.IGNORECASE)
        if match:
            left, right = float(match.group(1)), float(match.group(2))
            probs = self.predict(left, right)
            ops = [">", "<", "="]
            return f"{left} {ops[int(np.argmax(probs))]} {right} ({probs.max():.3f})"
        return None

    def get_weights(self) -> Dict:
        return {"fc1_w": self.fc1_w.copy(), "fc1_b": self.fc1_b.copy(),
                "fc2_w": self.fc2_w.copy(), "fc2_b": self.fc2_b.copy()}

    def load_weights(self, w: Dict):
        self.fc1_w = w["fc1_w"].copy()
        self.fc1_b = w["fc1_b"].copy()
        self.fc2_w = w["fc2_w"].copy()
        self.fc2_b = w["fc2_b"].copy()


# ============================================================
# Unified NeuralMVModel
# ============================================================
class NeuralMVModel:
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        self.detector = NeuralMathDetector(embed_dim)
        self.type_classifier = NeuralTypeClassifier(embed_dim, hidden_dim)
        self.aim_classifier = NeuralAimClassifier(embed_dim, hidden_dim)
        self.symbolic_router = NeuralSymbolicRouter()
        self.math_filter = NeuralMathFilter(embed_dim)
        self.op_mapper = NeuralOperatorMapper(embed_dim)
        self.arith_solver = NeuralArithmeticSolver(hidden_dim)
        self.algebra_solver = NeuralAlgebraicSolver(hidden_dim)
        self.comparison_solver = NeuralComparisonSolver(hidden_dim)
        self.solve_history = []

    def is_math_query(self, text: str) -> bool:
        return self.detector.is_math_query(text)

    def math_confidence(self, text: str) -> float:
        return self.detector.forward(text)

    def solve(self, query: str) -> Dict:
        problem_type = self.type_classifier.classify(query)
        aim = self.aim_classifier.identify(query)
        type_probs = self.type_classifier.forward(query)
        aim_probs = self.aim_classifier.forward(query)
        engine = self.symbolic_router.route(type_probs, aim_probs)

        result = None
        if engine == "Native_Compute_Engine":
            if problem_type == "Comparison" or aim == "Compare":
                result = self.comparison_solver.solve(query)
            if result is None:
                cleaned = self.math_filter.filter(query)
                symbolic = self.op_mapper.apply_to_text(cleaned)
                result = self.arith_solver.solve(symbolic)
        elif engine == "SymPy_Engine":
            if problem_type == "Comparison":
                result = self.comparison_solver.solve(query)
            if result is None:
                result = self.algebra_solver.solve(query)
            if result is None:
                cleaned = self.math_filter.filter(query)
                symbolic = self.op_mapper.apply_to_text(cleaned)
                result = self.arith_solver.solve(symbolic)

        if result is None:
            result = "Unable to solve"

        solution = {
            "query": query, "problem_type": problem_type, "aim": aim,
            "engine": engine, "result": result,
            "tokenized": {"original": query, "cleaned": self.math_filter.filter(query),
                          "symbolic": self.op_mapper.apply_to_text(query)},
            "type_probs": {k: float(v) for k, v in self.type_classifier.get_probs(query).items()},
            "aim_probs": {k: float(v) for k, v in self.aim_classifier.get_probs(query).items()},
        }
        self.solve_history.append(solution)
        return solution

    def get_all_weights(self) -> Dict:
        return {
            "math_detector": self.detector.get_weights(),
            "type_classifier": self.type_classifier.get_weights(),
            "aim_classifier": self.aim_classifier.get_weights(),
            "symbolic_router": self.symbolic_router.get_weights(),
            "math_filter": self.math_filter.get_weights(),
            "op_mapper": self.op_mapper.get_weights(),
            "arith_solver": self.arith_solver.get_weights(),
            "algebra_solver": self.algebra_solver.get_weights(),
            "comparison_solver": self.comparison_solver.get_weights(),
        }

    def load_all_weights(self, weights: Dict):
        self.detector.load_weights(weights.get("math_detector", {}))
        self.type_classifier.load_weights(weights.get("type_classifier", {}))
        self.aim_classifier.load_weights(weights.get("aim_classifier", {}))
        self.symbolic_router.load_weights(weights.get("symbolic_router", {}))
        self.math_filter.load_weights(weights.get("math_filter", {}))
        self.op_mapper.load_weights(weights.get("op_mapper", {}))
        self.arith_solver.load_weights(weights.get("arith_solver", {}))
        self.algebra_solver.load_weights(weights.get("algebra_solver", {}))
        self.comparison_solver.load_weights(weights.get("comparison_solver", {}))

    def count_weights(self) -> int:
        total = 0
        for cw in self.get_all_weights().values():
            for arr in cw.values():
                total += arr.size
        return total


# ============================================================
# NeuralMVTrainer
# ============================================================
class NeuralMVTrainer:
    def __init__(self, model: NeuralMVModel, lr: float = 0.001):
        self.model = model
        self.lr = lr

    def train_detector(self, texts: List[str], labels: List[int], epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0
            for text, label in zip(texts, labels):
                pred = self.model.detector.forward(text)
                loss = (pred - label) ** 2
                total_loss += loss
                eps = 1e-4
                for attr in ["fc1_w", "fc1_b", "fc2_w", "fc2_b"]:
                    arr = getattr(self.model.detector, attr)
                    flat = arr.flatten()
                    for i in range(min(len(flat), 50)):
                        idx = np.unravel_index(i, arr.shape)
                        orig = arr[idx]
                        arr[idx] = orig + eps
                        pred_plus = self.model.detector.forward(text)
                        loss_plus = (pred_plus - label) ** 2
                        grad = (loss_plus - loss) / eps
                        arr[idx] = orig - self.lr * grad
            if (epoch + 1) % 5 == 0:
                print(f"  Detector epoch {epoch+1}/{epochs}, loss: {total_loss/len(texts):.4f}")

    def train_type_classifier(self, texts: List[str], labels: List[int], epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0
            for text, label in zip(texts, labels):
                probs = self.model.type_classifier.forward(text)
                target = np.zeros(5); target[label] = 1.0
                loss = -np.sum(target * np.log(probs + 1e-10))
                total_loss += loss
                eps = 1e-4
                for attr in ["fc1_w", "fc1_b", "fc2_w", "fc2_b"]:
                    arr = getattr(self.model.type_classifier, attr)
                    flat = arr.flatten()
                    for i in range(min(len(flat), 50)):
                        idx = np.unravel_index(i, arr.shape)
                        orig = arr[idx]
                        arr[idx] = orig + eps
                        probs_plus = self.model.type_classifier.forward(text)
                        loss_plus = -np.sum(target * np.log(probs_plus + 1e-10))
                        grad = (loss_plus - loss) / eps
                        arr[idx] = orig - self.lr * grad
            if (epoch + 1) % 5 == 0:
                print(f"  Type epoch {epoch+1}/{epochs}, loss: {total_loss/len(texts):.4f}")

    def train_arith_solver(self, expressions: List[str], answers: List[float], epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            for expr, answer in zip(expressions, answers):
                pred = self.model.arith_solver.predict(expr)
                if pred is None:
                    continue
                loss = (pred - answer) ** 2
                total_loss += loss
                count += 1
                eps = 1e-4
                for attr in ["fc1_w", "fc1_b", "fc2_w", "fc2_b", "fc3_w", "fc3_b"]:
                    arr = getattr(self.model.arith_solver, attr)
                    flat = arr.flatten()
                    for i in range(min(len(flat), 50)):
                        idx = np.unravel_index(i, arr.shape)
                        orig = arr[idx]
                        arr[idx] = orig + eps
                        pred_plus = self.model.arith_solver.predict(expr)
                        if pred_plus is None:
                            arr[idx] = orig
                            continue
                        loss_plus = (pred_plus - answer) ** 2
                        grad = (loss_plus - loss) / eps
                        arr[idx] = orig - self.lr * grad
            if count > 0 and (epoch + 1) % 5 == 0:
                print(f"  Arith epoch {epoch+1}/{epochs}, loss: {total_loss/count:.4f}")


# ============================================================
# Neural Orchestration System (reasoning)
# ============================================================
class NeuralOrchestrationSystem:
    def __init__(self, num_workers: int = 8, num_neurons: int = 16,
                 max_retries: int = 4, d_model: int = 1024):
        self.num_workers = num_workers
        self.num_neurons = num_neurons
        self.max_retries = min(max_retries, num_neurons)
        self.d_model = d_model
        self._init_components()
        self.metrics = {"worker_outputs": 0, "orchestrator_scores": 0,
                       "manager_routing_decisions": 0, "safety_filter_activations": 0,
                       "verifier_acceptances": 0, "verifier_rejections": 0,
                       "retry_attempts": 0, "retry_successes": 0, "unsafe_content_blocked": 0}
        print(f"Neural Orchestration: {num_workers} workers, {num_neurons} neurons, {max_retries} retries")

    def _init_components(self):
        self.worker_nodes = [{"weights": np.random.randn(self.d_model, self.d_model).astype(np.float32)*0.02,
                              "bias": np.random.randn(self.d_model).astype(np.float32)*0.02,
                              "activation": "gelu"} for _ in range(self.num_workers)]
        self.orchestrator = {
            "scoring_weights": np.random.randn(self.d_model, self.num_neurons).astype(np.float32)*0.01,
            "routing_weights": np.random.randn(self.d_model, self.num_neurons).astype(np.float32)*0.01,
            "composite_weights": np.random.randn(self.num_neurons*2, 1).astype(np.float32)*0.01,
        }
        self.manager_node = {"decision_threshold": 0.7,
                            "selection_weights": np.random.randn(self.num_neurons, 1).astype(np.float32)*0.01}
        self.safety_guardrail = {
            "query_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32)*0.02,
            "key_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32)*0.02,
            "value_weights": np.random.randn(self.d_model, self.d_model).astype(np.float32)*0.02,
            "bad_matrices": np.random.randn(self.d_model, 10).astype(np.float32)*0.1,
            "safety_threshold": 0.8,
        }
        self.verifier = {
            "normalization_factor": 1.0,
            "aggregation_weights": np.random.randn(4, 1).astype(np.float32)*0.01,
            "acceptance_threshold": 0.3,
        }
        self.retry_policy = {"retry_counter": 0, "max_retries": self.max_retries, "retry_decay": 0.9}

    def get_state(self) -> dict:
        return {"num_workers": self.num_workers, "num_neurons": self.num_neurons,
                "max_retries": self.max_retries, "d_model": self.d_model,
                "worker_nodes": [{"weights": n["weights"].copy(), "bias": n["bias"].copy(),
                                  "activation": n["activation"]} for n in self.worker_nodes],
                "orchestrator": {k: v.copy() for k, v in self.orchestrator.items()},
                "manager_node": {k: v.copy() if isinstance(v, np.ndarray) else v
                                 for k, v in self.manager_node.items()},
                "safety_guardrail": {k: v.copy() if isinstance(v, np.ndarray) else v
                                     for k, v in self.safety_guardrail.items()},
                "verifier": {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in self.verifier.items()},
                "retry_policy": dict(self.retry_policy),
                "orchestration_metrics": dict(self.metrics)}

    def restore_state(self, state: dict):
        self.num_workers = state["num_workers"]
        self.num_neurons = state["num_neurons"]
        self.max_retries = state["max_retries"]
        self.d_model = state["d_model"]
        self.worker_nodes = state["worker_nodes"]
        self.orchestrator = {k: v.copy() for k, v in state["orchestrator"].items()}
        self.manager_node = {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in state["manager_node"].items()}
        self.safety_guardrail = {k: v.copy() if isinstance(v, np.ndarray) else v
                                 for k, v in state["safety_guardrail"].items()}
        self.verifier = {k: v.copy() if isinstance(v, np.ndarray) else v
                         for k, v in state["verifier"].items()}
        self.retry_policy = dict(state["retry_policy"])
        if "orchestration_metrics" in state:
            self.metrics.update(state["orchestration_metrics"])


# ============================================================
# ScavengerDataset - auto-finds 8000 math JSON
# ============================================================
class ScavengerDataset:
    def __init__(self, max_size: int = 8000, min_quality: float = 0.7,
                 auto_scavenge: bool = True, dataset_path: Optional[str] = None):
        self.max_size = max_size
        self.min_quality = min_quality
        self.samples = []
        self.sources_used = []
        self.quality_scores = []

        if dataset_path:
            self._load_path(dataset_path)
        elif auto_scavenge:
            self._auto_find_math_json()

    def _auto_find_math_json(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [base_dir, os.path.join(base_dir, "data"), os.path.join(base_dir, "datasets")]
        for sd in search_dirs:
            if os.path.isdir(sd):
                for f in os.listdir(sd):
                    if f.endswith('.json') and ('math' in f.lower() or 'synthetic' in f.lower()):
                        fp = os.path.join(sd, f)
                        self._load_json(fp)
                        self.sources_used.append(fp)
                        print(f"  Auto-found: {fp} ({len(self.samples)} samples)")
                        return
        print("  No math JSON found. Generating synthetic dataset...")
        self._generate_synthetic()

    def _load_path(self, path: str):
        if os.path.exists(path):
            self._load_json(path)
            self.sources_used.append(path)
            print(f"  Loaded: {path} ({len(self.samples)} samples)")
        else:
            print(f"  Path not found: {path}. Generating synthetic...")
            self._generate_synthetic()

    def _load_json(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        problems = data.get("problems", [])
        for p in problems[:self.max_size]:
            text = f"<math>{p['problem']}</math>"
            answer = p.get("answer", "")
            self.samples.append(text)
            self.quality_scores.append(0.95)

    def _generate_synthetic(self):
        from synthetic_math_dataset import SyntheticMathDatasetGenerator
        gen = SyntheticMathDatasetGenerator(seed=42)
        batch = gen.generate_batch(self.max_size)
        stats = gen.get_statistics()
        print(f"  Generated: {stats['total_generated']} problems")
        print(f"  Categories: {stats['category_distribution']}")
        for p in batch[:self.max_size]:
            self.samples.append(f"<math>{p['problem']}</math>")
            self.quality_scores.append(0.95)
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_math_dataset.json")
        gen.save_json(output)
        self.sources_used.append(output)

    def get_sample_count(self) -> int:
        return len(self.samples)

    def get_samples(self) -> List[str]:
        return self.samples

    def get_quality_analysis(self) -> Dict:
        return {"overall_quality": sum(self.quality_scores)/max(1, len(self.quality_scores)),
                "quality_distribution": {"high": len([q for q in self.quality_scores if q >= 0.8]),
                                         "medium": len([q for q in self.quality_scores if 0.5 <= q < 0.8]),
                                         "low": len([q for q in self.quality_scores if q < 0.5])},
                "sources_used": len(self.sources_used)}

    def print_quality_report(self):
        qa = self.get_quality_analysis()
        print(f"  Samples: {self.get_sample_count()}")
        print(f"  Quality: {qa['overall_quality']:.2f}")
        print(f"  Sources: {qa['sources_used']}")


# ============================================================
# Checkpoint Manager
# ============================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

def find_latest_checkpoint() -> Optional[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith("_mv_weights.pkl")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    return os.path.join(CHECKPOINT_DIR, ckpts[0])

def load_checkpoint_state() -> Optional[Dict]:
    state_path = os.path.join(CHECKPOINT_DIR, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return None

def save_checkpoint(model: NeuralMVModel, total_epochs: int, dataset_name: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    mv_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_mv_weights.pkl")
    with open(mv_path, "wb") as f:
        pickle.dump(model.get_all_weights(), f)
    state = {
        "total_epochs": total_epochs,
        "dataset": dataset_name,
        "checkpoint_file": mv_path,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"Checkpoint saved: {mv_path} (epoch {total_epochs})")


# ============================================================
# Training Pipeline
# ============================================================
def train_neural_mv(dataset_path: str = "synthetic_math_dataset.json",
                    epochs: int = 5, lr: float = 0.01, resume: bool = True):
    print("=" * 60)
    print("Training Neural MV Pipeline")
    print("=" * 60)

    with open(dataset_path, "r") as f:
        data = json.load(f)
    problems = data["problems"]
    print(f"Loaded {len(problems)} math problems")

    model = NeuralMVModel(embed_dim=64, hidden_dim=128)
    start_epoch = 0

    if resume:
        ckpt = find_latest_checkpoint()
        state = load_checkpoint_state()
        if ckpt and state:
            print(f"\nLoading checkpoint: {ckpt}")
            with open(ckpt, "rb") as f:
                weights = pickle.load(f)
            model.load_all_weights(weights)
            start_epoch = state.get("total_epochs", 0)
            print(f"Resuming from epoch {start_epoch}")

    trainer = NeuralMVTrainer(model, lr=lr)
    print(f"Learnable parameters: {model.count_weights():,}")

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Phase 1: Math Detector
    print("\n[1/4] Training Math Detector...")
    math_texts = [p["problem"] for p in problems[:500]]
    math_labels = [1] * 500
    non_math = ["Hello how are you","The quick brown fox","What is the capital of France",
                "I love programming","The weather is nice","Tell me a story",
                "How do I bake a cake","Describe photosynthesis","Can you help me write",
                "What is the meaning of life"]
    non_labels = [0] * len(non_math)
    trainer.train_detector(math_texts + non_math, math_labels + non_labels, epochs=epochs)

    for t in ["5 + 3", "Hello world", "Solve for x: 2x = 10"]:
        print(f"  '{t}' -> math: {model.detector.forward(t):.3f}")

    # Phase 2: Type Classifier
    print("\n[2/4] Training Type Classifier...")
    type_map = {"Number Theory": 0, "Algebra": 1, "Combinatorics": 0,
                "Geometry": 3, "Inequalities": 0, "Sequences": 0, "Diophantine Equations": 1}
    type_texts = [p["problem"] for p in problems[:1000]]
    type_labels = [type_map.get(p["category"], 4) for p in problems[:1000]]
    trainer.train_type_classifier(type_texts, type_labels, epochs=min(epochs, 3))

    for t in ["Find the remainder when 50^56 is divided by 23",
              "Solve the system: 2x + 7y = 35",
              "Find the area of a circle with radius 5"]:
        probs = model.type_classifier.get_probs(t)
        print(f"  '{t[:50]}...' -> {max(probs, key=probs.get)} ({max(probs.values()):.3f})")

    # Phase 3: Arithmetic Solver
    print("\n[3/4] Training Arithmetic Solver...")
    arith_exprs, arith_answers = [], []
    for _ in range(1000):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            arith_exprs.append(f"{a}+{b}"); arith_answers.append(float(a + b))
        elif op == "-":
            if a >= b: arith_exprs.append(f"{a}-{b}"); arith_answers.append(float(a - b))
            else: arith_exprs.append(f"{b}-{a}"); arith_answers.append(float(b - a))
        else:
            a, b = random.randint(1, 20), random.randint(1, 20)
            arith_exprs.append(f"{a}*{b}"); arith_answers.append(float(a * b))
    trainer.train_arith_solver(arith_exprs[:500], arith_answers[:500], epochs=min(epochs, 3))

    for expr in ["10+5", "20-3", "6*7"]:
        pred = model.arith_solver.predict(expr)
        if pred is not None:
            print(f"  {expr} -> {pred:.2f}")

    # Phase 4: Evaluate
    print("\n[4/4] Evaluation on held-out set...")
    test = problems[7500:7600]
    correct, total = 0, 0
    for p in test:
        result = model.solve(p["problem"])
        total += 1
        try:
            pred_str = str(result["result"])
            if "=" in pred_str:
                pred_val = float(pred_str.split("=")[-1].strip())
            else:
                pred_val = float(pred_str)
            true_val = float(p["answer"])
            if abs(pred_val - true_val) / (abs(true_val) + 1e-8) < 0.01:
                correct += 1
        except (ValueError, IndexError, AttributeError):
            pass
    acc = correct / total if total > 0 else 0
    print(f"  Accuracy: {correct}/{total} = {acc:.1%}")

    total_epochs = start_epoch + epochs
    save_checkpoint(model, total_epochs, dataset_name)
    print(f"Total parameters: {model.count_weights():,}")
    return model


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 1. Generate/find 8000 math problems
    print("\n=== ScavengerDataset: Auto-finding math JSON ===")
    dataset = ScavengerDataset(max_size=8000, auto_scavenge=True)
    print(f"Samples loaded: {dataset.get_sample_count()}")

    # 2. Test tokenizer
    print("\n=== Tokenizer Test ===")
    tok = TekkenTokenizer(vocab_size=130000)
    test_text = "What is 25 plus 17?"
    tokens = tok.tokenize(test_text)
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids[:10]}...")
    print(f"Decoded: {decoded}")
    r2l = tok.tokenize_numbers_r2l("value is 1234 and 56.78")
    print(f"R2L: {r2l}")

    # 3. Train Neural MV
    print("\n=== Neural MV Training ===")
    dataset_path = dataset.sources_used[0] if dataset.sources_used else None
    if dataset_path is None:
        for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if f.endswith('.json') and ('math' in f.lower() or 'synthetic' in f.lower()):
                dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
                break
    if dataset_path is None:
        print("ERROR: No math dataset found. Exiting.")
        sys.exit(1)
    print(f"Training with: {dataset_path}")

    ckpt = find_latest_checkpoint()
    state = load_checkpoint_state()
    if ckpt and state:
        print(f"Found existing checkpoint: {ckpt} (epoch {state['total_epochs']})")
    else:
        print("No checkpoint found. Starting fresh training.")

    mv_model = train_neural_mv(
        dataset_path=dataset_path,
        epochs=5, lr=0.01, resume=True,
    )

    # 4. Test trained MV model
    print("\n=== Post-Training Test ===")
    test_queries = [
        "What is 25 plus 17?",
        "Solve for x: 2x + 3 = 11",
        "Compare 5.5 and 3.2",
        "Find the least common multiple of 71 and 141",
    ]
    for q in test_queries:
        is_math = mv_model.is_math_query(q)
        conf = mv_model.math_confidence(q)
        result = mv_model.solve(q)
        print(f"\nQ: {q}")
        print(f"  Math: {is_math} ({conf:.3f})")
        print(f"  Type: {result['problem_type']} | Aim: {result['aim']}")
        print(f"  Engine: {result['engine']}")
        print(f"  Result: {result['result']}")

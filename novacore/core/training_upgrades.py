"""Training-time learning upgrades for NovaCore.
All zero external dependency, pure numpy + built-ins.

Upgrades:
1. SVD Embeddings — semantic word vectors from co-occurrence
2. IDF Weighting — smart word importance
3. Pattern Confidence — Z-score ranking
4. Multi-Head Reservoir — topic clustering
5. 8-bit Quantization — memory-efficient storage
6. Semantic Search — cosine similarity retrieval
"""
import json
import math
import os
import pickle
from collections import Counter, defaultdict
import numpy as np


class TrainingUpgrader:
    def __init__(self, vocab, config=None):
        self.vocab = vocab
        self.config = config or {}
        up = self.config.get('training_upgrades', {})
        self.enabled = up.get('enable_svd_embeddings', False)

        if not self.enabled:
            return

        self.svd_vocab_size = up.get('svd_vocab_size', 5000)
        self.svd_dim = up.get('svd_dim', 128)
        self.window = up.get('cooccurrence_window', 5)
        self.idf_enabled = up.get('enable_idf_weighting', True)
        self.hierarchy_enabled = up.get('enable_pattern_hierarchy', True)
        self.confidence_enabled = up.get('enable_confidence_scoring', True)
        self.multi_head_enabled = up.get('enable_multi_head_reservoir', True)
        self.quantize_enabled = up.get('quantize_embeddings', True)
        self.quantize_bits = up.get('quantize_bit_width', 8)
        self.semantic_search_enabled = up.get('enable_semantic_search', True)
        self.num_heads = up.get('num_reservoir_heads', 5)

        # Storage
        self.word_to_idx = {}
        self.cooccurrence = None
        self.embeddings = None          # float32 (vocab_size, svd_dim)
        self.quantized_embeddings = None # uint8 (vocab_size, svd_dim)
        self.quant_scale = None         # per-row scale for dequant
        self.idf_weights = {}
        self.confidence_scores = {}
        self.head_assignments = []
        self.reservoir_heads = []
        self.collocations = []

    # ------------------------------------------------------------------
    # 1. Co-occurrence + SVD Embeddings
    # ------------------------------------------------------------------
    def build_from_reservoir(self, reservoir):
        """Build co-occurrence matrix from reservoir, then SVD."""
        if not self.enabled or not reservoir:
            return

        print("[Upgrade] Building co-occurrence from reservoir...")

        # Collect top N frequent words
        freq = Counter()
        sample = reservoir[:min(len(reservoir), 50000)]
        for text in sample:
            for t in self.vocab._tokenize(text):
                freq[t] += 1

        top_words = [w for w, _ in freq.most_common(self.svd_vocab_size)]
        self.word_to_idx = {w: i for i, w in enumerate(top_words)}
        V = len(top_words)

        # Build co-occurrence matrix
        cooc = np.zeros((V, V), dtype=np.float32)
        for text in sample:
            tokens = self.vocab._tokenize(text)
            for i, w1 in enumerate(tokens):
                if w1 not in self.word_to_idx:
                    continue
                i1 = self.word_to_idx[w1]
                for j in range(max(0, i - self.window), min(len(tokens), i + self.window + 1)):
                    if i == j:
                        continue
                    w2 = tokens[j]
                    if w2 not in self.word_to_idx:
                        continue
                    i2 = self.word_to_idx[w2]
                    weight = 1.0 / (abs(i - j) + 1)
                    cooc[i1, i2] += weight

        # Normalize rows
        row_sums = cooc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cooc = cooc / row_sums

        # SVD
        print("[Upgrade] Computing SVD embeddings...")
        try:
            U, S, Vt = np.linalg.svd(cooc, full_matrices=False)
            self.embeddings = U[:, :self.svd_dim] * S[:self.svd_dim]
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms
        except np.linalg.LinAlgError:
            print("[Upgrade] SVD failed, using random projection fallback")
            self.embeddings = np.random.randn(V, self.svd_dim).astype(np.float32)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms

        print(f"[Upgrade] SVD embeddings: {self.embeddings.shape}")

        # Quantize for memory efficiency
        if self.quantize_enabled and self.embeddings is not None:
            self._quantize_embeddings()

        # Compute IDF
        if self.idf_enabled:
            self._compute_idf(reservoir)

    def _quantize_embeddings(self):
        """Quantize float32 embeddings to uint8 for 4x memory savings."""
        print(f"[Upgrade] Quantizing embeddings to {self.quantize_bits}-bit...")
        emb = self.embeddings.astype(np.float32)

        if self.quantize_bits == 8:
            # Per-row min-max quantization to uint8
            row_min = emb.min(axis=1, keepdims=True)
            row_max = emb.max(axis=1, keepdims=True)
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0
            self.quant_scale = np.concatenate([row_min, row_range], axis=1)  # (V, 2)
            self.quantized_embeddings = ((emb - row_min) / row_range * 255).astype(np.uint8)
        elif self.quantize_bits == 4:
            # 4-bit: pack two values per byte
            row_min = emb.min(axis=1, keepdims=True)
            row_max = emb.max(axis=1, keepdims=True)
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0
            self.quant_scale = np.concatenate([row_min, row_range], axis=1)
            quantized = ((emb - row_min) / row_range * 15).astype(np.uint8)
            # Pack pairs
            even = quantized[:, 0::2]
            odd = quantized[:, 1::2]
            if even.shape[1] > odd.shape[1]:
                odd = np.pad(odd, ((0, 0), (0, 1)), constant_values=0)
            self.quantized_embeddings = (even << 4) | odd
        else:
            self.quantized_embeddings = None
            return

        orig_bytes = emb.nbytes
        new_bytes = self.quantized_embeddings.nbytes
        print(f"[Upgrade] Quantized: {orig_bytes / 1024 / 1024:.1f}MB -> {new_bytes / 1024 / 1024:.1f}MB ({orig_bytes / max(new_bytes, 1):.1f}x)")

    def _dequantize_row(self, idx):
        """Dequantize a single row for search."""
        if self.quantized_embeddings is None:
            return self.embeddings[idx] if self.embeddings is not None else None
        if self.quant_scale is None:
            return None
        row_min, row_range = self.quant_scale[idx]
        if self.quantize_bits == 8:
            return row_min + (self.quantized_embeddings[idx].astype(np.float32) / 255.0) * row_range
        elif self.quantize_bits == 4:
            packed = self.quantized_embeddings[idx]
            high = (packed >> 4).astype(np.float32)
            low = (packed & 0x0F).astype(np.float32)
            dequant = np.zeros(self.svd_dim, dtype=np.float32)
            dequant[0::2] = high[:len(high)]
            dequant[1::2] = low[:len(low)]
            actual_len = min(len(dequant), self.embeddings.shape[1] if self.embeddings is not None else self.svd_dim)
            return row_min + (dequant[:actual_len] / 15.0) * row_range
        return None

    # ------------------------------------------------------------------
    # 2. IDF Weighting
    # ------------------------------------------------------------------
    def _compute_idf(self, reservoir):
        """Compute Inverse Document Frequency."""
        print("[Upgrade] Computing IDF weights...")
        doc_freq = Counter()
        for text in reservoir:
            seen = set()
            for t in self.vocab._tokenize(text):
                if t in self.word_to_idx:
                    seen.add(t)
            for w in seen:
                doc_freq[w] += 1

        total_docs = len(reservoir)
        self.idf_weights = {}
        for w, df in doc_freq.items():
            self.idf_weights[w] = math.log(total_docs / (df + 1)) + 1

    # ------------------------------------------------------------------
    # 3. Confidence Scoring
    # ------------------------------------------------------------------
    def compute_confidence(self, patterns):
        """Compute Z-score confidence for each n-gram pattern."""
        if not self.confidence_enabled or not patterns:
            return

        print("[Upgrade] Computing pattern confidence scores...")
        freqs = list(patterns.values())
        if not freqs:
            return

        mean_freq = np.mean(freqs)
        std_freq = np.std(freqs) if len(freqs) > 1 else 1.0
        if std_freq == 0:
            std_freq = 1.0

        for gram, freq in patterns.items():
            z = (freq - mean_freq) / std_freq
            confidence = max(0.0, min(1.0, (z + 2) / 4))
            self.confidence_scores[gram] = confidence

    # ------------------------------------------------------------------
    # 4. Multi-Head Reservoir (Topic Clustering)
    # ------------------------------------------------------------------
    def cluster_reservoir(self, reservoir):
        """Cluster reservoir into topic heads using K-means on embeddings."""
        if not self.multi_head_enabled or self.embeddings is None or not reservoir:
            return

        print("[Upgrade] Clustering reservoir into topic heads...")

        def get_text_emb(text):
            vec = np.zeros(self.svd_dim, dtype=np.float32)
            count = 0
            for t in self.vocab._tokenize(text):
                if t in self.word_to_idx:
                    idx = self.word_to_idx[t]
                    if idx < self.embeddings.shape[0]:
                        vec += self.embeddings[idx]
                        count += 1
            return vec / count if count > 0 else vec

        # Sample for speed
        sample_size = min(len(reservoir), 10000)
        indices = np.random.choice(len(reservoir), sample_size, replace=False) if len(reservoir) > sample_size else np.arange(len(reservoir))

        embs = []
        valid_texts = []
        valid_idx = []
        for i in indices:
            text = reservoir[i]
            emb = get_text_emb(text)
            if np.linalg.norm(emb) > 1e-6:
                embs.append(emb)
                valid_texts.append(text)
                valid_idx.append(i)

        if not embs:
            return

        embs = np.array(embs, dtype=np.float32)
        K = min(self.num_heads, len(embs))

        # K-means (10 iterations)
        centroids = embs[:K].copy()
        labels = np.zeros(len(embs), dtype=int)
        for _ in range(15):
            dists = np.linalg.norm(embs[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                embs[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(K)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids

        self.reservoir_heads = [[] for _ in range(K)]
        self.head_assignments = [0] * len(reservoir)
        for i, text in enumerate(valid_texts):
            self.reservoir_heads[labels[i]].append(text)
            self.head_assignments[valid_idx[i]] = labels[i]

        print(f"[Upgrade] Reservoir heads: {[len(h) for h in self.reservoir_heads]}")

    # ------------------------------------------------------------------
    # 5. Semantic Search (at inference time)
    # ------------------------------------------------------------------
    def get_text_embedding(self, text):
        """Get embedding for a text (average of word embeddings + IDF)."""
        if self.embeddings is None:
            return None

        vec = np.zeros(self.svd_dim, dtype=np.float32)
        count = 0
        for t in self.vocab._tokenize(text):
            if t in self.word_to_idx:
                idx = self.word_to_idx[t]
                if idx < self.embeddings.shape[0]:
                    weight = self.idf_weights.get(t, 1.0)
                    vec += self.embeddings[idx] * weight
                    count += 1

        if count == 0:
            return None
        vec /= count
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def semantic_search(self, query, candidates, top_k=5):
        """Search candidates using cosine similarity on SVD embeddings."""
        if self.embeddings is None or not candidates:
            return []

        q_emb = self.get_text_embedding(query)
        if q_emb is None:
            return []

        scores = []
        for i, text in enumerate(candidates):
            t_emb = self.get_text_embedding(text)
            if t_emb is not None:
                score = float(np.dot(q_emb, t_emb))
                scores.append((score, i, text))

        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, dir_path):
        """Save all upgrade artifacts to directory."""
        if not self.enabled:
            return

        os.makedirs(dir_path, exist_ok=True)

        # Embeddings (quantized or float32)
        if self.quantized_embeddings is not None:
            np.save(os.path.join(dir_path, "embeddings_q.npy"), self.quantized_embeddings)
            np.save(os.path.join(dir_path, "quant_scale.npy"), self.quant_scale)
        elif self.embeddings is not None:
            np.save(os.path.join(dir_path, "embeddings.npy"), self.embeddings)

        # Word index
        if self.word_to_idx:
            with open(os.path.join(dir_path, "word_to_idx.json"), 'w', encoding='utf-8') as f:
                json.dump(self.word_to_idx, f)

        # IDF
        if self.idf_weights:
            with open(os.path.join(dir_path, "idf_weights.json"), 'w', encoding='utf-8') as f:
                json.dump(self.idf_weights, f)

        # Confidence
        if self.confidence_scores:
            conf_str = {str(k): v for k, v in self.confidence_scores.items()}
            with open(os.path.join(dir_path, "confidence.json"), 'w', encoding='utf-8') as f:
                json.dump(conf_str, f)

        # Reservoir heads
        if self.reservoir_heads:
            for i, head in enumerate(self.reservoir_heads):
                head_path = os.path.join(dir_path, f"head_{i}.pkl")
                with open(head_path, 'wb') as f:
                    pickle.dump(head, f)
            meta = {"num_heads": len(self.reservoir_heads),
                    "total": sum(len(h) for h in self.reservoir_heads)}
            with open(os.path.join(dir_path, "head_meta.json"), 'w') as f:
                json.dump(meta, f)

        # Config snapshot
        snapshot = {
            'svd_dim': self.svd_dim,
            'svd_vocab_size': self.svd_vocab_size,
            'quantize_bits': self.quantize_bits,
            'num_heads': len(self.reservoir_heads),
            'idf_count': len(self.idf_weights),
            'confidence_count': len(self.confidence_scores),
        }
        with open(os.path.join(dir_path, "upgrade_meta.json"), 'w') as f:
            json.dump(snapshot, f, indent=2)

    @classmethod
    def load(cls, dir_path, vocab=None, config=None):
        """Load upgrade artifacts from directory."""
        upgrader = cls(vocab, config)

        meta_path = os.path.join(dir_path, "upgrade_meta.json")
        if not os.path.exists(meta_path):
            upgrader.enabled = False
            return upgrader

        with open(meta_path) as f:
            meta = json.load(f)

        upgrader.enabled = True
        upgrader.svd_dim = meta.get('svd_dim', 128)

        # Load embeddings
        q_path = os.path.join(dir_path, "embeddings_q.npy")
        f_path = os.path.join(dir_path, "embeddings.npy")
        if os.path.exists(q_path):
            upgrader.quantized_embeddings = np.load(q_path)
            scale_path = os.path.join(dir_path, "quant_scale.npy")
            if os.path.exists(scale_path):
                upgrader.quant_scale = np.load(scale_path)
            upgrader.quantize_bits = meta.get('quantize_bits', 8)
        elif os.path.exists(f_path):
            upgrader.embeddings = np.load(f_path)

        # Word index
        idx_path = os.path.join(dir_path, "word_to_idx.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                upgrader.word_to_idx = json.load(f)

        # IDF
        idf_path = os.path.join(dir_path, "idf_weights.json")
        if os.path.exists(idf_path):
            with open(idf_path) as f:
                upgrader.idf_weights = json.load(f)

        # Confidence
        conf_path = os.path.join(dir_path, "confidence.json")
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                raw = json.load(f)
            upgrader.confidence_scores = raw

        # Reservoir heads
        head_meta_path = os.path.join(dir_path, "head_meta.json")
        if os.path.exists(head_meta_path):
            with open(head_meta_path) as f:
                hmeta = json.load(f)
            num_heads = hmeta.get('num_heads', 0)
            upgrader.reservoir_heads = []
            for i in range(num_heads):
                hp = os.path.join(dir_path, f"head_{i}.pkl")
                if os.path.exists(hp):
                    with open(hp, 'rb') as f:
                        upgrader.reservoir_heads.append(pickle.load(f))

        return upgrader

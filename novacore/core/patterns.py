"""Core: Pattern extraction - learn co-occurrence patterns from training-free text."""
import re
from collections import Counter
import numpy as np


# Precompiled regex (compiled once, reused for every text)
_PAT_TOKEN_RE = re.compile(r'\b\w+\b|[.,!?;:"\']')


class PatternExtractor:
    """
    Extracts n-gram patterns and co-occurrence stats from text.
    These patterns become the 'memory' - no training required.
    """

    def __init__(self, ngram_range=None, min_freq=None, max_size=None):
        from ..config import get_default
        self.ngram_range = tuple(ngram_range or get_default('ngram_range'))
        self.min_freq = min_freq if min_freq is not None else get_default('pattern_min_freq')
        self.max_size = max_size if max_size is not None else get_default('pattern_max_size')
        self.patterns = Counter()
        self.cooccurrence = {}
        self._reset()

    def _reset(self):
        self.patterns = Counter()
        self.cooccurrence = {}

    def _tokenize(self, text):
        """Tokenize text into words."""
        return _PAT_TOKEN_RE.findall(text.lower())

    def update(self, text):
        """Add text, extract patterns."""
        from ..config import get_default
        if not text:
            return
        tokens = _PAT_TOKEN_RE.findall(text.lower())
        if not tokens:
            return
        pat = self.patterns
        lt = len(tokens)

        # Extract n-grams (tuple keys - predictor relies on tuple semantics)
        nlo, nhi = self.ngram_range
        pat_cap = self.max_size * 2  # soft cap to prevent unbounded growth
        for n in range(nlo, nhi + 1):
            end = lt - n + 1
            if end <= 0:
                continue
            if len(pat) >= pat_cap:
                break
            for i in range(end):
                pat[tuple(tokens[i:i + n])] += 1

        # Co-occurrence within window — sliding window (near-linear).
        # CAP at 500K entries to prevent MemoryError on large corpora.
        CO_CAP = 500_000
        window = get_default('cooccurrence_window')
        co = self.cooccurrence
        for i in range(lt):
            if len(co) >= CO_CAP:
                break
            ti = tokens[i]
            hi = i + window + 1
            if hi > lt:
                hi = lt
            for j in range(i + 1, hi):
                g = (ti, tokens[j])
                co[g] = co.get(g, 0) + 1

    def update_many(self, texts, log=None, label="patterns", total=None):
        """Add many texts. Optional log for progress so a long extraction
        never looks like it hung.  `total` is an optional hint for progress;
        when omitted, an increment counter is used instead of a fraction.
        Accepts lists, generators, and TextStream objects safely (no
        pre-count that would consume a generator)."""
        if log is None:
            for t in texts:
                self.update(t)
            return
        from time import time
        start = time()
        done = 0
        for t in texts:
            self.update(t)
            done += 1
            if done % 5000 == 0:
                el = time() - start
                per = el / max(1, done)
                if total:
                    eta = per * (total - done)
                    log.info(
                        f"{label}: n-gram extraction {done}/{total} docs "
                        f"({el:.0f}s, ~{eta:.0f}s left)"
                    )
                else:
                    log.info(
                        f"{label}: n-gram extraction {done} docs "
                        f"({el:.0f}s)"
                    )

    def prune(self):
        """Remove rare patterns (below min_freq) and cap size."""
        self.patterns = Counter({
            k: v for k, v in self.patterns.items()
            if v >= self.min_freq
        })
        if len(self.patterns) > self.max_size:
            self.patterns = Counter(
                dict(sorted(self.patterns.items(), key=lambda x: -x[1])[:self.max_size])
            )

    def top_patterns(self, k=None):
        from ..config import get_default
        k = k if k is not None else get_default('pattern_top_k')
        return self.patterns.most_common(k)

    def encode(self, dim=None):
        """Encode all patterns into a fixed-size weight vector."""
        from ..config import get_default
        if dim is None:
            dim = get_default('dim')
        vec = np.zeros(dim, dtype=np.float64)
        for gram, count in self.patterns.items():
            h = abs(hash(gram)) % dim
            vec[h] += count
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def save(self):
        """Serialize patterns for storage."""
        return {
            "patterns": list(self.patterns.items()),
            "min_freq": self.min_freq,
            "ngram_range": list(self.ngram_range),
            "max_size": self.max_size,
        }

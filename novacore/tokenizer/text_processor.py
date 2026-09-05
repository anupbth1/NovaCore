"""Tokenizer: TextProcessor - converts text to feature vectors (no training)."""
import numpy as np
from .vocab import Vocabulary
from ..core.hasher import murmurhash3


class TextProcessor:
    """
    Converts text to fixed-size weight vectors via feature hashing.
    This is the bridge between tokens and the encoding engine.
    No training needed - purely deterministic transform.
    """

    def __init__(self, vocab=None, dim=None, use_vocab=True):
        from ..config import get_default
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.dim = dim if dim is not None else get_default('dim')
        self.use_vocab = use_vocab

    def text_to_vector(self, text, normalize=True):
        """Convert text to a dense feature vector using hashing."""
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = self.vocab._tokenize(text) if self.use_vocab else self._simple_tokenize(text)
        for tok in tokens:
            h = murmurhash3(tok)
            # Signed hashing for unbiased
            bucket = h % self.dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[bucket] += sign
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        return vec

    def _simple_tokenize(self, text):
        return text.lower().split()

    def id_sequence_to_matrix(self, ids, seq_len=None):
        """
        Convert a sequence of token IDs to a matrix (for prediction).
        Each row: one-hot style hashed embedding of that token.
        """
        if seq_len is None:
            seq_len = len(ids)
        matrix = np.zeros((seq_len, self.dim), dtype=np.float64)
        for t, tid in enumerate(ids[-seq_len:]):
            h = murmurhash3(str(tid) if tid not in (0,1,2,3) else f"<{tid}>")
            bucket = h % self.dim
            matrix[t, bucket] = 1.0
        return matrix

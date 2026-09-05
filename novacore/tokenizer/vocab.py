"""Tokenizer: Vocabulary builder from text."""
import json
import re
from collections import Counter


# Precompiled tokenization regex (much faster than re.findall per call)
_TOKEN_RE = re.compile(r'\b\w+\b|[.,!?;:"\'()]')


class Vocabulary:
    """Builds and manages a token vocabulary."""

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, vocab_size=None, min_freq=None):
        from ..config import get_default
        self.vocab_size = vocab_size if vocab_size is not None else get_default('vocab_size')
        self.min_freq = min_freq if min_freq is not None else get_default('token_min_freq')
        self.token_to_id = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.freqs = Counter()

    def _tokenize(self, text):
        """Convert text to tokens (words + punctuation)."""
        return _TOKEN_RE.findall(text.lower())

    def build(self, texts):
        """Build vocabulary from list of texts."""
        # Fast single-pass tokenization with Counter.update
        freqs = self.freqs
        for text in texts:
            if not text:
                continue
            freqs.update(_TOKEN_RE.findall(text.lower()))

        # Add frequent tokens (excluding reserved)
        token_to_id = self.token_to_id
        id_to_token = self.id_to_token
        vocab_size = self.vocab_size
        min_freq = self.min_freq
        for tok, freq in freqs.most_common():
            if freq < min_freq:
                continue
            if tok in token_to_id:
                continue
            if len(token_to_id) >= vocab_size:
                break
            tid = len(token_to_id)
            token_to_id[tok] = tid
            id_to_token[tid] = tok
        return len(self.token_to_id)

    def encode(self, text):
        """Convert text to list of token IDs."""
        ids = [self.BOS]
        token_to_id = self.token_to_id
        for tok in _TOKEN_RE.findall(text.lower()):
            ids.append(token_to_id.get(tok, self.UNK))
        ids.append(self.EOS)
        return ids

    def grow(self, texts):
        """Expand an existing vocabulary with new texts (no reset)."""
        token_to_id = self.token_to_id
        id_to_token = self.id_to_token
        vocab_size = self.vocab_size
        added = 0
        for text in texts:
            if not text:
                continue
            for tok in _TOKEN_RE.findall(text.lower()):
                self.freqs[tok] += 1
                if tok not in token_to_id:
                    if len(token_to_id) >= vocab_size:
                        continue
                    tid = len(token_to_id)
                    token_to_id[tok] = tid
                    id_to_token[tid] = tok
                    added += 1
        return added

    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            if i == self.UNK:
                tokens.append("<unk>")
            else:
                tokens.append(self.id_to_token[i])
        return " ".join(tokens)

    def save(self, path):
        """Save vocabulary to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "token_to_id": self.token_to_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load vocabulary from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        v = cls(data['vocab_size'], data['min_freq'])
        v.token_to_id = data['token_to_id']
        v.id_to_token = {v: k for k, v in v.token_to_id.items()}
        return v

    def __len__(self):
        return len(self.token_to_id)

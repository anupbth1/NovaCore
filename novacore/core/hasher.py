"""Core: Feature hashing - map high-dimensional sparse features to compact weights."""
import numpy as np


def murmurhash3(text, seed=0):
    """Simple deterministic hash function (MurmurHash3-style)."""
    if isinstance(text, str):
        data = text.encode('utf-8')
    else:
        data = bytes(text)
    m = 0x5bd1e995
    r = 24
    h = seed ^ len(data)
    for i in range(0, len(data) - 3, 4):
        k = int.from_bytes(data[i:i+4], 'little')
        k = (k * m) & 0xFFFFFFFF
        k ^= k >> r
        k = (k * m) & 0xFFFFFFFF
        h = (h * m) & 0xFFFFFFFF
        h ^= k
    return h & 0xFFFFFFFF


class FeatureHasher:
    """
    Count-Min-Sketch style feature hashing.
    Maps arbitrary features to a fixed-size weight matrix.
    Supports signed hashing for unbiased estimators.
    """

    def __init__(self, num_buckets=None, depth=None):
        from ..config import get_default
        if num_buckets is None:
            num_buckets = get_default('dim') * get_default('hasher_buckets_multiplier')
        if depth is None:
            depth = get_default('hasher_depth')
        self.num_buckets = num_buckets
        self.depth = depth
        self.counts = np.zeros((depth, num_buckets), dtype=np.float64)
        self.seeds = [i * get_default('hasher_seed_stride') + 1 for i in range(depth)]

    def _hash(self, feature, seed):
        """Hash feature to (bucket, sign)."""
        h = murmurhash3(feature, seed)
        bucket = h % self.num_buckets
        sign = 1 if (h >> 8) % 2 == 0 else -1
        return bucket, sign

    def add(self, feature, value=1.0):
        """Add feature with value contribution."""
        for d, seed in enumerate(self.seeds):
            bucket, sign = self._hash(feature, seed)
            self.counts[d, bucket] += sign * value

    def add_list(self, features):
        """Add a list of features (each weight 1)."""
        for f in features:
            self.add(f, 1.0)

    def add_weighted(self, features, weights):
        """Add features with per-feature weights."""
        for f, w in zip(features, weights):
            self.add(f, w)

    def estimate(self, feature):
        """Estimate frequency of a feature (min of depth counters)."""
        vals = []
        for d, seed in enumerate(self.seeds):
            bucket, sign = self._hash(feature, seed)
            vals.append(sign * self.counts[d, bucket])
        return min(vals)

    def vector(self):
        """Return flattened weight vector."""
        return self.counts.flatten()

    def reset(self):
        self.counts.fill(0.0)

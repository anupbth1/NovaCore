"""Core: Reservoir sampling - pick representative samples from large datasets."""
import numpy as np


class ReservoirSampler:
    """
    Memory-efficient sampling from large datasets.
    Guarantees uniform random sample without loading all data.
    """

    def __init__(self, k=None, seed=None):
        from ..config import get_default
        if k is None:
            k = get_default('reservoir_sample_size')
        if seed is None:
            seed = get_default('reservoir_seed')
        if seed is None:
            seed = 42
        self.k = k
        self.reservoir = []
        self.seen = 0
        self._rng = np.random.RandomState(seed)

    def add(self, item):
        """Process one item (streaming). Memory-safe: large strings
        are truncated to avoid unbounded growth in the reservoir buffer.
        """
        self.seen += 1
        # Truncate massive items to keep memory bounded (~8KB per item)
        if isinstance(item, str) and len(item) > 8192:
            item = item[:8192]
        elif isinstance(item, dict):
            # If a dict sneaks in (e.g. metadata wrapper), keep only 'text'
            if 'text' in item and isinstance(item['text'], str):
                t = item['text']
                if len(t) > 8192:
                    t = t[:8192]
                item = t
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            j = self._rng.randint(0, self.seen)
            if j < self.k:
                self.reservoir[j] = item

    def add_many(self, items, log=None, label="sampling", total=None):
        """Process many items. `total` is optional hint for progress; when
        omitted, progress lines use an increment counter instead of a
        fraction.  Accepts lists, generators, and TextStream objects safely
        (no pre-count that would consume a generator)."""
        if log is None:
            for item in items:
                self.add(item)
            return
        from time import time
        start = time()
        done = 0
        for item in items:
            self.add(item)
            done += 1
            if done % 200000 == 0:
                el = time() - start
                per = el / max(1, done)
                if total:
                    eta = per * (total - done)
                    log.info(
                        f"{label}: reservoir {done}/{total} docs "
                        f"({el:.0f}s, ~{eta:.0f}s left)"
                    )
                else:
                    log.info(
                        f"{label}: reservoir {done} docs "
                        f"({el:.0f}s)"
                    )

    def sample(self):
        """Return the reservoir sample."""
        return self.reservoir

    @property
    def count(self):
        return self.seen

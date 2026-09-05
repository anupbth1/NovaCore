"""Dataset: HF (Hugging Face) loader - login, load, download, stream."""
import gc
import os
import json
import re
import glob
import hashlib
import shutil
import random
import itertools
import tempfile
import time

import numpy as np

from ..logger import Logger, _c, _Color


# ----------------------------------------------------------------------
# Schema detection constants (semantic roles + column aliases)
# ----------------------------------------------------------------------
_ROLE_HIGH = {
    "instruction": {"instruction", "instructions", "problem", "prompt", "query", "task"},
    "input": {"input", "context", "background"},
    "question": {"question"},
    "reasoning": {"reasoning", "chain_of_thought", "cot", "thought", "thinking",
                  "analysis", "rationale", "explanation", "plan", "planning",
                  "steps", "step", "solution_steps"},
    "answer": {"answer", "output", "response", "completion", "final_answer",
               "solution", "result"},
    "messages": {"messages", "conversation", "chat"},
    "text": {"text", "content", "document", "body", "story"},
}

_ROLE_TOKENS = {
    "reasoning": {"reasoning", "cot", "thought", "thinking", "analysis",
                  "rationale", "explanation", "plan", "planning", "step",
                  "steps", "chain", "strategy"},
    "answer": {"answer", "output", "response", "completion", "result",
               "solution", "final"},
    "instruction": {"instruction", "problem", "prompt", "query", "task"},
    "input": {"input", "context", "background"},
    "text": {"text", "content", "document", "body", "story"},
}

# Serialization order (lowest first)
_ROLE_PRIORITY = {
    "instruction": 1, "question": 2, "input": 3,
    "reasoning": 4, "answer": 5, "text": 90, "messages": 95,
}

_CHAT_COLS = {"messages", "conversation", "chat", "message"}

# Columns that can serve as a stable per-row identity for dedupe
_UID_COLS = {"uid", "id", "_id", "guid", "unique_id", "key", "hash", "sample_id"}

# Reservoir scan bound for random selection (we never scan the whole repo)
_RANDOM_SCAN_FACTOR = 20

# Maximum number of UIDs to keep in a RAM set before switching to a lighter
# dedup strategy.  Beyond this threshold, duplicate suppression is done via
# a compact on-disk Bloom-style hash set (each entry is a 4-byte hash32,
# ~12 MB for 3 M rows) instead of a full Python set (~300+ MB).
_UID_RAM_THRESHOLD = 500_000


# ======================================================================
# TextStream – memory-efficient file-backed text iterable
# ======================================================================
class TextStream:
    """Read texts lazily from a pool.jsonl file without loading them all
    into RAM.  Supports ``len()`` (from manifest row count) and full
    iteration.  Slicing (``stream[:n]``) materialises the requested slice
    as a plain list so callers that only slice small prefixes (vocab cap,
    pattern cap) stay fast.
    """

    def __init__(self, rows_path, count=None):
        self._path = rows_path
        # count: known row count (from manifest.json); avoids line-counting.
        self._count = count if count is not None else self._count_lines()

    # -- helpers --------------------------------------------------------
    def _count_lines(self):
        """Fall-back: count lines when manifest count is unavailable."""
        n = 0
        try:
            with open(self._path, encoding="utf-8") as fh:
                for _ in fh:
                    n += 1
        except OSError:
            pass
        self._count = n
        return n

    # -- protocol -------------------------------------------------------
    def __len__(self):
        return self._count

    def __iter__(self):
        """Yield text strings from pool.jsonl one at a time.

        Uses an 8 MB read buffer (vs default 8 KB) for ~3-5x faster I/O on
        Windows / NTFS, where small read syscalls are slow.
        """
        BUF = 8 * 1024 * 1024  # 8 MB buffer
        try:
            with open(self._path, "r", encoding="utf-8", buffering=BUF) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        item = json.loads(ln)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    text = item.get("text") if isinstance(item, dict) else None
                    if text and isinstance(text, str):
                        yield text
        except OSError:
            return

    def __getitem__(self, index):
        """Slice support: ``stream[:100_000]`` returns a plain list."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step != 1:
                raise NotImplementedError("TextStream does not support step slicing")
            result = []
            for i, text in enumerate(self):
                if i >= stop:
                    break
                if i >= start:
                    result.append(text)
            return result
        elif isinstance(index, int):
            if index < 0:
                return list(self)[index]
            for i, text in enumerate(self):
                if i == index:
                    return text
            raise IndexError("TextStream index out of range")
        raise TypeError(f"indices must be integers or slices, not {type(index).__name__}")

    def __bool__(self):
        return self._count > 0

    def __eq__(self, other):
        """Two TextStreams pointing to the same file with the same count are equal."""
        if isinstance(other, TextStream):
            return self._path == other._path and self._count == other._count
        return NotImplemented

    def __hash__(self):
        return hash((self._path, self._count))

    def __repr__(self):
        return f"TextStream({self._path!r}, count={self._count})"


# ======================================================================
# RowStream – file-backed row iterable (preserves original columns)
# ======================================================================
class RowStream:
    """Like :class:`TextStream` but yields the original ``row`` dict from
    the pool (so callers see all native columns, not just ``text``).  When
    a pool row has no ``row`` field (older pools), falls back to a
    ``{"text": ...}`` dict for backward compatibility.
    """

    def __init__(self, rows_path, count=None):
        self._path = rows_path
        self._count = count if count is not None else self._count_lines()

    def _count_lines(self):
        n = 0
        try:
            with open(self._path, encoding="utf-8") as fh:
                for _ in fh:
                    n += 1
        except OSError:
            pass
        self._count = n
        return n

    def __len__(self):
        return self._count

    def __iter__(self):
        BUF = 8 * 1024 * 1024  # 8 MB buffer for faster I/O
        try:
            with open(self._path, "r", encoding="utf-8", buffering=BUF) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        item = json.loads(ln)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(item, dict):
                        continue
                    # New flat format: {uid, text, source, domain, ...}
                    # All columns stored flat (no nested "row" object).
                    text = item.get("text", "")
                    if isinstance(text, str) and text:
                        # Yield full item as the row dict (all original columns)
                        yield item
        except OSError:
            return

    def __bool__(self):
        return self._count > 0

    def __eq__(self, other):
        if isinstance(other, RowStream):
            return self._path == other._path and self._count == other._count
        return NotImplemented

    def __hash__(self):
        return hash((self._path, self._count))

    def __repr__(self):
        return f"RowStream({self._path!r}, count={self._count})"


# ======================================================================
# CompactHashSet – memory-light dedup for very large UID pools
# ======================================================================
class _CompactHashSet:
    """A 32-bit-hash-based dedup set that uses ~4 bytes per entry instead
    of the ~100+ bytes of a Python ``set`` of strings.  Accepts a small
    false-positive rate (< 1/2^32 per probe with linear probing over a
    2× table).
    """

    def __init__(self, capacity=1_000_000):
        self._cap = max(256, capacity)
        self._table = np.zeros(self._cap, dtype=np.uint32)
        self._count = 0

    @staticmethod
    def _hash32(val):
        """Fast 32-bit hash from a string (~10x faster than md5).
        Uses FNV-1a + bit mixing; collision rate <1/2^32 for our use case.
        """
        if isinstance(val, str):
            val = val.encode("utf-8", "replace")
        h = 2166136261  # FNV offset basis
        for b in val:
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        # bit-mix to spread bits (Murmur finalizer)
        h ^= h >> 16
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        return h | 1  # ensure non-zero (0 = empty slot)

    def add(self, uid):
        h = self._hash32(uid)
        idx = h % self._cap
        # linear probing
        for _ in range(min(16, self._cap)):
            if self._table[idx] == 0:
                self._table[idx] = h
                self._count += 1
                return True  # new
            if self._table[idx] == h:
                return False  # duplicate
            idx = (idx + 1) % self._cap
        # table nearly full or long probe – still add (false positive possible)
        self._table[idx] = h
        self._count += 1
        return True

    def __contains__(self, uid):
        h = self._hash32(uid)
        idx = h % self._cap
        for _ in range(min(16, self._cap)):
            v = self._table[idx]
            if v == 0:
                return False
            if v == h:
                return True
            idx = (idx + 1) % self._cap
        return False

    def __len__(self):
        return self._count

    def update(self, uids):
        for u in uids:
            self.add(u)


def _normalize_col(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _camel_to_snake(name):
    """Mirror huggingface `datasets.naming.camelcase_to_snakecase`."""
    s = str(name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _norm(name):
    return re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_")


def _fmt_bytes(n):
    """Format a byte count as a human-readable string."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024


def _role_for_col(name):
    """Guess a semantic role for a column name.

    Returns (role, confidence) with confidence in {'HIGH', 'MEDIUM'}
    or (None, None) if no match -> caller treats it as unknown/metadata.
    """
    n = _normalize_col(name)
    if not n:
        return None, None
    toks = set(t for t in n.split("_") if t)
    if not toks:
        return None, None

    # Exact match against well-known role names -> HIGH confidence
    for role, prims in _ROLE_HIGH.items():
        if n in prims:
            return role, "HIGH"

    if toks & _CHAT_COLS:
        return "messages", "HIGH"

    rsn = toks & _ROLE_TOKENS["reasoning"]
    ans = toks & _ROLE_TOKENS["answer"]
    if rsn:
        # e.g. 'solution_steps' -> reasoning (steps), but plain 'solution' -> answer
        if ans and not (rsn & {"step", "steps", "plan", "planning", "reasoning"}):
            return "answer", "MEDIUM"
        return "reasoning", "MEDIUM"
    if ans:
        return "answer", "MEDIUM"
    if toks & _ROLE_TOKENS["instruction"]:
        return "instruction", "MEDIUM"
    if "question" in toks:
        return "question", "MEDIUM"
    if toks & _ROLE_TOKENS["input"]:
        return "input", "MEDIUM"
    if toks & _ROLE_TOKENS["text"]:
        return "text", "MEDIUM"
    return None, None


class HFLoader:
    """
    Hugging Face datasets integration for NovaCore.
    Supports: login, download, load (memoized), and streaming iteration.
    """

    def __init__(self, cache_dir=None):
        # Lazy import so package works even without HF installed
        self._datasets = None
        self._hub = None
        self._schema_entries = []
        self._skipped = 0
        self.last_schema = None
        self._setup_env()

    def _setup_env(self):
        """Point ALL HF caches (raw parquet + hub + datasets) into the project
        data folder (config 'hf_cache', default: <project>/data/hf)."""
        try:
            from ..config import get_path
            cfg = get_path('hf_cache')
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache = cfg if cfg and os.path.isabs(cfg) else os.path.join(base, cfg or "data/hf")
            cache = os.path.abspath(cache)
            os.environ["HF_HOME"] = cache
            os.environ["HF_HUB_CACHE"] = os.path.join(cache, "hub")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        except Exception:
            pass

    def _import_optional(self):
        if self._datasets is not None:
            return self._datasets, self._hub
        try:
            import datasets
            import huggingface_hub as hub
        except ImportError as e:
            raise RuntimeError(
                "Hugging Face libraries not installed. Run: pip install datasets huggingface_hub"
            ) from e
        self._datasets = datasets
        self._hub = hub
        return datasets, hub

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------
    def login(self, token=None, write_permission=False):
        """Login to Hugging Face. Uses token if given, else interactive."""
        datasets_obj, hub = self._import_optional()
        if token:
            hub.login(token=token, add_to_git_credential=False)
        else:
            hub.login(add_to_git_credential=False)
        return self.whoami()

    def whoami(self):
        """Return current logged-in user info (None if not logged in)."""
        datasets_obj, hub = self._import_optional()
        try:
            return hub.whoami()
        except Exception:
            return None

    def logout(self):
        """Logout from Hugging Face."""
        datasets_obj, hub = self._import_optional()
        hub.logout()
        return True

    def auth_status(self):
        """Return auth status dict."""
        user = self.whoami()
        return {
            "logged_in": user is not None,
            "user": user,
        }

    # ------------------------------------------------------------------
    # Loading (download + cache)
    # ------------------------------------------------------------------
    def load(self, dataset_name, split=None, cache_dir=None, streaming=False,
             text_column=None, num_rows=None, name=None, fields=None, verbose=True,
             random_sample=False, add_datasets=False, uid_key=None, local_only=False):
        """
        Load a HF dataset by name.

        Args:
            dataset_name: e.g. "squad", "wikitext", "user/dataset"
            split: which split to load ("train", "test", ...) - default from
                   config 'hf.default_split'
            cache_dir: where to cache (default: config hf_cache)
            streaming: use streaming mode (True for huge datasets)
            text_column: column to use as text (if None, auto-detect or config)
            num_rows: limit rows (optional)
            name: config name / subset (passed to load_dataset)
            fields: ordered {column: role} map for schema-aware serialization.
                    Each field is wrapped in <role>...</role> markers so the
                    semantic structure (instruction/input/output/reasoning/
                    messages) is preserved. If None, falls back to config.
            verbose: print the schema detection report (confidence + unknowns)
            random_sample: shuffle the returned pool texts in-place (all rows
                    still retained; never discards a downloaded row).
            add_datasets: True -> pull `num_rows` (or all if None) NEW unique
                    rows into the accumulated pool. False -> reuse whatever is
                    already downloaded; nothing new is fetched (first run still
                    builds the pool once).
            uid_key: column used as the per-row identity for dedupe. Auto-
                    detected ('uid'/'id'/...) when not given; falls back to a
                    content hash.
            local_only: absorb only shards already on disk (zero network);
                    skip any missing files.

        Returns:
            iterable of text strings (list or streaming generator)
        """
        from ..config import get_hf, get_dataset
        datasets_obj, hub = self._import_optional()
        if split is None:
            split = get_hf('default_split', 'train')
        overrides = get_dataset(dataset_name) or {}
        use_fields = fields
        if text_column is None:
            text_column = overrides.get('column')
        if use_fields is None and text_column is None:
            use_fields = overrides.get('fields')
        if name is None:
            name = overrides.get('name', overrides.get('subset'))
        if split == get_hf('default_split', 'train') and overrides.get('split'):
            split = overrides['split']

        # Interrupted-download recovery + cache reuse. The stale-marker sweep is
        # done ONCE per process (cheap guard) so a number of datasets in one run
        # don't re-walk the whole (possibly multi-GB) cache on every step.
        stale = self._clean_incomplete_once()

        log = Logger()
        label = f"{dataset_name}" + (f" [{name}]" if name else "")
        mode_label = _c(_Color.CYAN, "STREAM" if streaming else "DOWNLOAD")
        cache_dir = self._resolve_cache(cache_dir)
        default_split = get_hf('default_split', 'train')

        # Plan the fetch for a bounded (num_rows set, materialized) download.
        # datasets 5.x EAGER hub materialization mis-selects the first data_files
        # glob for card-YAML multi-split configs that lack dataset_infos.json
        # (observed: UltraData-SFT-2605 requests split 'no_think' but downloads
        # the WHOLE 'think' shard set). The streaming iterable honors per-split
        # globs, so for any non-default split we route through it and cap rows
        # ourselves, persisting a small texts-cache so re-runs reuse without
        # re-downloading. Default ('train') splits on classic datasets keep the
        # standard eager path (navigable + positional random sampling).
        use_stream = streaming
        texts_cache = None
        pool = None
        # Track pool info so cmd_train can stream directly from pool files.
        self.last_pool_path = None
        self.last_pool_count = 0
        if not streaming:
            # Accumulated pool for (dataset, config, split). add_datasets=OFF +
            # an existing pool => pure reuse: return what is already downloaded
            # and never touch the network or re-sample (no duplicates ever).
            print(f"[NovaCore] {label}: checking pool (add_datasets={add_datasets}) ...", flush=True)
            pool = self._load_pool(cache_dir, dataset_name, name, split,
                                   uid_key=uid_key, need_uids=add_datasets)
            if pool is None:
                print(f"[NovaCore] {label}: no existing pool found, will build from shards", flush=True)
                pdir = self._pool_dir(cache_dir, dataset_name, name, split)
                pool = {
                    "dir": pdir, "uids": _CompactHashSet(100_000),
                    "meta": {"dataset": dataset_name, "config": name,
                             "split": split, "uid_key": uid_key},
                    "row_count": 0,
                    "rows_path": os.path.join(pdir, "pool.jsonl"),
                    "meta_path": os.path.join(pdir, "manifest.json"),
                }
            if pool.get("row_count", 0) and not add_datasets:
                texts = self._pool_texts(pool)
                log.info(
                    f"{label}: DOWNLOAD -> pool reuse ({pool['row_count']} row(s) in "
                    f"{os.path.basename(pool['dir'])}) - add_datasets OFF, no fetch"
                )
                meta = pool["meta"]
                self.last_schema = {
                    "dataset": dataset_name,
                    "column": meta.get("column") or text_column,
                    "fields": meta.get("fields") or use_fields or {},
                    "source": "pool",
                }
                self.last_pool_path = pool["rows_path"]
                self.last_pool_count = pool["row_count"]
                return texts
        if not streaming and split and split != default_split:
            if num_rows is not None:
                key = f"{dataset_name}__{name or 'default'}__{split}__{num_rows}"
                key = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
                texts_cache = os.path.join(cache_dir, "texts", key + ".jsonl")
                if os.path.isfile(texts_cache):
                    with open(texts_cache, "r", encoding="utf-8") as f:
                        texts = [json.loads(ln) for ln in f if ln.strip()]
                    if random_sample and num_rows is not None and len(texts) > num_rows:
                        texts = random.sample(texts, num_rows)
                    log.info(
                        f"{label}: {mode_label} -> texts-cache hit "
                        f"({len(texts)} text(s)) - reusing, no download"
                    )
                    self.last_schema = {
                        "dataset": dataset_name, "column": text_column,
                        "fields": use_fields or {}, "source": "texts-cache",
                    }
                    return texts
            # DOWNLOAD mode -> real bytes on disk. Resolve this (config, split)'s
            # raw shard files and pull them through the HF hub cache (persistent,
            # resumable - future models reuse the same files without re-download).
            # add_datasets=ON pulls only NEW unique rows (uid dedupe against the
            # pool); OFF with rows already downloaded reuses them instantly.
            # If the split's files can't be resolved (heterogeneous repo), fall
            # back to the per-split streaming iterable.
            files = self._resolve_split_files(dataset_name, split, name)
            if files:
                result = self._fetch_split_to_disk(
                    dataset_name, split, name, files, cache_dir,
                    num_rows=num_rows, random_sample=random_sample,
                    fields=use_fields, texts_cache=texts_cache,
                    verbose=verbose, label=label, log=log,
                    pool=pool, add_datasets=add_datasets, uid_key=uid_key,
                    local_only=local_only,
                )
                if result is not None:
                    return result
                print(f"[NovaCore] {label}: disk-file route failed, falling back to streaming iterable", flush=True)
            use_stream = True

        status = self.cache_status(dataset_name, split=split, name=name)
        if status["status"] == "cached":
            log.info(
                f"{label}: {mode_label} -> data already on disk "
                f"({status['count']} files, {_fmt_bytes(status['bytes'])}) - reading from cache"
            )
        elif status["status"] == "partial":
            log.warn(
                f"{label}: {mode_label} -> partial cache ({len(status['incomplete'])} interrupted marker(s)), resuming"
                + (f", cleaned {stale} stale marker(s)" if stale else "")
            )
        else:
            log.info(f"{label}: {mode_label} -> cache miss, fetching fresh data")

        if not streaming and split and split != default_split and num_rows is not None:
            log.info(
                f"{label}: split '{split}' -> per-split download via iterable "
                f"(datasets 5.x eager-route picks wrong split for multi-split repos)"
            )

        kwargs = {"cache_dir": cache_dir}
        if name:
            kwargs["name"] = name
        # Streaming iterables fetch only the requested split's shards (lazily),
        # so a `num_rows` cap below pulls just the shards covering those rows.
        print(f"[NovaCore] {label}: loading HF dataset (split={split}, streaming={use_stream}) ...", flush=True)
        ds = datasets_obj.load_dataset(
            dataset_name, split=split,
            streaming=use_stream, **kwargs
        )
        print(f"[NovaCore] {label}: HF dataset loaded, detecting schema ...", flush=True)
        col = text_column
        report = None
        if col is None and not use_fields:
            # Schema-aware auto-detect (semantic roles, confidence, unknowns)
            col, use_fields, report = self._detect_schema(ds, dataset_name, verbose=verbose)
        # Unbounded streams stay lazy (caller iterates); bounded loads are
        # materialized so exact counts can be logged.
        is_lazy_stream = use_stream and num_rows is None
        progress_word = "downloaded" if not streaming else "streamed"
        result = self._extract_texts(ds, col, num_rows, fields=use_fields,
                                     random_sample=random_sample, log=log,
                                     progress_word=progress_word)
        if not is_lazy_stream:
            result = list(result) if result is not None else []
        # Accumulate into the pool for download mode (train splits too): each
        # materialized row is uid-deduped (content hash fallback) so repeated
        # runs never duplicate; add_datasets=OFF reuses the pool next time.
        if not streaming and pool is not None and result:
            existing = pool["uids"]
            added = []
            for t in result:
                if not isinstance(t, str) or not t:
                    continue
                uid = self._uid_for(None, uid_key, t)
                if uid in existing:
                    continue
                existing.add(uid)
                added.append({"uid": uid, "text": t})
            if added:
                self._append_pool(pool, added, fields=use_fields or {}, col=col)
        # Persist the small texts-cache for non-default-split bounded downloads
        # so the next run reuses locally instead of re-reading raw shards.
        # Stored as one JSON-encoded string per line (handles embedded
        # newlines / LaTeX without ambiguity).
        if texts_cache and result and not random_sample:
            try:
                os.makedirs(os.path.dirname(texts_cache), exist_ok=True)
                with open(texts_cache, "w", encoding="utf-8") as f:
                    for t in result:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
            except OSError:
                pass
        # Once a pool exists, training always uses the WHOLE accumulated pool:
        # add_datasets=ON keeps growing it, OFF reuses it - never duplicates.
        if not streaming and pool is not None and pool.get("row_count", 0):
            result = self._pool_texts(pool)
            self.last_pool_path = pool["rows_path"]
            self.last_pool_count = pool["row_count"]
        self.last_schema = {
            "dataset": dataset_name,
            "column": col,
            "fields": use_fields or {},
            "report": report,
        }
        if is_lazy_stream:
            return result
        n = len(result) if hasattr(result, '__len__') else None
        if self._skipped:
            log.warn(f"{label}: skipped {self._skipped} empty/invalid rows")
        if n:
            log.ok(f"{label}: {n} row(s) ready for training  ({log.total()})")
        else:
            log.skip(f"{label}: 0 usable rows (check dataset/split/columns)")
        # Free HF dataset object from memory — pool is on disk now.
        del ds
        gc.collect()
        return result

    def stream(self, dataset_name, split=None, cache_dir=None,
               text_column=None, num_rows=None, name=None, fields=None, verbose=True,
               random_sample=False):
        """
        Stream a dataset lazily (great for very large datasets).
        Returns a generator of text strings.
        """
        return self.load(
            dataset_name, split=split, cache_dir=cache_dir,
            streaming=True, text_column=text_column, num_rows=num_rows,
            name=name, fields=fields, verbose=verbose, random_sample=random_sample
        )

    def download(self, dataset_name, save_dir=None, split="train", num_rows=None):
        """
        Download a dataset and save it to a local text file (one doc per line).
        Returns the path to the saved text file - ready for `encode`.
        Save dir defaults to config path 'hf_data' (e.g. data/hf).
        """
        # Detect total rows if streaming to show progress
        texts = self.load(dataset_name, split=split, streaming=True)
        if save_dir is None:
            save_dir = self._config_hf_data()
        os.makedirs(save_dir, exist_ok=True)
        safe_name = dataset_name.replace("/", "_")
        out_path = os.path.join(save_dir, f"{safe_name}__{split}.txt")
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            for text in texts:
                if num_rows is not None and count >= num_rows:
                    break
                f.write(text.replace("\n", " ") + "\n")
                count += 1
        return {"path": out_path, "rows": count}

    # ------------------------------------------------------------------
    # Cache status + interrupted-download recovery
    # ------------------------------------------------------------------
    def cache_status(self, dataset_name, split=None, name=None):
        """Check whether a dataset is already fully cached locally.

        Returns a dict: {'status': 'cached'|'partial'|'missing',
                          'rows': int, 'files': [], 'incomplete': [],
                          'count': int, 'bytes': int}
        Status is 'cached' when the datasets cache holds complete data
        files (parquet/arrow/json) for the requested split/config.
        'partial' means some files exist or stale .incomplete markers were
        found - in which case the next load() resumes / re-downloads.
        """
        cache = self._resolve_cache(None)
        if not cache:
            return {"status": "missing", "rows": 0, "files": [], "incomplete": [], "count": 0, "bytes": 0}
        base = self._ns_base_dir(cache, dataset_name)
        cfg_root = os.path.join(base, name) if (name and os.path.isdir(base)) else base
        if not os.path.isdir(cfg_root):
            return {"status": "missing", "rows": 0, "files": [], "incomplete": [], "count": 0, "bytes": 0}

        found_files, bad = [], []
        data_bytes = 0
        for root, _dirs, files in os.walk(cfg_root):
            for d in _dirs:
                if d.endswith(".incomplete"):
                    bad.append(os.path.join(root, d))
            for f in files:
                p = os.path.join(root, f)
                if f.endswith(".incomplete"):
                    bad.append(p)
                elif f.endswith((".parquet", ".arrow", ".json")):
                    found_files.append(p)
                    try:
                        data_bytes += os.path.getsize(p)
                    except OSError:
                        pass
        if bad:
            return {"status": "partial", "rows": 0, "files": found_files,
                    "incomplete": bad, "count": len(found_files), "bytes": data_bytes}
        if not found_files:
            return {"status": "missing", "rows": 0, "files": [], "incomplete": [],
                    "count": 0, "bytes": 0}
        return {"status": "cached", "rows": len(found_files), "files": found_files,
                "incomplete": [], "count": len(found_files), "bytes": data_bytes}

    @staticmethod
    def _ns_candidates(dataset_name):
        """Candidate cache namespace dir names for a dataset id.

        datasets derives them as {owner}___{module_name} where module_name is
        the camel-cased repo name converted to snake_case (hyphens kept) and
        lower-cased, e.g. 'openbmb/UltraData-SFT-2605' -> 'openbmb___ultra_data-sft-2605'.
        Older versions kept non-alphanumerics as underscores. We try every
        form so already-downloaded data is detected regardless of the version
        that created the cache.
        """
        owner, sep, repo = str(dataset_name).partition("/")
        candidates = set()
        raw = str(dataset_name)
        if sep:
            snake = _camel_to_snake(repo)
            candidates.add(f"{owner}___{snake}")
            candidates.add(f"{owner}___{snake.lower()}")
            candidates.add(f"{_norm(owner)}___{_norm(repo)}")
        else:
            candidates.add(re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_"))
            candidates.add(re.sub(r"[^a-z0-9_]+", "_", raw).strip("_"))
        candidates.add(raw.replace("/", "___"))
        candidates.add(raw.replace("/", "___").lower())
        return {c for c in candidates if c}

    def _ns_base_dir(self, cache, dataset_name):
        """Return the existing namespace dir for a dataset under the cache root,
        or the most likely one if none exists yet."""
        likely = []
        for ns in self._ns_candidates(dataset_name):
            for base in (os.path.join(cache, "datasets", ns), os.path.join(cache, ns)):
                likely.append(base)
                if os.path.isdir(base):
                    return base
        return likely[0] if likely else None

    def _clean_incomplete(self, max_age_minutes=30):
        """Remove stale '*.incomplete' markers left by interrupted downloads.

        HF writes these files while a snapshot/data file is being downloaded.
        If a prior run died mid-way, the marker stays behind and load_dataset
        can fail (or re-download from scratch). Removing old markers lets the
        next load resume cleanly. Fresh markers (<max_age_minutes old) are
        kept so an in-flight download in another process is not disturbed.
        Returns the number of markers removed.
        """
        import time
        cache = self._resolve_cache(None)
        if not cache:
            return 0
        removed = 0
        now = time.time()
        for root, dirs, files in os.walk(cache, topdown=True):
            # stale *.incomplete DIRECTORY markers (interrupted builder state)
            for d in list(dirs):
                if d.endswith(".incomplete"):
                    p = os.path.join(root, d)
                    try:
                        if now - os.path.getmtime(p) > max_age_minutes * 60:
                            shutil.rmtree(p, ignore_errors=True)
                            removed += 1
                            continue
                    except OSError:
                        pass
            for f in files:
                if f.endswith(".incomplete"):
                    p = os.path.join(root, f)
                    try:
                        if now - os.path.getmtime(p) > max_age_minutes * 60:
                            os.remove(p)
                            removed += 1
                    except OSError:
                        pass
        return removed

    def _clean_incomplete_once(self, max_age_minutes=30):
        """Run the stale-marker sweep at most once per process.

        The full walk over the (potentially multi-GB) cache is expensive, so
        it is guarded by a class-level flag. Subsequent dataset loads skip the
        sweep entirely and rely on cache_status for quick per-dataset checks.
        """
        if getattr(HFLoader, "_swept", False):
            return 0
        HFLoader._swept = True
        return self._clean_incomplete(max_age_minutes=max_age_minutes)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_cache(self, cache_dir):
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            return cache_dir
        cfg_cache = self._config_hf_cache()
        if cfg_cache:
            os.makedirs(cfg_cache, exist_ok=True)
            return cfg_cache
        return None

    def _config_hf_cache(self):
        """Helper to read cache path from config without importing app config (avoid cycle)."""
        try:
            from ..config import resolve_path
            return resolve_path('hf_cache')
        except Exception:
            pass
        return None

    def _config_hf_data(self):
        """Resolve HF download/save dir (config 'hf_data', default: <project>/data/hf)."""
        try:
            from ..config import resolve_path
            resolved = resolve_path('hf_data')
            if resolved:
                return resolved
        except Exception:
            pass
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "hf"
        )

    def _detect_schema(self, ds, dataset_name, sample_rows=5, verbose=True):
        """Detect native schema -> semantic role mapping.

        1. Inspect actual native columns (+ types + first sample values)
        2. Map column names to semantic roles (HIDE/medium confidence)
        3. Unknown columns are NEVER dropped silently -> reported as metadata
        4. If nothing maps with confidence, raise (avoid training on garbage)

        Returns (col, fields, report_str). col=None + fields={} means no mapping.
        """
        cols = []
        if hasattr(ds, 'column_names'):
            cols = ds.column_names if isinstance(ds.column_names, list) else list(ds.column_names)

        # Gather types + sample values per column (structure/value inspection)
        types = {}
        samples = {}
        try:
            features = getattr(ds, 'features', None)
            for c in cols:
                f = features[c] if features else None
                try:
                    types[c] = str(getattr(f, 'dtype', type(f).__name__))
                except Exception:
                    types[c] = 'unknown'
        except Exception:
            pass
        try:
            sample_iter = iter(ds)
            for _ in range(sample_rows):
                row = next(sample_iter)
                if not isinstance(row, dict):
                    row = dict(row)
                for c in cols:
                    if c in row and c not in samples:
                        v = row[c]
                        if isinstance(v, (list, dict)):
                            samples[c] = type(v).__name__
                        else:
                            s = str(v).strip()
                            samples[c] = s[:60] if s else ''
        except StopIteration:
            pass
        except Exception:
            pass

        # Map columns -> roles
        fields = {}
        unknown = []
        for c in cols:
            role, conf = _role_for_col(c)
            if role:
                fields[c] = role
                self._schema_entries.append((c, role, conf))
            else:
                unknown.append(c)
                self._schema_entries.append((c, None, "UNKNOWN"))

        # Ordered by role priority (instruction -> reasoning -> answer)
        ordered = dict(sorted(fields.items(), key=lambda kv: _ROLE_PRIORITY.get(kv[1], 50)))

        report = self._schema_report(
            dataset_name, cols, types, samples, ordered, unknown, fields
        )
        if verbose:
            print(report)

        if not ordered and unknown:
            raise RuntimeError(
                f"[NovaCore] SCHEMA ERROR: dataset '{dataset_name}' columns "
                f"({', '.join(cols)}) could not be mapped to any semantic role. "
                f"Add a manual 'fields' map in data_config.json (see report above)."
            )

        # column name kept for message compatibility (unused when fields set)
        col = None
        return col, ordered, report

    def _detect_local_schema(self, first_rows, dataset_name, verbose=True):
        """Role mapping for locally-parsed JSON records (no datasets object).

        Mirrors `_detect_schema` using the same role/confidence heuristics so
        the printed report + `fields` serialization behave identically for
        the disk-download path.
        """
        self._schema_entries = []
        cols = []
        types = {}
        samples = {}
        for row in first_rows:
            if not isinstance(row, dict):
                continue
            for c, v in row.items():
                if c not in cols:
                    cols.append(c)
                    types[c] = type(v).__name__
                if c not in samples:
                    if isinstance(v, (list, dict)):
                        samples[c] = type(v).__name__
                    else:
                        s = str(v).strip()
                        samples[c] = s[:60] if s else ''
        fields = {}
        unknown = []
        for c in cols:
            role, conf = _role_for_col(c)
            if role:
                fields[c] = role
                self._schema_entries.append((c, role, conf))
            else:
                unknown.append(c)
                self._schema_entries.append((c, None, "UNKNOWN"))
        ordered = dict(sorted(fields.items(), key=lambda kv: _ROLE_PRIORITY.get(kv[1], 50)))
        report = self._schema_report(dataset_name, cols, types, samples, ordered, unknown, fields)
        if verbose:
            print(report)
        if not ordered and unknown:
            raise RuntimeError(
                f"[NovaCore] SCHEMA ERROR: dataset '{dataset_name}' columns "
                f"({', '.join(cols)}) could not be mapped to any semantic role. "
                f"Add a manual 'fields' map in data_config.json (see report above)."
            )
        return None, ordered, report

    def _resolve_split_files(self, dataset_name, split, name):
        """List the repo and find the raw data file paths for (config, split).

        Card-YAML repos keep split shards under `data/<split>/<config>/...`, so
        a prefix filter on `list_repo_files` yields exactly the requested
        split's files (this is how we bypass the datasets 5.x eager allocation
        bug that picks the FIRST config/split glob). Returns a sorted list of
        repo paths ([] when unresolved -> caller falls back to streaming).
        """
        datasets_obj, hub = self._import_optional()
        try:
            files = hub.list_repo_files(dataset_name, repo_type="dataset")
        except Exception:
            return []
        prefix = f"data/{split}/"
        if name:
            prefix += f"{name}/"
        cands = [f for f in files if f.startswith(prefix) and f.endswith(".jsonl")]
        if not cands and name is None:
            cands = [f for f in files if f.startswith(f"data/{split}/") and f.endswith(".jsonl")]
        return sorted(cands)

    # ------------------------------------------------------------------
    # Accumulated pool + dedupe (random/add_datasets support)
    #
    # One pool per (dataset, config, split):
    #   data/hf_cache/pool/<safe_key>/pool.jsonl   - {"uid","text"} JSON lines
    #   data/hf_cache/pool/<safe_key>/manifest.json - metadata + row counts
    # Rows are NEVER duplicated: a new row is added only when its uid is not
    # already in the pool. add_datasets=ON -> pull N NEW rows; OFF -> reuse
    # whatever is already downloaded (nothing new added).
    # ------------------------------------------------------------------
    def _pool_dir(self, cache_dir, dataset_name, name, split):
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{dataset_name}__{name or 'default'}__{split}")
        return os.path.join(cache_dir, "pool", key)

    def _uid_for(self, row, uid_key, text):
        """Stable per-row identity: uid_key column value when available, else
        a fast 32-bit content hash of the extracted text (~10x faster than
        sha1 with negligible collision risk for 32-bit content addressing)."""
        if uid_key and isinstance(row, dict):
            v = row.get(uid_key)
            if v not in (None, "", []):
                return "u:" + str(v)
        # Fast 32-bit hash for content addressing (FNV-1a, no full sha1).
        h = 2166136261
        for b in text.encode("utf-8", "replace"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        return f"h:{h:08x}"

    def _pool_uid_key(self, uid_key, cols):
        """Pick the uid column: explicit config beats an auto-detected one."""
        if uid_key:
            return uid_key
        for c in cols or []:
            if _normalize_col(c) in _UID_COLS:
                return c
        return None

    def _load_pool(self, cache_dir, dataset_name, name, split, uid_key=None,
                   need_uids=False):
        """Load pool metadata for (dataset, config, split) with minimal RAM.

        By default **only** the row-count (from manifest.json) and filesystem
        paths are loaded – text content stays on disk.  When *need_uids* is
        ``True`` (typically ``add_datasets=True``), the set of UIDs is loaded
        for dedup; for pools exceeding ``_UID_RAM_THRESHOLD`` rows a compact
        32-bit hash set (~4 bytes/row) replaces the full Python set.

        Returns a dict with ``row_count``, ``uids``, ``meta``, ``rows_path``,
        ``meta_path``, or ``None`` when no pool AND no legacy seed exists.
        """
        d = self._pool_dir(cache_dir, dataset_name, name, split)
        rows_path = os.path.join(d, "pool.jsonl")
        meta_path = os.path.join(d, "manifest.json")
        uids = set()
        meta = {}
        row_count = 0

        if os.path.isfile(rows_path):
            # --- fast path: read manifest only (no row content) -----------
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
            row_count = meta.get("rows", 0)
            pool_size = _fmt_bytes(os.path.getsize(rows_path)) if os.path.isfile(rows_path) else "0 B"
            print(f"[NovaCore]   pool found: {row_count} row(s), {pool_size} on disk")

            # If we don't need UIDs we can return immediately (O(1) memory).
            if need_uids and row_count > 0:
                # ALWAYS use compact hash set to save RAM (~4 bytes/row
                # vs ~100+ for Python set).  Critical on low-RAM machines.
                print(f"[NovaCore]   loading {row_count:,} UIDs for dedup (compact hash set) ...", flush=True)
                uids = self._load_pool_uids(rows_path, uid_key, compact=True)
                print(f"[NovaCore]   UIDs loaded: {len(uids):,} unique  ({_fmt_bytes(os.path.getsize(rows_path)) if os.path.isfile(rows_path) else '?'})")

            meta.setdefault("dataset", dataset_name)
            meta["config"] = name
            meta["split"] = split
            meta.setdefault("uid_key", uid_key)
            meta["rows"] = row_count
            return {
                "dir": d, "uids": uids, "meta": meta,
                "row_count": row_count,
                "rows_path": rows_path, "meta_path": meta_path,
            }

        # --- no pool yet -> seed from legacy texts-cache archives ---------
        # SAFETY: skip if total cache files > 100 MB (would blow up RAM).
        texts_dir = os.path.join(cache_dir, "texts")
        prefix = re.sub(r"[^A-Za-z0-9._-]+", "_",
                        f"{dataset_name}__{name or 'default'}__{split}")
        seeded = []
        if os.path.isdir(texts_dir):
            candidates = sorted(glob.glob(os.path.join(texts_dir, prefix + "__*.jsonl")))
            # Check total size before reading — RAM guard
            total_cache_bytes = sum(os.path.getsize(f) for f in candidates)
            if total_cache_bytes > 100 * 1024 * 1024:  # 100 MB limit
                print(f"[NovaCore]   skipping legacy text-cache seeding ({_fmt_bytes(total_cache_bytes)} > 100 MB limit)", flush=True)
            else:
                for f in candidates:
                    try:
                        with open(f, encoding="utf-8", buffering=8*1024*1024) as fh:
                            for ln in fh:
                                ln = ln.strip()
                                if not ln:
                                    continue
                                try:
                                    t = json.loads(ln)
                                except ValueError:
                                    continue
                                if isinstance(t, str) and t:
                                    uid = "h:" + hashlib.sha1(
                                        t.encode("utf-8", "replace")).hexdigest()
                                    seeded.append({"uid": uid, "text": t})
                    except OSError:
                        continue
        if seeded:
            try:
                os.makedirs(d, exist_ok=True)
                with open(rows_path, "w", encoding="utf-8") as fh:
                    for item in seeded:
                        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                meta = {
                    "dataset": dataset_name, "config": name, "split": split,
                    "uid_key": uid_key, "rows": len(seeded), "seeded": True,
                }
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
            except OSError:
                pass
            # build UID set from the just-written file
            if need_uids:
                uids = self._load_pool_uids(rows_path, uid_key,
                                            compact=(len(seeded) > _UID_RAM_THRESHOLD))
            else:
                uids = set()
            return {
                "dir": d, "uids": uids, "meta": meta,
                "row_count": len(seeded),
                "rows_path": rows_path, "meta_path": meta_path,
            }
        return None

    # ------------------------------------------------------------------
    def _load_pool_uids(self, rows_path, uid_key=None, compact=False):
        """Stream UIDs from *rows_path* into a dedup set.

        For large pools (``compact=True``) a ``_CompactHashSet`` is used
        (~4 bytes / row instead of ~100+ for a Python ``set``).  Text
        content is never loaded – only the ``uid`` field of each JSON line.

        Optimized: 8 MB read buffer + regex UID extraction (no full JSON
        parse needed — avoids MemoryError on rows with huge text fields).
        """
        import re
        from time import time as _time
        if compact:
            uid_set = _CompactHashSet()
        else:
            uid_set = set()
        add = uid_set.add
        # Extract just the "uid" value — avoids full JSON parse of huge lines
        _uid_re = re.compile(r'"uid"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        BUF = 8 * 1024 * 1024  # 8 MB
        _t0 = _time()
        count = 0
        try:
            with open(rows_path, "r", encoding="utf-8", buffering=BUF) as fh:
                for ln in fh:
                    if not ln or ln[0] != "{":
                        continue
                    m = _uid_re.search(ln)
                    if m:
                        add(m.group(1))
                    count += 1
                    if count % 500_000 == 0:
                        elapsed = _time() - _t0
                        rate = count / max(1, elapsed)
                        print(f"[NovaCore]     · UIDs: {count:,} scanned, {len(uid_set):,} unique  ({elapsed:.0f}s, ~{rate:,.0f} rows/s)", flush=True)
        except OSError:
            pass
        elapsed = _time() - _t0
        if count >= 500_000:
            print(f"[NovaCore]     · UID scan done: {count:,} rows in {elapsed:.1f}s", flush=True)
        return uid_set

    def _append_pool(self, pool, new_rows, fields=None, col=None):
        """Append new rows to the on-disk pool + manifest.

        Text rows are flushed to pool.jsonl line-by-line (never builds a
        giant string in memory).  The in-memory ``pool["uids"]`` set is
        updated for dedup.
        """
        if not new_rows:
            return
        try:
            os.makedirs(pool["dir"], exist_ok=True)
            with open(pool["rows_path"], "a", encoding="utf-8",
                      buffering=8*1024*1024) as fh:
                for item in new_rows:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            meta = pool["meta"]
            meta["rows"] = (meta.get("rows") or 0) + len(new_rows)
            meta["added_last"] = len(new_rows)
            if fields is not None:
                meta["fields"] = fields
            if col is not None:
                meta["column"] = col
            with open(pool["meta_path"], "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except OSError:
            return
        # update in-memory counters
        pool["uids"].update(r["uid"] for r in new_rows)
        pool["row_count"] = pool.get("row_count", 0) + len(new_rows)
        pool["meta"]["rows"] = pool["row_count"]

    def _pool_texts(self, pool, cap=None):
        """Return a ``TextStream`` that reads texts lazily from pool.jsonl."""
        rows_path = pool.get("rows_path", "")
        count = pool.get("row_count", 0)
        if rows_path and os.path.isfile(rows_path):
            return TextStream(rows_path, count=count)
        return []

    def _fetch_split_to_disk(self, dataset_name, split, name, files, cache_dir,
                             num_rows, random_sample, fields, texts_cache,
                             verbose, label, log, pool=None, add_datasets=False,
                             uid_key=None, local_only=False):
        """DOWNLOAD-mode: real bytes ON DISK for one (config, split).

        Downloads the raw shard files of the requested split into the HF hub
        cache (data/hf_cache/hub) - persistent and resumable - then parses them
        locally, deduping against the accumulated pool. Every downloaded shard
        is FULLY consumed (all rows preserved, nothing discarded) so no data is
        wasted after the network cost. add_datasets=ON pulls only files not yet
        on disk; OFF reuses only what is already present. local_only=True skips
        any shard that isn't on disk yet (zero network, absorb-downloaded-only).

        Returns a ``TextStream`` (or list for tiny pools); ``None`` means the
        route failed (caller then falls back to the per-split streaming
        iterable).
        """
        if pool is None:
            pool = self._load_pool(cache_dir, dataset_name, name, split,
                                   uid_key=uid_key, need_uids=add_datasets)
        if pool is None:
            pdir = self._pool_dir(cache_dir, dataset_name, name, split)
            pool = {
                "dir": pdir, "uids": _CompactHashSet(100_000), "meta": {},
                "row_count": 0,
                "rows_path": os.path.join(pdir, "pool.jsonl"),
                "meta_path": os.path.join(pdir, "manifest.json"),
            }
        if pool.get("row_count", 0) and not add_datasets:
            texts = self._pool_texts(pool)
            print(f"[NovaCore] {label}: pool reuse ({pool['row_count']} row(s)) - add_datasets OFF", flush=True)
            return texts

        datasets_obj, hub = self._import_optional()
        hub_cache = os.path.join(cache_dir, "hub")
        try:
            os.makedirs(hub_cache, exist_ok=True)
        except OSError:
            return None

        add_mode = bool(add_datasets or not pool.get("row_count", 0))
        want = num_rows if add_mode else None

        total = len(files)
        log.info(f"{label}: DOWNLOAD -> split '{split}' shards resolved: {total} file(s) in repo")
        log.info(
            f"{label}: DOWNLOAD -> absorbing {'all' if want is None else str(want)}+ unique row(s)"
            f" into pool; every downloaded shard is fully consumed (nothing discarded)"
        )

        keep = []
        new_count = 0          # total NEW rows flushed to disk so far
        skipped = 0
        skipped_local = 0
        dup_skipped = 0
        failed = 0
        col = None
        merged_fields = fields
        report = None
        detected = False
        used_uid = uid_key

        # Resume support: load set of already-processed shards from manifest
        # so we can skip re-reading them (fast resume for interrupted builds).
        _done_shards = set(pool.get("meta", {}).get("done_shards", []))
        if _done_shards:
            print(f"[NovaCore]   resume: {len(_done_shards)} shard(s) already processed, skipping", flush=True)

        def fetch_one(f):
            try:
                return hub.hf_hub_download(
                    dataset_name, f, repo_type="dataset",
                    cache_dir=hub_cache, local_files_only=True,
                )
            except Exception:
                return None

        for idx, f in enumerate(files, 1):
            if want is not None and new_count >= want:
                break
            # Resume: skip shards already fully processed in a prior run
            _fname = os.path.basename(f)
            if _fname in _done_shards:
                print(f"[NovaCore] {label}: [file {idx}/{total}] {_fname} - already processed, skip", flush=True)
                continue
            local = fetch_one(f)
            if local is None:
                if local_only:
                    skipped_local += 1
                    continue
                try:
                    print(f"[NovaCore] {label}: [file {idx}/{total}] {os.path.basename(f)} - downloading ...", flush=True)
                    local = hub.hf_hub_download(
                        dataset_name, f, repo_type="dataset", cache_dir=hub_cache,
                    )
                except Exception:
                    print(f"[NovaCore] {label}: [file {idx}/{total}] {os.path.basename(f)} - DOWNLOAD FAILED, skipping", flush=True)
                    failed += 1
                    continue
            else:
                shard_size = _fmt_bytes(os.path.getsize(local)) if os.path.isfile(local) else "?"
                print(f"[NovaCore] {label}: [file {idx}/{total}] {os.path.basename(f)} - on disk ({shard_size}), parsing ...", flush=True)
            keep.append(local)

            if not detected:
                first_rows = []
                try:
                    with open(local, "r", encoding="utf-8", buffering=8*1024*1024) as fh:
                        for ln in fh:
                            if not ln or ln[0] != "{":
                                continue
                            try:
                                row = json.loads(ln)
                            except ValueError:
                                continue
                            if isinstance(row, dict):
                                first_rows.append(row)
                                if len(first_rows) >= 5:
                                    break
                except OSError:
                    log.warn(
                        f"{label}: DOWNLOAD -> cannot read shard for schema "
                        f"(skip) {os.path.basename(local)}"
                    )
                    failed += 1
                    continue
                try:
                    col, use_fields, report = self._detect_local_schema(
                        first_rows, dataset_name, verbose
                    )
                except RuntimeError:
                    log.error(f"{label}: DOWNLOAD -> schema could not be mapped")
                    return None
                merged_fields = fields or use_fields or {}
                if not merged_fields:
                    log.error(f"{label}: DOWNLOAD -> no usable column mapping")
                    return None
                cols = list(first_rows[0].keys()) if first_rows else []
                used_uid = self._pool_uid_key(uid_key, cols)
                detected = True
                print(f"[NovaCore] {label}: schema detected -> {list(merged_fields.values()) if merged_fields else 'none'} (uid_key={used_uid})", flush=True)

            # Parse shard → flush entire shard to pool at once.
            # Safety cap at 50K rows prevents OOM on huge shards.
            # This is faster than per-3K flush: 1 write per shard.
            file_rows = []
            try:
                fh = open(local, "r", encoding="utf-8", buffering=8*1024*1024)
            except OSError:
                log.warn(
                    f"{label}: DOWNLOAD -> cannot open shard (skip) "
                    f"{os.path.basename(local)}"
                )
                failed += 1
                continue
            try:
                with fh:
                    for ln in fh:
                        if not ln or ln[0] != "{":
                            continue
                        if want is not None and new_count >= want:
                            break
                        try:
                            row = json.loads(ln)
                        except ValueError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        text = self._row_text(row, col, merged_fields)
                        if not text:
                            skipped += 1
                            continue
                        uid = self._uid_for(row, used_uid, text)
                        if uid in pool["uids"]:
                            dup_skipped += 1
                            continue
                        pool["uids"].add(uid)
                        # Store uid + text + ALL original columns (except the
                        # mapped text column which is already in `text`).
                        # This preserves source/domain/think_type etc. for
                        # quality training without duplicating the heavy
                        # messages content.
                        pool_row = {"uid": uid, "text": text}
                        _mapped_text_col = (merged_fields or {}).get(
                            "text", col or "text"
                        )
                        for k, v in row.items():
                            if k == _mapped_text_col or k == uid_key or k in pool_row:
                                continue
                            pool_row[k] = v
                        file_rows.append(pool_row)
                        # Safety cap: flush mid-shard if >50K rows (OOM guard)
                        if len(file_rows) >= 50_000:
                            self._append_pool(pool, file_rows,
                                              fields=merged_fields, col=col)
                            new_count += len(file_rows)
                            file_rows = []
                            print(
                                _c(_Color.GREY,
                                    f"    · absorbed {new_count} NEW rows into pool" +
                                    (f" / {want}" if want else "")),
                                flush=True,
                            )
                            gc.collect()
            except OSError:
                log.warn(
                    f"{label}: DOWNLOAD -> interrupted while reading shard "
                    f"(kept partial) {os.path.basename(local)}"
                )
            # Flush ALL remaining rows from this shard at once
            if file_rows:
                self._append_pool(pool, file_rows, fields=merged_fields, col=col)
                new_count += len(file_rows)
                file_rows = []
                print(
                    _c(_Color.GREY,
                        f"    · absorbed {new_count} NEW rows into pool" +
                        (f" / {want}" if want else "")),
                    flush=True,
                )
                # Free memory after each shard — critical for 12 GB RAM
                gc.collect()
            # Log shard completion
            file_dup = dup_skipped  # approximate dup count for this shard
            _done_shards.add(_fname)
            print(
                _c(_Color.GREY,
                    f"    · shard done: {_fname}"),
                flush=True,
            )
            # Persist done_shards set to manifest after each shard so an
            # interrupted run can resume without re-reading finished shards.
            try:
                pool["meta"]["done_shards"] = sorted(_done_shards)
                with open(pool["meta_path"], "w", encoding="utf-8") as fh:
                    json.dump(pool["meta"], fh, ensure_ascii=False, indent=2)
            except OSError:
                pass

        if failed:
            log.warn(f"{label}: DOWNLOAD -> {failed} file(s) failed, used {len(keep)}")
        if skipped_local:
            log.info(
                f"{label}: DOWNLOAD -> {skipped_local} file(s) not yet on disk "
                f"(absorb_local_only: skipped, already-downloaded only)"
            )
        if not keep and not _done_shards:
            return None
        # Resume: if all shards were already done in a prior run, fall through
        # and return pool texts via the normal pool_texts path below.

        # Mark pool as "complete" only if ALL shards were consumed (no
        # failures, no local-only skips, and want was None/full).
        # An incomplete pool will be resumed on the next run.
        _all_consumed = (len(keep) == total and failed == 0
                         and skipped_local == 0
                         and (want is None or new_count >= want))
        try:
            pool["meta"]["complete"] = _all_consumed
            pool["meta"]["shards_consumed"] = len(keep)
            pool["meta"]["shards_total"] = total
            if failed:
                pool["meta"]["shards_failed"] = failed
            with open(pool["meta_path"], "w", encoding="utf-8") as fh:
                json.dump(pool["meta"], fh, ensure_ascii=False, indent=2)
        except OSError:
            pass
        try:
            total_bytes = sum(os.path.getsize(p) for p in keep)
        except OSError:
            total_bytes = 0
        log.info(
            f"{label}: DOWNLOAD -> {len(keep)} shard file(s) fully consumed "
            f"({_fmt_bytes(total_bytes)}) - all rows preserved in pool"
        )
        if want is not None and new_count < want:
            log.warn(
                f"{label}: DOWNLOAD -> only {new_count} NEW row(s) absorbed "
                f"(all local shards consumed before {want})"
            )
        if dup_skipped:
            log.info(
                f"{label}: DOWNLOAD -> {dup_skipped} duplicate(s) skipped (already in pool)"
            )

        self._skipped = skipped + dup_skipped

        # Write texts_cache for non-default-split bounded downloads so the
        # next run can reuse without re-parsing raw shards.  Extract new
        # texts from pool.jsonl (they were flushed during shard processing).
        if new_count and texts_cache:
            try:
                os.makedirs(os.path.dirname(texts_cache), exist_ok=True)
                with open(texts_cache, "w", encoding="utf-8") as fh_cache:
                    # Read the last `new_count` lines from pool.jsonl
                    with open(pool["rows_path"], encoding="utf-8") as fh_pool:
                        for ln in fh_pool:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                item = json.loads(ln)
                                text = item.get("text") if isinstance(item, dict) else None
                                if text and isinstance(text, str):
                                    fh_cache.write(json.dumps(text, ensure_ascii=False) + "\n")
                            except (ValueError, json.JSONDecodeError):
                                continue
            except OSError:
                pass

        self.last_schema = {
            "dataset": dataset_name, "column": col, "fields": merged_fields,
            "report": report, "source": "disk-files", "uid_key": used_uid,
        }
        texts = self._pool_texts(pool)
        n = pool.get("row_count", 0)
        if n:
            log.ok(f"{label}: {n} row(s) ready for training  ({log.total()})")
        else:
            log.skip(f"{label}: 0 usable rows (check dataset/split/columns)")
        return texts

    def _schema_report(self, dataset_name, cols, types, samples, ordered, unknown, fields):
        """ASCII report: native columns, detected mapping, confidence, unknowns."""
        lines = []
        lines.append("==================================================")
        lines.append("NOVACORE DATASET SCHEMA")
        lines.append("==================================================")
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Native columns ({len(cols)}):")
        for c in cols:
            t = types.get(c, '?')
            s = samples.get(c, '')
            suf = f"  sample: {s!r}" if s else ""
            lines.append(f"  {c:<24} {t:<12}{suf}")
        lines.append("")
        lines.append("Detected mapping:")
        if ordered:
            for c, role in ordered.items():
                lines.append(f"  {c:<22} -> {role}")
        else:
            lines.append("  (none - ambiguous/unknown columns)")
        # confidence breakdown
        confs = {}
        for c, role, conf in getattr(self, '_schema_entries', []):
            confs.setdefault((c, role), conf)
        if ordered:
            lines.append("")
            lines.append("Confidence:")
            for c, role in ordered.items():
                conf = confs.get((c, role), "MEDIUM")
                lines.append(f"  {c:<20} -> {role:<12} [{conf}]")
        if unknown:
            lines.append("")
            lines.append("Unknown (kept as metadata, NOT trained):")
            for c in unknown:
                t = types.get(c, '?')
                s = samples.get(c, '')
                suf = f"  sample: {s!r}" if s else ""
                lines.append(f"  {c:<24} {t:<12}{suf}")
        lines.append("")
        if ordered and unknown:
            lines.append("Mapping clear - training proceeds. Unknown columns preserved as metadata.")
        elif ordered and not unknown:
            lines.append("All columns mapped. No unknown columns.")
        else:
            lines.append("WARNING: no semantic mapping found - manual fields needed.")
        lines.append("==================================================")
        return "\n".join(lines)

    def _extract_texts(self, ds, col, num_rows, fields=None, random_sample=False, log=None, progress_word="streamed"):
        """Convert loaded/streaming HF dataset to text strings.

        If `random_sample` is True and num_rows is set, rows are picked at
        random positions instead of from the start (useful for quick training
        on a subset). Falls back to sequential when impossible.
        """
        self._skipped = 0
        is_streaming = hasattr(ds, 'to_iterable_dataset') or type(ds).__name__ == 'IterableDataset'
        if is_streaming:
            # streaming
            if random_sample and num_rows is not None:
                # reservoir sampling: k=num_rows, uniform across the whole stream
                reservoir = []
                seen = 0
                for row in ds:
                    text = self._row_text(row, col, fields)
                    if not text:
                        self._skipped += 1
                        continue
                    seen += 1
                    if len(reservoir) < num_rows:
                        reservoir.append(text)
                    else:
                        j = random.randrange(seen)
                        if j < num_rows:
                            reservoir[j] = text
                def gen_r():
                    for t in reservoir:
                        yield t
                return gen_r()

            def gen():
                count = 0
                for row in ds:
                    if num_rows is not None and count >= num_rows:
                        break
                    text = self._row_text(row, col, fields)
                    if text:
                        count += 1
                        yield text
                        if count % 1000 == 0 and log is not None:
                            print(_c(_Color.GREY, f"    · {progress_word} {count} rows") +
                                  (f" / {num_rows}" if num_rows else ""), flush=True)
                    else:
                        self._skipped += 1
            return gen()
        else:
            # materialized
            texts = []
            if random_sample and num_rows is not None:
                n = len(ds)
                if n > num_rows:
                    # random positions from anywhere in the dataset
                    indices = random.sample(range(n), num_rows)
                    for i in indices:
                        row = ds[i]
                        text = self._row_text(row, col, fields)
                        if text:
                            texts.append(text)
                        else:
                            self._skipped += 1
                    return texts
            _print_every = 10_000
            for i, row in enumerate(ds):
                if num_rows is not None and i >= num_rows:
                    break
                text = self._row_text(row, col, fields)
                if text:
                    texts.append(text)
                    if (i + 1) % _print_every == 0 and log is not None:
                        print(_c(_Color.GREY, f"    · loaded {i + 1} rows") +
                              (f" / {num_rows}" if num_rows else ""), flush=True)
                else:
                    self._skipped += 1
            return texts

    def _row_text(self, row, col, fields=None):
        """Extract a text string from one HF row dict.

        If `fields` is an ordered {column: role} map, serialize every listed
        column with <role>...</role> markers (schema-aware). Chat columns
        (list of {role, content}) are emitted as <user>/<assistant>/... turns.
        """
        if fields:
            return self._serialize_fields(row, fields)
        if isinstance(row, dict):
            if col and col in row:
                v = row[col]
                if isinstance(v, str):
                    return v
                if isinstance(v, list):
                    # chat-style 'messages' or list of strings -> join
                    parts = []
                    for item in v:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            content = (item.get('content') or item.get('text')
                                       or item.get('value') or item.get('message'))
                            if isinstance(content, str):
                                parts.append(content)
                            elif isinstance(content, list):
                                parts.extend(c for c in content if isinstance(c, str))
                    joined = " ".join(p for p in parts if p)
                    if joined:
                        return joined
                    return None
                try:
                    return str(v)
                except Exception:
                    return None
            for c in row.values():
                if isinstance(c, str):
                    return c
            return None
        elif hasattr(row, 'get'):
            return self._row_text(dict(row), col)
        elif isinstance(row, str):
            return row
        return None

    # ------------------------------------------------------------------
    # Schema-aware serialization (preserve column semantics)
    # ------------------------------------------------------------------
    def _serialize_fields(self, row, fields):
        """Serialize a row using an ordered {column: role} map.

        Output example:
            <instruction>Solve the problem.</instruction>
            <input>25 x 4</input>
            <output>100</output>
        For chat columns (list of {role, content}) the role is used as tag:
            <user>What is gravity?</user>
            <assistant>Gravity is...</assistant>
        A role starting with "@" means "take the tag from that column", e.g.
        {"text": "@role"} -> <prompter>/<assistant> using the row's role.
        """
        parts = []
        for col, role in fields.items():
            if col not in row:
                continue
            value = row[col]
            tag = role or col
            if isinstance(role, str) and role.startswith("@"):
                tag = row.get(role[1:], col)
            if isinstance(value, list):
                for item in value:
                    seg = self._field_segment(item, tag)
                    if seg:
                        parts.append(seg)
            else:
                seg = self._field_segment(value, tag)
                if seg:
                    parts.append(seg)
        return "\n".join(parts) if parts else None

    def _field_segment(self, value, tag):
        """Render one field value -> '<tag>text</tag>' (or None)."""
        if value is None:
            return None
        if isinstance(value, dict):
            role = value.get('role')
            content = value.get('content', value.get('text', value.get('value')))
            if role:
                tag = str(role)
            if isinstance(content, list):
                inner = []
                for c in content:
                    if isinstance(c, str):
                        inner.append(c)
                content = " ".join(inner) if inner else ""
            if isinstance(content, str):
                text = content.strip()
            else:
                try:
                    text = str(content).strip()
                except Exception:
                    return None
            return f"<{tag}>{text}</{tag}>" if text else None
        if isinstance(value, list):
            segs = []
            for v in value:
                s = self._field_segment(v, tag)
                if s:
                    segs.append(s)
            return "\n".join(segs) if segs else None
        if isinstance(value, str):
            text = value.strip()
            return f"<{tag}>{text}</{tag}>" if text else None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return f"<{tag}>{text}</{tag}>" if text else None

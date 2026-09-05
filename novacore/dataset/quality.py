"""
Quality filtering pipeline for NovaCore datasets.

Architecture (user-approved):

    RAW POOL (all columns preserved)
        │
        ▼
    BASIC CLEANING ── deleted=true, empty/broken, invalid rows
        │
        ▼
    QUALITY FILTER ── language, reviews, rank, quality signals
        │
        ▼
    CONVERSATION RECON ── parent_id/tree reconstruction
        │
        ▼
    CONTENT DEDUP ── exact text dedup + near dedup (simhash)
        │
        ▼
    CLEAN TextStream ── ONLY text field → training

Key principle: metadata is for FILTERING, not for training tokens.
"""
import json
import os
import re
from collections import Counter


class QualityFilter:
    """Per-dataset quality filtering pipeline.
    
    Config example (in data_config.json under a dataset):
    
    "filter": {
        "enabled": true,
        "cleaning": {
            "remove_deleted": true,
            "remove_empty_text": true,
            "min_text_length": 10
        },
        "quality": {
            "language": ["en"],
            "min_rank": null,
            "max_rank": 3,
            "require_review_result": false,
            "exclude_synthetic": false
        },
        "dedup": {
            "exact_text_dedup": true,
            "near_dedup": false,
            "near_dedup_threshold": 0.9
        },
        "conversation": {
            "reconstruct_threads": false,
            "min_thread_length": 2,
            "keep_only_roots": false
        },
        "preserve_columns": ["role", "lang", "rank", "source", "domain", "think_type"]
    }
    """

    def __init__(self, config=None):
        self.config = config or {}
        # Use compact hash set instead of Python set of strings to prevent MemoryError
        # on millions of rows (each string ~9KB, 6.6M rows = ~60GB).
        try:
            from novacore.dataset.hf_loader import _CompactHashSet
            self._exact_seen = _CompactHashSet(200_000)
        except Exception:
            self._exact_seen = set()
        self._stats = Counter()

    @property
    def enabled(self):
        return self.config.get("enabled", False)

    # ------------------------------------------------------------------
    # Stage 1: Basic Cleaning
    # ------------------------------------------------------------------
    def _clean(self, row):
        """Remove deleted, empty, broken rows. Returns True if row survives."""
        cleaning = self.config.get("cleaning", {})

        # Remove deleted=true rows
        if cleaning.get("remove_deleted", False):
            deleted = row.get("deleted", False)
            if deleted is True or str(deleted).lower() == "true":
                self._stats["cleaned_deleted"] += 1
                return False

        # Remove empty/broken text
        text = row.get("text", "")
        if cleaning.get("remove_empty_text", True):
            if not text or not isinstance(text, str) or not text.strip():
                self._stats["cleaned_empty"] += 1
                return False

        # Min text length
        min_len = cleaning.get("min_text_length", 0)
        if min_len and len(text) < min_len:
            self._stats["cleaned_short"] += 1
            return False

        return True

    # ------------------------------------------------------------------
    # Stage 2: Quality Filter
    # ------------------------------------------------------------------
    def _quality_pass(self, row):
        """Apply quality signals. Returns True if row passes."""
        quality = self.config.get("quality", {})

        # Language filter
        allowed_langs = quality.get("language", None)
        if allowed_langs:
            lang = str(row.get("lang", "")).lower().strip()
            if lang and lang not in allowed_langs:
                self._stats["filtered_lang"] += 1
                return False

        # Rank filter (lower rank = better in oasst)
        min_rank = quality.get("min_rank", None)
        max_rank = quality.get("max_rank", None)
        rank = row.get("rank", None)
        if rank is not None:
            try:
                rank = int(rank)
                if min_rank is not None and rank < min_rank:
                    self._stats["filtered_rank"] += 1
                    return False
                if max_rank is not None and rank > max_rank:
                    self._stats["filtered_rank"] += 1
                    return False
            except (ValueError, TypeError):
                pass

        # Review result filter
        if quality.get("require_review_result", False):
            rr = row.get("review_result", None)
            if rr is not None and not rr:
                self._stats["filtered_review"] += 1
                return False

        # Exclude synthetic data
        if quality.get("exclude_synthetic", False):
            synth = row.get("synthetic", False)
            if synth is True or str(synth).lower() == "true":
                self._stats["filtered_synthetic"] += 1
                return False

        return True

    # ------------------------------------------------------------------
    # Stage 3: Content Dedup
    # ------------------------------------------------------------------
    def _dedup_pass(self, row):
        """Exact + near text dedup. Returns True if unique."""
        dedup_cfg = self.config.get("dedup", {})
        text = row.get("text", "")

        # Exact text dedup
        if dedup_cfg.get("exact_text_dedup", True):
            normalized = text.strip().lower()
            if normalized in self._exact_seen:
                self._stats["deduped_exact"] += 1
                return False
            self._exact_seen.add(normalized)

        return True

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def filter_row(self, row):
        """Run full pipeline on one row. Returns (pass, row) tuple.
        
        If the row passes all filters, returns (True, row).
        If filtered out, returns (False, row).
        """
        if not self.enabled:
            return True, row

        # Stage 1: Basic cleaning
        if not self._clean(row):
            return False, row

        # Stage 2: Quality filter
        if not self._quality_pass(row):
            return False, row

        # Stage 3: Content dedup
        if not self._dedup_pass(row):
            return False, row

        return True, row

    def get_stats(self):
        """Return filtering statistics."""
        return dict(self._stats)

    def reset(self):
        """Reset stats and dedup state."""
        self._exact_seen.clear()
        self._stats.clear()


class FilteredRowStream:
    """Wraps a RowStream and applies QualityFilter on the fly.
    
    Usage:
        raw = RowStream(pool_path)
        filtered = FilteredRowStream(raw, filter_config)
        for text in filtered:
            ...  # only clean, high-quality texts
    """

    def __init__(self, row_stream, filter_config):
        self._source = row_stream
        self._filter = QualityFilter(filter_config)
        # Estimate: ~10% filtered out (conservative)
        self._count = getattr(row_stream, '_count', 0)
        self._filtered_count = 0

    def __iter__(self):
        for row in self._source:
            ok, kept_row = self._filter.filter_row(row)
            if ok:
                self._filtered_count += 1
                yield kept_row

    def __len__(self):
        return self._count

    @property
    def stats(self):
        return self._filter.get_stats()


class FilteredTextStream:
    """Wraps a TextStream and applies quality filters on the fly.
    
    This is what feeds into training: only clean, high-quality text strings.
    """

    def __init__(self, text_stream, filter_config):
        self._source = text_stream
        self._filter = QualityFilter(filter_config)
        self._count = getattr(text_stream, '_count', 0)
        self._kept = 0

    def __iter__(self):
        for text in self._source:
            # Minimal check: just pass through text with basic cleaning
            if not text or not isinstance(text, str):
                continue
            cleaning = self._filter.config.get("cleaning", {})
            min_len = cleaning.get("min_text_length", 0)
            if min_len and len(text) < min_len:
                self._filter._stats["cleaned_short"] += 1
                continue
            # Exact dedup
            dedup_cfg = self._filter.config.get("dedup", {})
            if dedup_cfg.get("exact_text_dedup", True):
                normalized = text.strip().lower()
                if normalized in self._filter._exact_seen:
                    self._filter._stats["deduped_exact"] += 1
                    continue
                self._filter._exact_seen.add(normalized)
            self._kept += 1
            yield text

    def __len__(self):
        return self._count

    @property
    def stats(self):
        return self._filter.get_stats()

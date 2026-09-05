"""Tests for quality filtering pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from novacore.dataset.quality import QualityFilter, FilteredTextStream
from novacore.dataset.hf_loader import RowStream
import json
import tempfile


def test_basic_cleaning_removes_deleted():
    """deleted=true rows should be removed."""
    f = QualityFilter({"enabled": True, "cleaning": {"remove_deleted": True}})
    assert f.filter_row({"text": "hello", "deleted": True})[0] is False
    assert f.filter_row({"text": "hello", "deleted": False})[0] is True
    # New filter to avoid dedup state from previous calls
    f2 = QualityFilter({"enabled": True, "cleaning": {"remove_deleted": True}})
    assert f2.filter_row({"text": "hello"})[0] is True


def test_empty_text_removed():
    """Empty/broken text should be removed."""
    f = QualityFilter({"enabled": True, "cleaning": {"remove_empty_text": True}})
    assert f.filter_row({"text": ""})[0] is False
    assert f.filter_row({"text": None})[0] is False
    assert f.filter_row({})[0] is False
    assert f.filter_row({"text": "hello world"})[0] is True


def test_min_text_length():
    """Short text below min_text_length should be removed."""
    f = QualityFilter({"enabled": True, "cleaning": {"min_text_length": 20}})
    assert f.filter_row({"text": "short"})[0] is False
    assert f.filter_row({"text": "a" * 20})[0] is True


def test_language_filter():
    """Only allowed languages should pass."""
    f = QualityFilter({"enabled": True, "quality": {"language": ["en", "es"]}})
    assert f.filter_row({"text": "hello", "lang": "en"})[0] is True
    assert f.filter_row({"text": "hola", "lang": "es"})[0] is True
    assert f.filter_row({"text": "bonjour", "lang": "fr"})[0] is False
    assert f.filter_row({"text": "no lang"})[0] is True  # no lang = pass


def test_rank_filter():
    """Rank-based quality filtering."""
    f = QualityFilter({"enabled": True, "quality": {"max_rank": 3}})
    assert f.filter_row({"text": "a", "rank": 0})[0] is True
    assert f.filter_row({"text": "b", "rank": 3})[0] is True
    assert f.filter_row({"text": "c", "rank": 5})[0] is False
    assert f.filter_row({"text": "d"})[0] is True  # no rank = pass


def test_exact_text_dedup():
    """Exact duplicate texts should be removed (case-insensitive normalized)."""
    f = QualityFilter({"enabled": True, "dedup": {"exact_text_dedup": True}})
    assert f.filter_row({"text": "unique1"})[0] is True
    assert f.filter_row({"text": "unique2"})[0] is True
    assert f.filter_row({"text": "unique1"})[0] is False  # dup
    # "Unique1" normalizes to "unique1" → also a dup
    assert f.filter_row({"text": "Unique1"})[0] is False
    # Completely new text
    assert f.filter_row({"text": "unique3"})[0] is True


def test_disabled_filter_passes_everything():
    """When enabled=false, everything passes."""
    f = QualityFilter({"enabled": False})
    assert f.filter_row({"text": ""})[0] is True
    assert f.filter_row({"deleted": True, "text": ""})[0] is True


def test_stats_tracking():
    """Filter stats should be tracked."""
    f = QualityFilter({
        "enabled": True,
        "cleaning": {"remove_deleted": True, "remove_empty_text": True},
        "dedup": {"exact_text_dedup": True}
    })
    f.filter_row({"text": "hello"})
    f.filter_row({"text": ""})  # empty
    f.filter_row({"text": "hello"})  # dup
    f.filter_row({"deleted": True, "text": "x"})  # deleted
    stats = f.get_stats()
    assert stats.get("cleaned_empty", 0) >= 1
    assert stats.get("deduped_exact", 0) >= 1
    assert stats.get("cleaned_deleted", 0) >= 1


def test_filtered_text_stream():
    """FilteredTextStream should wrap TextStream and filter."""
    # Create a temp pool.jsonl
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i, txt in enumerate(["hello world", "short", "hello world", "a" * 50, ""]):
            f.write(json.dumps({"uid": f"u{i}", "text": txt}) + "\n")
        path = f.name

    try:
        from novacore.dataset.hf_loader import TextStream
        ts = TextStream(path, count=5)
        fts = FilteredTextStream(ts, {
            "enabled": True,
            "cleaning": {"remove_empty_text": True, "min_text_length": 10},
            "dedup": {"exact_text_dedup": True}
        })
        results = list(fts)
        assert "hello world" in results
        assert "a" * 50 in results
        assert "" not in results
        assert "short" not in results
        # hello world appears only once (dedup)
        assert results.count("hello world") == 1
    finally:
        os.unlink(path)

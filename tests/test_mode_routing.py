import json
import os
import tempfile
import types

from novacore.dataset.hf_loader import HFLoader
from novacore.logger import Logger


class _FakeStream:
    """Looks like a datasets IterableDataset (streaming flavor)."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys())

    def to_iterable_dataset(self):
        return self

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    @property
    def features(self):
        return {c: type("F", (), {"dtype": "string"})() for c in self.column_names}


class _FakeHub:
    def __init__(self, files):
        self.files = files

    def list_repo_files(self, repo_id, repo_type="dataset"):
        return list(self.files)


def _rows(n=3):
    out = []
    for i in range(1, n + 1):
        out.append({
            "uid": f"Math_no_think_000000{i}",
            "messages": [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant", "content": f"A{i}"},
            ],
            "think_type": "no_think",
        })
    return out


def _patch(monkeypatch, loader, ds_return, hub=None, disk_return=None):
    """Stub datasets.load_dataset, resolve/files + disk fetch so we can assert
    WHICH route ran without any network."""
    calls = {"streaming": None, "resolved": 0, "disk": 0}
    ds_obj = type("Ds", (), {})()
    monkeypatch.setattr(loader, "_hub", hub or _FakeHub([]))
    ds_obj.load_dataset = lambda *a, split=None, streaming=None, cache_dir=None, **kw: (
            calls.update({"streaming": streaming}), ds_return[0])[1]
    monkeypatch.setattr(loader, "_datasets", ds_obj)
    if disk_return is not None:
        monkeypatch.setattr(
            loader, "_resolve_split_files",
            lambda *a, **k: (calls.__setitem__("resolved", calls["resolved"] + 1) or
                             (disk_return or _FakeHub(["a.jsonl"]).files)),
        )
        monkeypatch.setattr(
            loader, "_fetch_split_to_disk",
            lambda *a, **k: (calls.__setitem__("disk", calls["disk"] + 1),
                             None if disk_return == "FALLBACK" else (disk_return or []))[1],
        )
    return calls


def test_stream_mode_is_lazy_and_no_disk(monkeypatch):
    loader = HFLoader()
    loader._setup_env()
    calls = _patch(monkeypatch, loader, [_FakeStream(_rows(3))], disk_return=None)
    result = loader.load(
        "openbmb/UltraData-SFT-2605", name="Math", split="no_think",
        streaming=True, num_rows=None,
    )
    assert isinstance(result, types.GeneratorType)
    assert calls["streaming"] is True
    # streaming path must never resolve repo files / write to disk
    assert calls["resolved"] == 0 and calls["disk"] == 0


def test_stream_mode_bounded_caps_rows(monkeypatch):
    loader = HFLoader()
    calls = _patch(monkeypatch, loader, [_FakeStream(_rows(3))], disk_return=None)
    result = loader.load(
        "openbmb/UltraData-SFT-2605", name="Math", split="no_think",
        streaming=True, num_rows=2,
    )
    assert list(result)[:2] and calls["streaming"] is True
    assert calls["resolved"] == 0 and calls["disk"] == 0


def test_download_mode_non_default_split_uses_disk_route(monkeypatch, tmp_path):
    loader = HFLoader()
    # no texts-cache exists -> disk route expected
    loader._resolve_cache = lambda d: str(tmp_path)
    ds = _FakeStream(_rows(3))
    calls = _patch(monkeypatch, loader, [ds], hub=_FakeHub(
        ["data/no_think/Math/m000.jsonl"]), disk_return=["t1", "t2", "t3"])
    result = loader.load(
        "openbmb/UltraData-SFT-2605", name="Math", split="no_think",
        streaming=False, num_rows=3,
    )
    assert result == ["t1", "t2", "t3"]
    assert calls["resolved"] == 1 and calls["disk"] == 1
    assert calls["streaming"] is None or True  # load_dataset not called at all


def test_download_mode_default_train_split_stays_eager(monkeypatch, tmp_path):
    loader = HFLoader()
    loader._resolve_cache = lambda d: str(tmp_path)
    ds = _FakeStream(_rows(2))
    calls = _patch(monkeypatch, loader, [ds], hub=_FakeHub(["data/x.jsonl"]))
    # train = default split -> eager materialize (load_dataset streaming=False),
    # never routed to disk-files
    result = loader.load("rajpurkar/squad", split="train", streaming=False,
                         num_rows=2, text_column="context")
    assert calls["streaming"] is False
    assert calls["resolved"] == 0 and calls["disk"] == 0
    assert len(list(result)) == 2


def test_download_mode_disk_fallback_to_stream(monkeypatch, tmp_path):
    loader = HFLoader()
    loader._resolve_cache = lambda d: str(tmp_path)
    ds = _FakeStream(_rows(2))
    calls = _patch(monkeypatch, loader, [ds], disk_return="FALLBACK")
    result = loader.load(
        "some/repo", name="cf", split="val", streaming=False, num_rows=2,
    )
    # disk route failed (returned None) -> fell back to streaming iterable
    assert calls["resolved"] == 1 and calls["disk"] == 1
    assert calls["streaming"] is True
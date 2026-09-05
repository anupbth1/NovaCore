import json
import os

from novacore.dataset.hf_loader import HFLoader
from novacore.logger import Logger

DS = "openbmb/UltraData-SFT-2605"


def _rows(start, n=5):
    out = []
    for i in range(start, start + n):
        out.append({
            "uid": f"Math_no_think_{i:07d}",
            "messages": [
                {"role": "user", "content": f"Question number {i}"},
                {"role": "assistant", "content": f"Answer {i}\nwith a second line"},
            ],
            "source": "UltraData-sft-2605",
            "domain": "Math",
            "think_type": "no_think",
        })
    return out


class _FakeHub:
    def __init__(self, files, cache, prewrite=None):
        self.files = files
        self.cache = cache
        self.prewrite = prewrite or {}

    def list_repo_files(self, repo_id, repo_type="dataset"):
        return list(self.files)

    def hf_hub_download(self, repo_id, filename, repo_type="dataset",
                        cache_dir=None, local_files_only=False, **kwargs):
        target = os.path.join(self.cache, filename.replace("/", "__"))
        if local_files_only and not os.path.isfile(target):
            raise FileNotFoundError(filename)
        if not os.path.isfile(target):
            rows = self.prewrite.get(filename, _rows(1))
            with open(target, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target


def _make_loader(fake_hub):
    loader = HFLoader()
    loader._datasets = object()
    loader._hub = fake_hub
    return loader


def _seed_pool(cache_dir, items):
    d = os.path.join(cache_dir, "pool", "openbmb_UltraData-SFT-2605__Math__no_think")
    os.makedirs(d, exist_ok=True)
    for item in items:
        with open(os.path.join(d, "pool.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    meta = {"dataset": DS, "config": "Math", "split": "no_think",
            "uid_key": "uid", "rows": len(items)}
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return d


def _pool_rows(cache_dir):
    rows = []
    d = os.path.join(cache_dir, "pool", "openbmb_UltraData-SFT-2605__Math__no_think")
    if not os.path.isfile(os.path.join(d, "pool.jsonl")):
        return rows
    with open(os.path.join(d, "pool.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def test_pool_reuse_when_add_off_skips_fetch(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _seed_pool(cache_dir, [{"uid": "u:x", "text": "t1"}, {"uid": "u:y", "text": "t2"}])
    fake = _FakeHub(["data/no_think/Math/m000.jsonl"], str(tmp_path))
    loader = _make_loader(fake)
    result = loader._fetch_split_to_disk(
        DS, "no_think", "Math", ["data/no_think/Math/m000.jsonl"], cache_dir,
        num_rows=1, random_sample=False, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(),
        add_datasets=False,
    )
    assert list(result) == ["t1", "t2"]
    written = [f for f in os.listdir(str(tmp_path)) if f.endswith(".jsonl")]
    assert written == []


def test_consume_full_shard_no_discard(tmp_path):
    cache_dir = str(tmp_path / "cache")
    fake = _FakeHub(["data/no_think/Math/m000.jsonl"], str(tmp_path))
    loader = _make_loader(fake)
    files = ["data/no_think/Math/m000.jsonl"]
    r1 = loader._fetch_split_to_disk(
        DS, "no_think", "Math", files, cache_dir,
        num_rows=2, random_sample=False, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(), add_datasets=True,
    )
    r2 = loader._fetch_split_to_disk(
        DS, "no_think", "Math", files, cache_dir,
        num_rows=2, random_sample=False, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(), add_datasets=True,
    )
    rows = _pool_rows(cache_dir)
    uids = [r["uid"] for r in rows]
    assert len(r1) == 5 and len(r2) == 5
    assert len(uids) == len(set(uids))


def test_random_add_all_unique_consumed(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _seed_pool(cache_dir, [
        {"uid": "u:Math_no_think_0000001", "text": "seed1"},
        {"uid": "u:Math_no_think_0000002", "text": "seed2"},
    ])
    fake = _FakeHub(["data/no_think/Math/m000.jsonl"], str(tmp_path))
    loader = _make_loader(fake)
    result = loader._fetch_split_to_disk(
        DS, "no_think", "Math", ["data/no_think/Math/m000.jsonl"], cache_dir,
        num_rows=2, random_sample=True, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(), add_datasets=True,
    )
    rows = _pool_rows(cache_dir)
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids))
    assert "u:Math_no_think_0000001" in uids
    assert "u:Math_no_think_0000002" in uids
    new = set(uids) - {"u:Math_no_think_0000001", "u:Math_no_think_0000002"}
    assert len(new) == 3
    assert all(u.startswith("u:Math_no_think_") for u in new)
    assert len(result) == 5


def test_absorb_local_only_skips_missing(tmp_path):
    cache_dir = str(tmp_path / "cache")
    present_rows = _rows(101, 3)
    target = os.path.join(str(tmp_path), "data__no_think__Math__m001.jsonl")
    with open(target, "w", encoding="utf-8") as fh:
        for row in present_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fake = _FakeHub(
        ["data/no_think/Math/m001.jsonl", "data/no_think/Math/m002.jsonl"],
        str(tmp_path),
    )
    loader = _make_loader(fake)
    result = loader._fetch_split_to_disk(
        DS, "no_think", "Math",
        ["data/no_think/Math/m001.jsonl", "data/no_think/Math/m002.jsonl"],
        cache_dir, num_rows=None, random_sample=False, fields=None,
        texts_cache=None, verbose=False, label="ds", log=Logger(),
        add_datasets=True, local_only=True,
    )
    assert result is not None
    assert len(result) == 3
    assert all("Question number" in t for t in result)
    rows = _pool_rows(cache_dir)
    assert len(rows) == 3
    assert not os.path.isfile(os.path.join(str(tmp_path), "data__no_think__Math__m002.jsonl"))


def test_load_download_train_appends_pool_and_reuses(monkeypatch, tmp_path):
    class Fs:
        def to_iterable_dataset(self):
            return self

        def __iter__(self):
            yield from _rows(1, 3)

        def __len__(self):
            return 3

        @property
        def features(self):
            return {}

    loader = _make_loader(_FakeHub([], str(tmp_path)))
    cache_dir = str(tmp_path / "cache")
    loader._resolve_cache = lambda d: cache_dir
    ds_obj = type("Ds", (), {})()
    ds_obj.load_dataset = lambda *a, split=None, streaming=None, cache_dir=None, **kw: Fs()
    monkeypatch.setattr(loader, "_datasets", ds_obj)
    result = loader.load("rajpurkar/squad", split="train", streaming=False,
                         num_rows=2, text_column="context", add_datasets=True)
    texts = list(result)
    assert len(texts) == 2
    d = os.path.join(cache_dir, "pool", "rajpurkar_squad__default__train")
    assert os.path.isfile(os.path.join(d, "pool.jsonl"))
    rows = [json.loads(ln) for ln in open(os.path.join(d, "pool.jsonl"), encoding="utf-8") if ln.strip()]
    assert len(rows) == 2

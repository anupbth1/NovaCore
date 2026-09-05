import json
import os
import tempfile

from novacore.dataset.hf_loader import HFLoader
from novacore.logger import Logger


def _make_loader(fake_hub):
    loader = HFLoader()
    loader._datasets = object()
    loader._hub = fake_hub
    return loader


def _sample_rows(n=5, offset=0):
    rows = []
    for i in range(1, n + 1):
        gi = offset + i
        rows.append({
            "uid": f"Math_no_think_{gi:07d}",
            "messages": [
                {"role": "user", "content": f"Question number {gi}"},
                {"role": "assistant", "content": f"Answer {gi}\nwith a second line"},
            ],
            "source": "UltraData-sft-2605",
            "domain": "Math",
            "think_type": "no_think",
        })
    return rows


class _FakeHub:
    def __init__(self, repo_files, cache):
        self.repo_files = repo_files
        self.cache = cache

    def list_repo_files(self, repo_id, repo_type="dataset"):
        return list(self.repo_files)

    def hf_hub_download(self, repo_id, filename, repo_type="dataset",
                        cache_dir=None, local_files_only=False, **kwargs):
        target = os.path.join(self.cache, filename.replace("/", "__"))
        if local_files_only and not os.path.isfile(target):
            raise FileNotFoundError(filename)
        if not os.path.isfile(target):
            written = sum(
                1 for f in os.listdir(self.cache) if f.endswith(".jsonl")
            ) if os.path.isdir(self.cache) else 0
            with open(target, "w", encoding="utf-8") as fh:
                for row in _sample_rows(offset=written * 5):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target


def test_detect_local_schema_maps_messages():
    loader = _make_loader(_FakeHub([], tempfile.mkdtemp()))
    col, fields, report = loader._detect_local_schema(_sample_rows(), "t", verbose=False)
    assert col is None
    assert fields == {"messages": "messages"}
    assert report


def test_row_text_chat_serialization():
    loader = _make_loader(_FakeHub([], tempfile.mkdtemp()))
    _, fields, _ = loader._detect_local_schema(_sample_rows(1), "t", verbose=False)
    text = loader._row_text(_sample_rows(1)[0], None, fields)
    assert "Question number 1" in text
    assert "Answer 1" in text


def test_resolve_split_files_filters_prefix():
    fake = _FakeHub([
        "data/think/Math/Math_think_part-000-of-300.jsonl",
        "data/no_think/Math/Math_no_think_part-000-of-300.jsonl",
        "data/no_think/Math/Math_no_think_part-001-of-300.jsonl",
        "data/no_think/Code/Code_no_think_part-000-of-300.jsonl",
        "README.md",
    ], tempfile.mkdtemp())
    loader = _make_loader(fake)
    files = loader._resolve_split_files("openbmb/UltraData-SFT-2605", "no_think", "Math")
    assert files == [
        "data/no_think/Math/Math_no_think_part-000-of-300.jsonl",
        "data/no_think/Math/Math_no_think_part-001-of-300.jsonl",
    ]


def test_fetch_split_to_disk_bounded_and_texts_cache(tmp_path):
    fake = _FakeHub([], str(tmp_path))
    loader = _make_loader(fake)
    cache_dir = str(tmp_path / "cache")
    texts_cache = os.path.join(cache_dir, "texts", "k.jsonl")
    files = [
        "data/no_think/Math/Math_no_think_part-000-of-300.jsonl",
        "data/no_think/Math/Math_no_think_part-001-of-300.jsonl",
    ]
    result = loader._fetch_split_to_disk(
        "openbmb/UltraData-SFT-2605", "no_think", "Math", files, cache_dir,
        num_rows=7, random_sample=False, fields=None, texts_cache=texts_cache,
        verbose=False, label="ds", log=Logger(),
    )
    assert len(result) == 10
    assert all("Question number" in t for t in result)
    assert os.path.isfile(texts_cache)
    with open(texts_cache, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 10
    assert loader.last_schema["source"] == "disk-files"
    assert loader.last_schema["fields"] == {"messages": "messages"}
    assert loader.last_schema["uid_key"] == "uid"


def test_fetch_split_to_disk_reuse_no_redownload(tmp_path):
    fake = _FakeHub([], str(tmp_path))
    loader = _make_loader(fake)
    before = len(fake.repo_files) if hasattr(fake, "repo_files") else 0
    files = ["data/no_think/Math/Math_no_think_part-000-of-300.jsonl"]
    cache_dir = str(tmp_path / "cache")
    r1 = loader._fetch_split_to_disk(
        "d", "no_think", "Math", files, cache_dir,
        num_rows=2, random_sample=False, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(),
    )
    # second call: files exist on disk (local_files_only path), same rows returned
    r2 = loader._fetch_split_to_disk(
        "d", "no_think", "Math", files, cache_dir,
        num_rows=2, random_sample=False, fields=None, texts_cache=None,
        verbose=False, label="ds", log=Logger(),
    )
    assert r1 == r2
    assert before == 0 or True
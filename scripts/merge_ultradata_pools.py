"""Merge full raw UltraData shards (from the HF hub cache on disk) into the
corresponding accumulated pools so training sees ALL downloaded rows.

Only the 3 under-full pools are merged (Math no_think, Multi-lang-Knowledge
no_think, Knowledge think). Rows already present in the pool (by uid) are
skipped - never duplicated. Zero network: reads only what is already on disk.

Usage (from C:\\project6\\NovaCore):
    python scripts/merge_ultradata_pools.py
"""
import json
import os
import sys

PYTHONPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYTHONPATH)

from novacore.dataset.hf_loader import HFLoader
from novacore.logger import Logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "hf_cache")
SNAP = os.path.join(
    CACHE, "hub", "datasets--openbmb--UltraData-SFT-2605", "snapshots",
    "affda6aca75e7cff78e73f93ad08d4c3b01f097c", "data",
)

# (dataset, config, split, hub_subdir, pool_dir_name)
TARGETS = [
    ("openbmb/UltraData-SFT-2605", "Math", "no_think",
     os.path.join(SNAP, "no_think", "Math"),
     "openbmb_UltraData-SFT-2605__Math__no_think"),
    ("openbmb/UltraData-SFT-2605", "Multi-lang-Knowledge", "no_think",
     os.path.join(SNAP, "no_think", "Multi-lang-Knowledge"),
     "openbmb_UltraData-SFT-2605__Multi-lang-Knowledge__no_think"),
    ("openbmb/UltraData-SFT-2605", "Knowledge", "think",
     os.path.join(SNAP, "think", "Knowledge"),
     "openbmb_UltraData-SFT-2605__Knowledge__think"),
]


def main():
    log = Logger()
    loader = HFLoader()

    for ds, config, split, hub_dir, pool_name in TARGETS:
        log.header(f" MERGE {ds} [{config}] / split '{split}' ")
        files = sorted(
            os.path.join(hub_dir, f)
            for f in os.listdir(hub_dir)
            if f.endswith(".jsonl")
        )
        log.info(f"hub raw shards found: {len(files)}")

        pool = loader._load_pool(CACHE, ds, config, split)
        if pool is None:
            pdir = loader._pool_dir(CACHE, ds, config, split)
            pool = {
                "dir": pdir, "rows": [], "uids": set(),
                "meta": {"dataset": ds, "config": config, "split": split},
                "rows_path": os.path.join(pdir, "pool.jsonl"),
                "meta_path": os.path.join(pdir, "manifest.json"),
            }
        before = len(pool["uids"])
        log.info(f"pool rows before: {before}")

        skipped_dup = 0
        skipped_invalid = 0
        added = 0
        col = None
        merged_fields = None
        used_uid = None
        detected = False

        for fi, fpath in enumerate(files, 1):
            file_rows = []
            try:
                fh = open(fpath, encoding="utf-8")
            except OSError as e:
                log.warn(f"  cannot open {os.path.basename(fpath)} ({e}) - skip")
                continue
            with fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue

                    if not detected:
                        first_rows = []
                        try:
                            with open(fpath, encoding="utf-8") as fh2:
                                for ln2 in fh2:
                                    ln2 = ln2.strip()
                                    if not ln2:
                                        continue
                                    try:
                                        r2 = json.loads(ln2)
                                    except ValueError:
                                        continue
                                    if isinstance(r2, dict):
                                        first_rows.append(r2)
                                        if len(first_rows) >= 5:
                                            break
                        except OSError:
                            pass
                        try:
                            col, use_fields, _ = loader._detect_local_schema(
                                first_rows, ds, False)
                        except Exception:
                            col, use_fields = None, None
                        merged_fields = use_fields or {}
                        cols = list(first_rows[0].keys()) if first_rows else []
                        used_uid = loader._pool_uid_key(None, cols)
                        detected = True

                    text = loader._row_text(row, col, merged_fields)
                    if not text:
                        skipped_invalid += 1
                        continue
                    uid = loader._uid_for(row, used_uid, text)
                    if uid in pool["uids"]:
                        skipped_dup += 1
                        continue
                    pool["uids"].add(uid)
                    file_rows.append({"uid": uid, "text": text})

            if file_rows:
                loader._append_pool(pool, file_rows, fields=merged_fields, col=col)
                added += len(file_rows)
                if fi % 10 == 0:
                    log.info(f"  [{fi}/{len(files)}] +{added} new | +{len(file_rows)} this file | pool now {len(pool['rows'])}")

        log.ok(f"merged: +{added} rows  (dedup-skipped {skipped_dup}, invalid {skipped_invalid})")
        log.ok(f"FINAL pool rows: {len(pool['rows'])}  (was {before})")

    log.ok("ALL 3 ULTRADATA POOLS MERGED")


if __name__ == "__main__":
    main()

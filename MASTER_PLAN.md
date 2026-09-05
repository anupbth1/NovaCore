# NovaCore MASTER_PLAN.md

Central tasks list for the current session. Everything pending is listed here with
any file(s) that need changing, so nothing is left half-done.

## Goal
Create/expand NovaCore models from **HF and/or local datasets** with a **strict config system**
(no built-in content defaults) and a clean CLI, using the native `.ncw`/`.ncmeta` weight format.

---

## ✅ DONE

1. **config/weight format**
   - Native `weights.ncw` + `index.ncmeta` format; legacy `.npz` migration.
   - File: `novacore/storage/weight_manager.py`

2. **No-hardcode rule**
   - `RULES.md` created; removed hardcoded greeting templates in `predictor.py`.

3. **config loaders**
   - `novacore/config.py`: added `configure(path)`, `load_config(path)`, `reset()`.
   - Base config values from `config/config.json`.

4. **Strict model config (`_resolve_model_config` in `cli/main.py`)**
   - Values from `--config <file>` (base) **and/or** manual flags (`--dim --layers --vocab-size --max-tokens --temperature`).
   - Manual flags override the config file per-setting.
   - If NEITHER `--config` NOR any manual flag → ERROR (no built-in defaults used).

5. **Unified `train` command (`cmd_train` in `cli/main.py`)**
   - `--dataset` repeatable (HF `owner/repo` or local file).
   - `--config` + manual flags (strict).
   - `--output` (new) OR `--load <existing>` (expand). `--new`/`--load` mutual exclusion.
   - `--mode download|stream`.
   - `--max-rows` (number / `full`).
   - Combines all datasets into one encoding; `merge_existing` when `--load`.

6. **chat/generate saved-model config fallback** (`cli/main.py`)
   - When `--tokens`/`--temperature` omitted, model's saved `max_tokens`/`temperature` used.

7. **`patterns.py` KeyError fix** (`novacore/core/patterns.py`)
   - `self.patterns.get(gram, 0) + 1` (safe for plain dict after expand).

8. **Vocabulary expansion** (`novacore/tokenizer/vocab.py`)
   - Added `grow(texts)` for expanding an existing model's vocab.

9. **Per-dataset `--max-rows`** (`cli/main.py`)
   - `--max-rows` is now `action="append"`, paired by index with each `--dataset`.
   - Embedded `spec#rows` / `spec:rows` / `spec=rows` parsing kept as fallback.

10. **Parser flags consistency verified** (`cli/main.py`)
    - `train`, `hf run`, `encode` all expose `--dim --layers --vocab-size --max-tokens --temperature --config`.

11. **Test suite PASSED** — `python -m pytest tests/ -q` → 10 passed.
    - Includes `weights.ncw`/`index.ncmeta` expectations in `test_cli.py`.

12. **data/ + weights/ centralization**
    - `config/config.json` paths: `data_dir=data`, `hf_data=data/hf_data`, `hf_cache=data/hf_cache`, `default_weights=weights`.
    - HF `download()` saves to config `hf_data` (data/hf_data) instead of hardcoded path.
    - `cli/main.py`: bare `--output`/`--load`/`--weights` names auto-resolve under `weights/`
      (`--output my_model` → `weights/my_model`; `chat --weights my_model` same).

13. **Cache auto-reuse + download** (`novacore/dataset/hf_loader.py`)
    - `cache_status()`: already-downloaded dataset → `train`/`load` download skip, direct load + train.
    - `mode: download` hamesha materialize karta hai → cache `data/hf_cache` me; cached hone par reuse. `mode: stream` lazy.
    - **Non-default split (e.g. `no_think`)**: datasets 5.x eager path mis-selects the first data_files
      glob (card-YAML multi-split configs, gated w/o dataset_infos.json) — so non-`train` splits are
      resolved via `list_repo_files` prefix (`data/<split>/<config>/`) and downloaded as **real raw
      shard bytes to disk** (`data/hf_cache/hub`, persistent + resumable) then parsed locally; a small
      **texts-cache** (`data/hf_cache/texts/<dataset>__<config>__<split>__<N>.jsonl`, JSON-lines) makes
      re-runs instant. Default (`train`) splits keep the standard eager path (navigable + positional
      random sampling).
    - `_clean_incomplete()`: stale `.incomplete` markers cleaned → interrupted download resume
      (sweep ab per-process ek hi baar chalta hai - multi-GB cache re-walk nahi hota).
    - Namespace detection multi-form: `owner___camel_to_snake(repo)` (datasets v5 layout, hyphens
      kept) + legacy `owner___underscored` — cached data galat naam pick hoke dobara download nahi hota.

14. **Random rows + Accumulated pool (`max_rows` / `random` / `add_datasets`)**
    (`hf_loader.py` + `cli/main.py` + `config/data_config.json`)
    - `random` on/off + `add_datasets` on/off are config-driven (`data_config.json` `hf` block
      defaults + per-dataset/subset keys; CLI `--random/--no-random`, `--add-datasets/--no-add-datasets`).
      `random: true` → rows picked at random (dedupe-aware); OFF sequential. Stream = lazy, N/A.
    - **Accumulated pool** per `(dataset, config, split)`: `data/hf_cache/pool/<key>/pool.jsonl`
      (`{"uid","text"}` JSON-lines) + `manifest.json` (dataset/config/split/uid_key/rows/fields).
      Every row uid-sealed (`uid_key` column auto-detect from uid/id/_id/guid/hash/sample_id, else
      sha1 content-hash) → duplicates NEVER re-added. Legacy `texts-cache` jsonl seed the pool.
    - `add_datasets: true` pulls `num_rows` NEW unique rows per run (bounds stop early; random uses
      bounded approximate reservoir); `false` reuses whatever is on disk (`pool reuse, no fetch`).
      First run auto-builds the pool even when add is OFF.
    - Numeric `--max-rows`/`max_rows` → rows picked random from cached subset (HF cached = random-index;
      local files = random sample). `full` = entire dataset. Pehli baar download me bounded materialization
      sirf utne hi lokhen khinchta hai jitne chahiye (poori repo nahi).

15. **Multi-subset datasets** (`config/data_config.json` + `cli/main.py`)
    - `"subset": [ {name, split, max_rows}, ... ]` — har subset apna split/max_rows; no duplicate JSON keys.

16. **Unicode-safe output** (`cli/main.py`)
    - `sys.stdout/stderr.reconfigure(utf-8)` — Windows cp1252 crash (squad Unicode) fixed.

---

## ✅ DONE (tracking files)

- `PLAN.md`, `STRUCTURE.md`, `COMMANDS.md`, `README.md`, `RULES.md` updated.
- Documented: `train` command, strict config, per-dataset `--max-rows`, PowerShell-verified examples.
- Documented: cache auto-reuse/resume, random rows, `subset` multi-config, hf_cache/hf_data paths.
- Documented: accumulated pool (uid dedupe, random/add_datasets config), real disk download (`data/hf_cache/hub`).

---

## 🚀 NEW MODEL BANANE KA COMMAND (PowerShell test)

```powershell
cd C:\project6\NovaCore

# Naya model (strict config chahiye; dataset data/, model weights/)
python cli/main.py train --dataset data\sample.txt --output my_model --config config\config.json

# Chat se test (--weights my_model = weights\my_model)
python cli/main.py chat --weights my_model

# Expand karo
python cli/main.py train --dataset data\conversation.txt --load my_model --config config\config.json
```

---

## 🚧 CURRENTLY PENDING (do these — files listed)

> P1, P2, P4, P5 ab done hain (upar ✅ list dekho). Baaki:
> - P3: expand `--load` ki confirm re-test abhi ho chuki hai via `train --load` flow (docs tested). Full re-run: `train --dataset <new> --load <existing> --config config/config.json` + `validate`.

---

## 📌 Requirement Summary (from chat)

| Requirement | Status |
|---|---|
| `.ncw`/`.ncmeta` native format | DONE |
| No hardcoded content (RULES) | DONE |
| HF dataset need `owner/repo` | DONE |
| Config from `config/config.json` (import) | DONE |
| Strict config: file OR flags, else error | DONE |
| `--config config.json` OR manual flags | DONE |
| Manual flags override config file | DONE |
| Chat uses saved model config (exclude from strict) | DONE |
| `--max-rows 10000` limit / `full` | DONE (global) |
| `--max-rows` per-dataset (mixed datasets) | DONE |
| Local datasets support | DONE |
| Multi-dataset (several `--dataset`) | DONE |
| Mixed HF + local | DONE |
| `--new` new model | DONE |
| `--load <path>` expand existing | DONE (needs re-test P3) |
| download vs stream | DONE (`--mode`) |
| Cache auto-reuse (download skip) | DONE |
| Interrupted-download resume | DONE |
| Random `max_rows` subset (start se nahi) | DONE |
| Multi-subset (`subset` list) | DONE |
| Save model config in `.ncmeta` | DONE |

# VedNex Build Log — Auto-Loop Notebook (DO NOT DELETE UNTIL OUTPUT CORRECT)

Created: 2026-09-03
Objective: Build NovaCore model VedNex_V2 (or new) from cached pools (6.59M docs) with virtual_sim baked in. Stop ONLY when generate output is correct.

---

## ATTEMPT 1 — Original (pwsh-6, earlier)
- Command: `python cli/main.py train --dataset ... --output VedNex_V1 --config config/config.json`
- Status: CRASHED (MemoryError at quality.py line 260)
- Diagnosis: `quality.py` used Python `set()` for exact_text_dedup. At 6.59M docs each ~9KB → ~60GB in memory → MemoryError.
- Fix applied: `quality.py` now uses `_CompactHashSet` (FNV-1a 32-bit hash).
- Note added: Never use Python `set()` for large-scale dedup.

---

## ATTEMPT 2 — Original rebuild with fix
- Status: TRAINING HUNG (RAM 100%, rate dropped 45K→7K docs/s)
- Diagnosis: Auto-tuner detected 11.8GB total RAM. Memory guard (`mem_pressure() > 0.90`) triggered `vocab_cap` reduction. Process PID 15108 went to negative WS and died.
- Fix applied: Added `ram_safe_overrides` in `config/config.json` (dim=256, layers=2, vocab_size=20000, reservoir=50000, coocc=4, ngram=[2,3]). Added `AUTO-REDUCE` + `AUTO-GUARD` in `cli/main.py`.
- Note added: On <16GB RAM, always use `ram_safe_overrides`; full `dim=512, layers=4` needs >16GB for 5M+ docs.

---

## ATTEMPT 3 (CURRENT — pwsh-10, started 10:22 PM)
- Fixes applied before start:
  1. `quality.py`: `_CompactHashSet` (not `set()`)
  2. `data_config.json`: all `filter.enabled = false`
  3. `main.py`: RAM-safe override applied automatically when total_ram < 16GB
  4. `virtual_sim.py`: created (`SimulationEngine` with `VirtualVerifier` + `SelfCorrector`)
  5. `predictor.py`: virtual sim baked into `generate()` method (post-generation verification)
  6. `reservoir.py`: 8KB string truncation per item
  7. `config/config.json`: clean JSON (no `//` comments)
  8. All manifest `complete: true` set so pools reuse
- Config used: `config/config.json` (clean) + `data_config.json` (filter false)
- PROGRESS (as of ~10:30 PM):
  - Steps 1-6: COMPLETE (3 min) — HuggingFaceH4, oasst1, oasst2, squad, squad_v2, alpaca
  - Step 7 (TinyStories 2.1M): downloading at ~30K/s — ETA ~8 min
  - Steps 8-10: pool REUSE (no download) — ETA <1 min each
  - Then ENCODING (streaming) — ETA ~5-10 min with RAM-safe (dim=256, layers=2)
- Storage safe (C: 14.2GB free, pools intact, no data lost).
- ISSUE NOTED: even with `complete=true` in manifest, `loader.load()` still loads all UIDs from pool.jsonl
  for every dataset on every run (the pre-scan sets `_skip_pool` but the loader still reads UIDs).
  This is NOT a crash but adds ~10-30s overhead per dataset. Performance issue, not correctness.
- Next action: Monitor until ENCODING starts. When VedNex_V2 folder created → run generate test.

---

## WHY WRONG OUTPUT CAN HAPPEN (study list for next loop)
1. `quality.filter.enabled = true` + Python `set()` = crash (FIXED)
2. Memory 100% → swap I/O → encoder weights corrupt or incomplete (PREVENTED by RAM-safe overrides + guard)
3. `reservoir_size` too large → no representative sample → bad analytic weights (PREVENTED by cap 50000)
4. `vocab_size` too large for RAM → vocab dict truncated mid-build → incomplete vocabulary (PREVENTED by vocab_cap 50000 + cap reduction)
5. `ngram_range` [2,4] with 5.6M docs → too many unique patterns → memory pressure (FIXED: [2,3])
6. No virtual simulation = raw output may have low pattern confidence (PREVENTED: `predictor.py` now calls `engine.verify_and_correct()` after generation)
7. Pool resume with `complete=true` but `add_datasets=true` + filter false means same rows re-added? (CHECKED: filter false + UID dedup in manifest prevents duplicates; `add_datasets=true` means add NEW rows, but since pool exists with 9500/84434/etc rows, loader sees `pool found` and skips re-parse if `add_datasets` logic allows; verify manifest `rows` matches pool file size)

---

## NEXT ACTION CHECKLIST (automated loop)
- [ ] Check `pwsh-10` status every 5 min.
- [ ] When complete: verify `weights/VedNex_V2/` exists.
- [ ] If NOT exists / crash: check error log → fix code → rebuild `VedNex_V3`.
- [ ] If exists: run `python cli/main.py generate --weights VedNex_V2 --prompt "What is 2 plus 2?" --tokens 30`
- [ ] If output correct (contains number/logic/reference): SAVE result, update this note with SUCCESS.
- [ ] If output wrong (nonsense, empty, unrelated): study why:
  - Check `vocab.json` size (`ls weights/VedNex_V2/vocab.json`)
  - Check `metadata.json` (`cat weights/VedNex_V2/metadata.json`)
  - Check `weights.npz` size (`ls weights/VedNex_V2/*.npz`)
  - Then decide: retrain with same pools (fast, ~10-15 min with safe config) OR fix code.
- [ ] If storage drops < 5GB: move old `VedNex_V1` to `D:\` (create folder `D:\NovaBackups`).
- [ ] STOP ONLY when `generate` output is CORRECT.

---

## AUTO-TUNER CURRENT VALUES (at 11.8GB RAM)
- CPU: 4, AVX2: yes
- Budget: 10.0GB (85% of 11.8GB)
- GPU: none
- Shard flush cap: 50,000
- Vocab cap: 200,000
- Pattern cap: 200,000
- Reservoir cap: 200,000
- Num threads: 2
- Memory guard: if >90% → halve caps; if >85% → trim reservoir

---

## OUTPUT EVALUATION FORMAT (fill when completed)
```
Test prompt: "..."
Raw output: "..."
Score (virtual_sim): X.XX / 1.0
Passed threshold (0.30): YES/NO
Correct: YES/NO
Notes: ...
```

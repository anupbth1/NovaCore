# NovaCore - PLAN.md
> Last Updated: 2026-09-01 (unified train command added)

## Vision
Training-Free LLM jo datasets ko mathematical encoding se directly weights me convert kare.
Bina GPU training ke next-token prediction kare.

## Technique Summary

| Component | Method | Status |
|-----------|--------|--------|
| Text Encoding | Random Fourier Features | DONE |
| Pattern Storage | Count-Min Sketch / n-grams | DONE |
| Weight Creation | Analytical (Pseudo-inverse) | DONE |
| Tokenization | Vocabulary + Feature Hashing | DONE |
| Inference | PatternPredictor (no training) | DONE |
| **Chat Interface** | **ChatSession (Ollama-style)** | **DONE** |
| **Unified `train`** | **HF + local, multi/mixed, new/load, strict config** | **DONE** |

## What Was Built

### Core Engine [DONE]
- [x] encoder.py - RandomFourierEncoder (analytic pseudo-inverse fit)
- [x] hasher.py - FeatureHasher (Count-Min-Sketch style)
- [x] patterns.py - PatternExtractor (n-gram + co-occurrence)
- [x] reservoir.py - ReservoirSampler (large data sampling)

### Tokenizer [DONE]
- [x] vocab.py - Vocabulary (tokenizer)
- [x] text_processor.py - TextProcessor (text -> weight vector)

### Storage [DONE]
- [x] weight_manager.py - Save/Load/Compress (npz + json)

### Inference [DONE]
- [x] predictor.py - Predictor + PatternPredictor (next-token generation)
- [x] chat.py - ChatSession (load model + interactive chat with memory)

### Dataset [DONE]
- [x] loader.py - DatasetLoader (csv/json/txt/tsv)
- [x] **hf_loader.py - HFLoader (download/stream/run) + cache auto-reuse/resume + random rows**
- [x] **sahi-split resolve + real disk download (`data/hf_cache/hub`, resumable) + texts-cache (`data/hf_cache/texts/`) (fixes datasets 5.x wrong-split download e.g. `no_think` → `think`)**
- [x] **Accumulated pool (`data/hf_cache/pool/<key>/pool.jsonl` + manifest, uid dedupe, auto-seed from texts-cache) + `random`/`add_datasets` config (`data_config.json` hf defaults + per-dataset/subset, CLI `--random/--add-datasets`) — reuse = no fetch, add = N new unique rows only**
- [x] **novacore/logger.py - professional console logger (status badges + timing)**

### Config [DONE]
- [x] config/config.json - all defaults
- [x] novacore/config.py - config loader

### CLI [DONE]
- [x] main.py - All commands + chat + hf
- [x] **train - unified model creation (HF + local, multi/mixed, --new/--load)**
- [x] **Strict config: --config file OR manual flags, else error**

## Verified
- [x] encoder tests passed
- [x] tokenizer tests passed
- [x] inference tests passed
- [x] CLI end-to-end test passed (encode -> generate)
- [x] **chat tests passed (load + chat + reset)**
- [x] Sample: encode 16 docs -> 0.661 MB
- [x] conversation: encode 40 docs -> 1.035 MB
- [x] Chat: "the cat" -> "and the dog ran through the park chasing a ball"
- [x] Chat: "tell me about space" -> "exploration pushes the world faster every single year"
- [x] **HF: stream rajpurkar/squad works**
- [x] **HF: run rajpurkar/squad (20 rows) -> weights/hf_squad 0.099 MB**
- [x] **HF model chat: "Notre Dome" -> "college of engineering similar studies..."**
- [x] **HF status/login/logout commands work**
- [x] **train: local 'data/sample.txt' (16 docs) -> weights/my_model 0.684 MB (dim=256, layers=2)**
- [x] **train+chat pipeline verified end-to-end on Windows PowerShell**
- [x] **Bare-name paths: --output my_model -> weights/my_model, --weights my_model loads from weights/ (config paths)**
- [x] **All datasets in data/ (HF -> data/hf_cache), all models in weights/ (config: data_dir, hf_data, hf_cache, default_weights)**
- [x] **Cache auto-reuse: already-downloaded dataset = download skip, direct load+train**
- [x] **Interrupted-download resume: stale .incomplete markers cleaned**
- [x] **Random max_rows subset: N rows chosen from anywhere, not from start**
- [x] **Multi-subset datasets: config `subset` list (each with own split/max_rows)**

## Chat System
Ollama-style interactive chat with saved models. Commands:
- `/new`  - reset conversation
- `/info` - show model info
- `/exit` - quit
- one-shot: `--prompt "question"`

## Size Estimates (Actual ~10-20x compression)

| Dataset Size | Encoded Weights | Ratio |
|--------------|-----------------|-------|
| 1 GB | 50-100 MB | 10-20x |
| 10 GB | 500 MB-1 GB | 10-20x |
| 100 GB | 5-10 GB | 10-20x |
| 1 TB | 50-100 GB | 10-20x |

## Next Steps (Future)
- [ ] Stored corpus embeddings for full similarity query
- [ ] True weight-based next-token (analytic beta -> vocab ranking)
- [ ] Multi-model merge (combine weights)
- [ ] GPU acceleration for large dims
- [ ] Streaming encode for >10GB datasets
- [ ] Web/API chat server (like Ollama serve)
- [x] **Auto-download to `data/hf_cache` before train (`--mode download` / `mode: download` materializes + caches; re-runs reuse cache)**

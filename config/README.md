# NovaCore Dataset Configs

This folder holds dataset configurations for `novacore train`.

## Files

| File | Role | Use case |
|---|---|---|
| `data_config.json` | **Universal config** — global defaults + dataset list | Read by every training run |
| `data_config.<name>.json` | **Per-dataset override** | Tweak one dataset without touching the universal file |
| `config.json` | Model config (dim, layers, vocab, etc.) — separate from data | Read by every training run |
| `config-test.json` | Test model config | For testing |

## How overrides work

Multiple config files can be passed:

```
python cli/main.py train \
  --dataset config\data_config.json \
  --dataset config\data_config.rajpurkar_squad.json \
  --output mymodel
```

`hf` blocks merge (later wins), `datasets` blocks merge (later wins), subset entries override by name.

## Priority chain (high → low)

1. **CLI flag** (`--max-rows`, `--mode`, `--random`, `--add-datasets`, `--absorb`)
2. **Per-dataset config** (later config files override earlier)
3. **`hf` global defaults** in `data_config.json`
4. Hardcoded fallback (e.g. `mode` missing → ERROR)

## Universal config (data_config.json)

The universal config defines:

- **`hf`** — global defaults applied to every dataset:
  - `mode`: `download` or `stream`
  - `max_rows`: `full`, a number, or absent
  - `random`: shuffle order
  - `add_datasets`: append new unique rows to the pool
  - `absorb`: zero-network reuse of shards already on disk
  - `uid_key`: column used as a stable per-row identity
  - `default_split`: split used when one isn't specified
- **`datasets`** — list of datasets to train on:
  - `openbmb/UltraData-SFT-2605` (multi-config: Math, Multi-lang-Knowledge, Knowledge)
  - `HuggingFaceH4/no_robots`, `OpenAssistant/oasst1/2`, `rajpurkar/squad[_v2]`,
    `tatsu-lab/alpaca`, `roneneldan/TinyStories`

## Per-dataset override (data_config.<name>.json)

Create a per-dataset file when you need different settings for **one** dataset.
The naming convention is `data_config.<owner>_<repo>.json` (slash becomes underscore).

```json
{
  "datasets": {
    "rajpurkar/squad": {
      "max_rows": 50000,
      "add_datasets": false
    }
  }
}
```

Only the fields you set here override the universal config. Everything else
(mode, random, absorb, etc.) inherits from `data_config.json` → `hf`.

---

## Model Config (config.json)

`config.json` controls model architecture and internal components.

### Internal Components (Phase 3+)

#### Neural Engine
```json
"neural_engine": {
  "enabled": true,
  "input_dim": 64,
  "hidden_dim": 32,
  "output_dim": 16,
  "activation_cache_size": 100,
  "use_fast_sigmoid": true,
  "enable_caching": true
}
```

#### Virtual Simulation
```json
"virtual_simulation": {
  "enabled": true,
  "quality_threshold": 0.4,
  "enable_caching": true,
  "cache_size": 1000,
  "scoring_weights": {
    "length": 0.4,
    "relevance": 0.4,
    "coherence": 0.2
  }
}
```

#### Python Terminal
```json
"python_terminal": {
  "enabled": true,
  "safe_mode": true,
  "max_execution_time_ms": 5000,
  "allowed_operations": ["print", "math", "variables", "loops", "conditionals"]
}
```

#### Verification Engine
```json
"verification_engine": {
  "enabled": true,
  "max_retries": 2,
  "min_score": 0.3,
  "enable_caching": true,
  "reservoir_search_limit": 200,
  "enable_fast_retry": true
}
```

#### Chat
```json
"chat": {
  "max_history": 10,
  "context_tokens": 200,
  "generation_history_tokens": 10,
  "use_model_brain": true
}
```

#### RAM-Safe Overrides (Auto-applied <16GB RAM)
```json
"ram_safe_overrides": {
  "dim": 256,
  "layers": 2,
  "vocab_size": 20000,
  "reservoir_sample_size": 50000,
  "pattern_vocab_cap": 50000,
  "pattern_sample_cap": 30000,
  "cooccurrence_window": 4,
  "ngram_range": [2, 3],
  "analytic_target_dim": 256
}
```

See `COMMANDS.md` for detailed configuration guide.
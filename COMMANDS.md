# NovaCore - COMMANDS.md
> Last Updated: 2026-09-09 (train-pools: `--workers` / `--cpu-only`, HW auto-tune, deterministic builds, `config_colab_stream_big.json`)

> **Paths (config/config.json se):**
> - Local datasets → `data/`
> - HF cache (downloads/token) → `data/hf_cache` (config: `paths.hf_cache`)
> - HF saved/downloaded text → `data/hf_data` (config: `paths.hf_data`)
> - Models → `weights/` (bare name `--output my_model` = `weights/my_model`, `--weights my_model` bhi `weights/my_model` se load hota hai)

> **Two config files (har ka apna kaam):**
> - `config/config.json` — **model config** (dim/layers/vocab/temp, paths, compress) 
> - `config/data_config.json` — **dataset config** (native schema `fields`, HF defaults, per-dataset split/max_rows)
> Code me koi hardcoded value nahi.

## Run (without install)
```bash
python cli/main.py <command> <args>
```

## Install as command
```bash
pip install -r requirements.txt
pip install .
novacore <command> <args>
```

## ⭐ train — New Model Banana (Unified: HF + Local, multi/mixed, new/load)

**Yahi main command hai naya model banane ke liye.** Config ke bina Nahi chalega (strict config rule).

## 📋 Per-Dataset Config (config/data_config.json → `datasets`)

Har HF dataset ka **native schema (`fields`), split, max_rows** `config/data_config.json` ke `datasets`
section me pre-set hai. `fields` = ordered `{column: role}` map — har column apne `<role>...</role>`
marker ke sath preserve hota hai (schema-aware, koi flattening nahi). CLI flags diye ho to override
karte hain; warna data_config use hota hai.

```json
"datasets": {
  "rajpurkar/squad":          { "split": "train", "fields": { "context": "context", "question": "question", "answers": "answer" }, "max_rows": "full" },
  "tatsu-lab/alpaca":         { "split": "train", "fields": { "instruction": "instruction", "input": "input", "output": "output" }, "max_rows": "full" },
  "HuggingFaceH4/no_robots":  { "split": "train", "fields": { "messages": "messages" }, "max_rows": "full" },
  "OpenAssistant/oasst1":     { "split": "train", "fields": { "text": "@role" }, "max_rows": "full" },
  "OpenAssistant/oasst2":     { "split": "train", "fields": { "text": "@role" }, "max_rows": "full" },
  "roneneldan/TinyStories":   { "split": "train", "column": "text", "max_rows": "full" }
}
```

> **Multi-subset dataset (e.g. UltraData) — duplicate keys mat likho, `subset` list use karo:**
> ```json
> "openbmb/UltraData-SFT-2605": {
>   "subset": [
>     { "name": "Math", "split": "no_think", "max_rows": 100000 },
>     { "name": "Multi-lang-Knowledge", "split": "no_think", "max_rows": 100000 },
>     { "name": "Knowledge", "split": "think", "max_rows": 100000 }
>   ]
> }
> ```
> Har subset ka apna `split` + `max_rows` hota hai; CLI inhe 3 alag specs me expand karta hai.
> `##` key-suffix internally use hota hai (data_config me nahi) — file me bas `subset` list likho.

> **Three-layer priority (jo pehle milta hai wo use hota hai):**
> 1. **CLI flags** — `--column` (single column) sabse upar
> 2. **Manual config** — `data_config.json` → `datasets.<name>.fields` (native schema map)
> 3. **Auto-detect** — dataset load hote hi Native columns → semantic roles (confidence HIGH/MEDIUM)
>    → chat `messages` (<user>/<assistant>), `text`/`@role`, unknown columns WARN se reported
>    (metadata, NOT trained), aur kuch map na ho to **ERROR** (silently garbage train nahi hota)

> **Har load pe schema report print hoti hai:**
> ```
> NOVACORE DATASET SCHEMA
> Dataset: <name> | Native columns | Detected mapping | Confidence | Unknown (metadata) 
> ```
> Aur `train` me per-dataset summary: `schema: N rows trained on fields [question->question, reasoning->reasoning, ...]`
> + skipped rows count (missing/empty columns) — taaki result me pata chale sahi hua ya galat.

> **Manual override = highest priority (auto-detect usko override nahi karta).**
> Naya dataset kholke dekho (report se), phir chaaho to `fields` map se lock karo:

```bash
# PowerShell se local dataset se naya model (bare name => weights/my_model)
python cli/main.py train --dataset data/sample.txt --output my_model --config config/config.json

# Naya model + custom dim/layers (flags config ko override karte hain)
python cli/main.py train --dataset data/conversation.txt --output weights/my_model --config config/config.json --dim 256 --layers 2

# HF dataset se naya model (dataset data/hf me cache hota hai, model weights/ me)
python cli/main.py train --dataset rajpurkar/squad --output hf_model --config config/config.json --max-rows 5000

# Multiple datasets (mixed HF + local) — per-dataset --max-rows
python cli/main.py train --dataset data/sample.txt --max-rows 100 --dataset rajpurkar/squad --max-rows 5000 --output weights/big_model --config config/config.json

# Purane model ko expand karo (naya data add)
python cli/main.py train --dataset data/conversation.txt --load weights/my_model --config config/config.json
```

Main options:
- `--dataset` / `-d` : dataset (HF `owner/repo` ya local file). Repeat karo multiple/mixed ke liye.
- `--output` / `-o` : NEW model save dir
- `--load` : EXISTING model expand (naya data add). `--new` ke saath don't mix.
- `--config` : config file (base values) — **strict config: `--config` ya manual flags mein se kuch dena hi zaroori hai**
- `--dim`, `--layers`, `--vocab-size`, `--max-tokens`, `--temperature` : manual overrides
- `--max-rows` : rows limit per dataset (`full`/`all` = poori dataset)
- `--mode` : `download` (fetch + cache locally, phir reuse) ya `stream` (lazy, bade dataset / without caching ke liye)
- `--random` / `--no-random` : rows random pick (dedupe-aware) / sequential — config `random` bhi chalega
- `--add-datasets` / `--no-add-datasets` : har run me N naye unique rows pool me add / pure reuse — config `add_datasets` bhi chalega
- `--column` : text column
- `--workers N` : parallel extract worker processes. Auto = ~90% cores (cap 16); on 2-core machines auto=1 (parallel off) → Colab free pe **explicit `--workers 2`** do
- `--cpu-only` : GPU acceleration OFF — SVD numpy/BLAS hi rahega (reproducible CPU-only builds)

> **Cache auto-reuse (download skip):**
> `mode: download` hamesha data fetch + cache karta hai `paths.hf_cache` me
> (`data/hf_cache`). Dataset pehle se locally cached ho to download **skip** hota hai
> (print: `-> data already on disk - reading from cache`) aur already-downloaded subset
> se random rows select hote hain. Interrupted download ke stale `.incomplete` markers
> clean hote hain aur download wahi se resume hota hai (`partial - resuming download`).
> Token bhi wahi saved rehta hai.
>
> **Non-default split (`no_think`/`think`/`test`...) — sahi split always:**
> datasets 5.x ka eager path card-YAML multi-split repos (jinke paas `dataset_infos.json`
> nahi, e.g. `openbmb/UltraData-SFT-2605`) me galat split ki files download kar deta tha
> (`no_think` maango → `think` ki 250 files). Ab `train` ke alawa splits ko repo me se
> **sahi split ke raw shard files resolve** karke real **disk download** kiya jata hai
> (`data/hf_cache/hub`, persistent + resumable — future models same files reuse karte hain).
> Bounded `max_rows` pe sirf utne hi shards khinchte hain jitne chahiye. Saath me chhota
> **texts-cache** (`data/hf_cache/texts/<dataset>__<config>__<split>__<N>.jsonl`) banta hai
> — doosri baar `-> texts-cache hit - reusing, no download` aur turant load. (`train` splits
> classic eager path pe rehte hain: navigable + random positions.)

> **Accumulated pool (reuse/expand, no duplicates ever):**
> `mode: download` ab har `(dataset, config, split)` ka data ek **pool** me accumulate karta
> hai — `data/hf_cache/pool/<dataset>__<config>__<split>/pool.jsonl` (JSON-lines
> `{"uid","text"}`) + `manifest.json` (dataset/config/split/uid_key/rows). Har row ka UID
> (explicit `uid_key` column ya content-hash) hota hai, isliye **kabhi duplicate nahi add hota**.
> Ek sabse pehle download saare rows pool me aa jaate hain (automatic first build).
>
> `random` on/off aur `add_datasets` on/off — `config/config.json` ke saath
> `config/data_config.json` me (`hf` block defaults; per-dataset/subset keys; CLI
> `--random/--no-random` + `--add-datasets/--no-add-datasets` inhe override karte hain):
> - `random: true` → rows pool me se **random** chune jaate hain (dedupe-aware: pehle se pool
>   me maujood rows skip, approximate reservoir bounded scan ke saath).
> - `add_datasets: true` → har run me `max_rows` **naye** unique rows pool me ADD hote hain;
>   same row dobara kabhi nahi khincha jata (`-> N duplicate(s) skipped (already in pool)`).
> - `add_datasets: false` → pool jo bhi downloaded hai usi pe train hota hai
>   (`-> pool reuse ... add_datasets OFF, no fetch`); **random bhi OFF** hota hai (koi naya data nahi).
> - Naya dataset add karna: `data_config.json` ke `datasets` me naya entry/subset daalo
>   (mode/random/add_datasets/uid_key/split/max_rows ke saath); pehli run pool bana degi.
> Pehli run pehla `download` pool build karta hai (chahe `add_datasets` OFF hi ho) — koi data
> kabhi khaali nahi jaata.

> **Random rows (suru se nahi, kahi se bhi):**
> `random: true` (config) ya `--random` (CLI) + numeric `max_rows` → rows **start se nahi**,
> balki pool/DL data me se **random N** liye jaate hain. Pool me pehle se maujood rows skip
> hote hain (`add_datasets` ON ho to naye hi target). `random` OFF/sirf reuse → sequential.
> `stream` mode me rows lazily stream hote hain (RAM save); stream ke saath random/add N/A.

> **IMPORTANT:** `--config config/config.json` diya to defaults (dim=512, layers=4, vocab=30000, max_tokens=80, temp=0.8) config se aayenge. Inhe CLI flags se override kar sakte ho.

> **HW auto-tune (train / train-pools) — 2026-09-09:** har run ki shuruaat me
> `[NovaCore] HW: CPU=.. RAM=.. GPU=.. threads=.. workers=..` print hota hai.
> BLAS threads = cpu−2; workers = 90% cores (cap 16; 1 jab <4 cores). GPU (torch CUDA) sirf
> SVD stage me `_svd_cuda_or_numpy()` se use hota hai — torch **optional** (numpy/BLAS fallback),
> `--cpu-only` se band. Encode loop + reasoning/creative extraction ab **multiprocessing parallel**
> (Pool, 20k rows/chunk, serial fallback; vocab/pattern/facts byte-identical result banata hai).
> Builds ab **deterministic** hain (encoder seed fix) — same config + same data = byte-identical
> `weights.ncw`. Bada/coverage config ready: `config/config_colab_stream_big.json` (§ below).

## 🤗 Hugging Face Commands

### hf login
```bash
python cli/main.py hf login --token <YOUR_HF_TOKEN>
```
Token: https://huggingface.co/settings/tokens

### hf status
```bash
python cli/main.py hf status
```

### hf logout
```bash
python cli/main.py hf logout
```

### hf run  ⭐ (dataset → model)
HF dataset load karke usse directly NovaCore model banao.
```bash
python cli/main.py hf run --dataset <owner/repo> [options]
```
Options:
- `--dataset` / `-d` : **HF dataset ID with namespace** (e.g. `rajpurkar/squad`)
- `--name` / `-n` : config/subset name
- `--split` : train/test (default: train)
- `--column` : text column (auto-detect jika omitted)
- `--output` / `-o` : save dir (default: `weights/hf_<owner>_<repo>`)
- `--max-rows` : limit rows
- `--stream` : stream lazily (bade datasets ke liye)
- `--dim`, `--layers` : model config
- `--cache` : cache dir

### hf stream  ⭐ (preview/save dataset)
```bash
python cli/main.py hf stream --dataset <owner/repo> --rows 10
python cli/main.py hf stream --dataset <owner/repo> --save data/hf/mydata.txt --rows 1000
```
> HF downloads/saves default `data/hf_cache` (cache) / `data/hf_data` (save) me jaate hain (config: `hf_data`, `hf_cache`).

> **IMPORTANT:** Dataset ID me hamesha `owner/repo` (namespace) chahiye, e.g. `rajpurkar/squad`.
> `squad` seedha nahi chalega naye datasets library me.

## Commands

### train  ⭐ (New model / expand)
```bash
# New model (quick PowerCell test)
python cli/main.py train --dataset data/sample.txt --output weights/my_model --config config/config.json

# Expand existing model with more data
python cli/main.py train --dataset data/conversation.txt --load weights/my_model --config config/config.json
```
> Details/options upar "train — New Model Banana" section me hain.

### encode
Local dataset ko weights me convert karo (training-free).
```bash
python cli/main.py encode --input <dataset> --output <dir> [options]
```
Options:
- `--input` / `-i` : Dataset file (CSV/JSON/TXT/TSV)
- `--output` / `-o` : Weights save dir (default: `weights/<filename>`)
- `--dim` : Weight dimension (default: config.json default 512)
- `--layers` : Encoding layers (default: config.json default 4)
- `--format` : auto/csv/json/txt/tsv
- `--text-column` : Text column (CSV/JSON)

### chat  ⭐ (Ollama-style)
Saved model load karke interactive chat karo.
```bash
# Interactive mode
python cli/main.py chat --weights <dir>

# One-shot mode
python cli/main.py chat --weights <dir> --prompt "your question"
```
Options:
- `--weights` / `-w` : Weights dir (required)
- `--system` / `-s` : System prompt
- `--tokens` / `-t` : Max tokens per reply (default from config)
- `--temperature` : Sampling temp (default from config)
- `--prompt` / `-p` : One-shot prompt (non-interactive)

Chat commands (interactive):
- `/new`  - reset conversation
- `/info` - show model info
- `/exit` - quit

### generate
One-off text generation.
```bash
python cli/main.py generate --weights <dir> --prompt <text> [options]
```

### info / compress / query / validate / list
Model management (see earlier docs).

## Model Config / Defaults
Saare defaults `config/config.json` me hain — CLI unhe use karta hai.
Apne model ki dim/layers badalni ho to `config/config.json` update karo ya CLI flag do.

> **Strict config:** `train`, `hf run`, `encode` me `--config <file>` ya manual flags (`--dim --layers --vocab-size --max-tokens --temperature`) me se KAM SE KAM EK dena zaroori hai. Warna error aayega.

### chat  ⭐ (Ollama-style)
Saved model load karke interactive chat karo.
```bash
# Interactive mode
python cli/main.py chat --weights <dir>

# One-shot mode
python cli/main.py chat --weights <dir> --prompt "your question"
```
Options:
- `--weights` / `-w` : Weights dir (required)
- `--system` / `-s` : System prompt
- `--tokens` / `-t` : Max tokens per reply (default from model config)
- `--temperature` : Sampling temp (default from model config)
- `--prompt` / `-p` : One-shot prompt (non-interactive)

Chat commands (interactive):
- `/new`  - reset conversation
- `/info` - show model info
- `/models` - list available models
- `/load <name>` - switch model
- `/exit` - quit

Note: `hi` / `hello` / `how are you` / `what is your name` / `thanks` / `bye`
jaisi greetings ab bhi respond karti hain (fallback + topic starters).

### generate
One-off text generation.
```bash
python cli/main.py generate --weights <dir> --prompt <text> [options]
```

### info
Saved weights ki info.
```bash
python cli/main.py info --weights <dir>
```

### compress
Weights ko compress.
```bash
python cli/main.py compress --weights <dir> --ratio <0.0-1.0>
```

### query
Text ko feature vector me process.
```bash
python cli/main.py query --weights <dir> --text <text>
```

### validate
Weights integrity check.
```bash
python cli/main.py validate --weights <dir>
```

### list
Saare saved models list.
```bash
python cli/main.py list
```

## Internal Components Configuration

NovaCore ke internal components (Neural Engine, Virtual Simulation, Verification Engine, Python Terminal) sab `config/config.json` ke `defaults` section se configure hote hain.

### Configuration Sections

#### 1. Neural Engine (Internal Neural Network)
Location: `config/config.json` → `defaults.neural_engine`

```json
"neural_engine": {
  "enabled": true,              // Enable/disable neural engine
  "input_dim": 64,              // Input layer neurons
  "hidden_dim": 32,             // Hidden layer neurons
  "output_dim": 16,             // Output layer neurons
  "activation_cache_size": 100, // Cache size for activations
  "use_fast_sigmoid": true,     // Fast sigmoid approximation
  "enable_caching": true        // Enable activation caching
}
```

#### 2. Virtual Simulation (Quality Control)
Location: `config/config.json` → `defaults.virtual_simulation`

```json
"virtual_simulation": {
  "enabled": true,              // Enable/disable simulation
  "quality_threshold": 0.4,     // Minimum score to pass
  "enable_caching": true,       // Cache scores
  "cache_size": 1000,           // Max cache entries
  "scoring_weights": {
    "length": 0.4,              // Length score weight
    "relevance": 0.4,           // Relevance weight
    "coherence": 0.2            // Coherence weight
  }
}
```

#### 3. Verification Engine (Retry Logic)
Location: `config/config.json` → `defaults.verification_engine`

```json
"verification_engine": {
  "enabled": true,              // Enable/disable verification
  "max_retries": 2,             // Max retry attempts
  "min_score": 0.3,             // Minimum acceptable score
  "enable_caching": true,       // Cache scores
  "reservoir_search_limit": 200,// Max items to search
  "enable_fast_retry": true     // Quick retry mode
}
```

#### 4. Python Terminal (Internal Code Execution)
Location: `config/config.json` → `defaults.python_terminal`

```json
"python_terminal": {
  "enabled": true,              // Enable/disable terminal
  "safe_mode": true,            // Safe execution mode
  "max_execution_time_ms": 5000,// Execution timeout
  "allowed_operations": [       // Allowed operations
    "print", "math", "variables", "loops", "conditionals"
  ]
}
```

#### 5. Chat Configuration
Location: `config/config.json` → `defaults.chat`

```json
"chat": {
  "max_history": 10,            // Conversation history limit
  "context_tokens": 200,        // Context window size
  "generation_history_tokens": 10, // Generation history
  "use_model_brain": true       // Use internal ModelBrain
}
```

#### 6. RAM-Safe Overrides (Auto-applied <16GB RAM)
Location: `config/config.json` → `defaults.ram_safe_overrides`

```json
"ram_safe_overrides": {
  "dim": 256,                    // Smaller model
  "layers": 2,                   // Fewer layers
  "vocab_size": 20000,           // Smaller vocab
  "reservoir_sample_size": 50000,// Smaller reservoir
  "pattern_vocab_cap": 50000,    // Smaller pattern vocab
  "pattern_sample_cap": 30000,   // Smaller pattern sample
  "cooccurrence_window": 4,      // Smaller window
  "ngram_range": [2, 3],         // Simpler n-grams
  "analytic_target_dim": 256     // Smaller target dim
}
```

### Kaha Se Change Karein

| Component | Config Path |
|-----------|-------------|
| Neural Network size | `defaults.neural_engine.{input_dim,hidden_dim,output_dim}` |
| Quality threshold | `defaults.virtual_simulation.quality_threshold` |
| Max retries | `defaults.verification_engine.max_retries` |
| Python timeout | `defaults.python_terminal.max_execution_time_ms` |
| Chat history | `defaults.chat.max_history` |
| RAM limits | `defaults.ram_safe_overrides` |

### Performance vs Quality Tradeoff

| Setting | Lower Value | Higher Value |
|---------|-------------|--------------|
| `neural_engine.input_dim` | Faster, less accurate | Slower, more accurate |
| `virtual_simulation.quality_threshold` | More responses pass | Stricter quality |
| `verification_engine.max_retries` | Faster, may miss quality | Slower, better quality |
| `verification_engine.reservoir_search_limit` | Faster, less coverage | Slower, better coverage |
| `ram_safe_overrides.dim/layers` | Lower RAM, lower quality | Higher RAM, higher quality |

### CPU-Friendly Defaults

All defaults are optimized for CPU:
- Small neural network (64x32x16 = 2560 params)
- Fast sigmoid approximation
- Activation caching
- Limited reservoir search (200 items)
- Max 2 retries
- Quality caching

### Examples

#### Maximum Quality (More RAM)
```json
"neural_engine": {
  "input_dim": 128,
  "hidden_dim": 64,
  "output_dim": 32
},
"virtual_simulation": {
  "quality_threshold": 0.6
},
"verification_engine": {
  "max_retries": 5,
  "reservoir_search_limit": 1000
}
```

#### Maximum Speed (Low RAM)
```json
"neural_engine": {
  "input_dim": 32,
  "hidden_dim": 16,
  "output_dim": 8
},
"virtual_simulation": {
  "quality_threshold": 0.2
},
"verification_engine": {
  "max_retries": 1,
  "reservoir_search_limit": 100
}
```

#### Balanced (Default)
```json
"neural_engine": {
  "input_dim": 64,
  "hidden_dim": 32,
  "output_dim": 16
},
"virtual_simulation": {
  "quality_threshold": 0.4
},
"verification_engine": {
  "max_retries": 2,
  "reservoir_search_limit": 200
}
```

### Notes

- All values are config-driven, no hardcoded values in code
- Changes apply immediately on next model load
- CPU-friendly by default, GPU not required
- Memory usage scales with reservoir size and neural network dimensions
## Verified Examples (PowerShell me tested)
```bash
# 1. Naya model banao — train, bare name -> weights/my_model (strict config)
python cli/main.py train --dataset data/sample.txt --output my_model --config config/config.json --dim 256 --layers 2

# 2. Chhat se test karo (--weights my_model bhi weights/ se load hota hai)
python cli/main.py chat --weights my_model

# 3. One-shot chat
python cli/main.py chat --weights my_model --prompt "tell me about space" --tokens 20

# 4. Purana model expand karo
python cli/main.py train --dataset data/conversation.txt --load my_model --config config/config.json

# 5. HF dataset se naya model (model weights/hf_squad_model me banega)
python cli/main.py train --dataset rajpurkar/squad --output hf_squad_model --config config/config.json --max-rows 20

python cli/main.py train --dataset roneneldan/TinyStories --output test_tinystories --config config/config.json --mode stream --max-rows 1000

python cli/main.py train --dataset roneneldan/TinyStories --output test_tinystories_20k --config config/config.json --max-rows 20000

python cli/main.py chat --weights test_tinystories

python cli/main.py train --dataset data\conversation.txt --output conv_model --config config\config.json
python cli/main.py chat --weights conv_model

python cli/main.py train --dataset roneneldan/TinyStories --output tinystories_20k --config config\config.json --max-rows 20000

python cli/main.py chat --weights test_tinystories --temperature 0.2 --tokens 20

python cli/main.py hf login --token hf_XXXXXX

python cli/main.py train --dataset roneneldan/TinyStories --output test_tinystories --config config/config.json --max-rows full

python cli/main.py train --dataset roneneldan/TinyStories --dataset rajpurkar/squad --output test_model --config config/config.json --max-rows 500 --max-rows 1000

python cli/main.py train --dataset config\data_config.json --dataset roneneldan/TinyStories --output test_model --config config/config.json

python cli/main.py train --dataset roneneldan/TinyStories --output test_tinystories --config config/config.json --max-rows 1000

python cli/main.py train --dataset config\data_config.json --output VedNex_V1 --config config/config.json

# Base model
python cli/main.py train-pools --dataset config/data_small.json --output NovaCore --config config_colab.json

# Domain adaptation: Code dataset
python cli/main.py train-pools --load NovaCore --dataset config/data_code.json --add-datasets --config config_colab.json

# Domain adaptation: Hindi dataset  
python cli/main.py train-pools --load NovaCore --dataset config/data_hindi.json --add-datasets --config config_colab.json

# Final model has ALL knowledge
python cli/main.py chat --weights weights/NovaCore

## Colab free GPU — streaming build (corrected 2026-09-09)

> **Pehle GitHub pe push karo** — `!git clone` remote code use karta hai; `--workers`,
> solver, parallel build, encoder determinism fix sab sirf local changes hain.
> Config: `config/config_colab_stream.json` (normal) ya `config/config_colab_stream_big.json`
> (bada coverage: vocab 80k, qa_bank 200k, pattern caps 400k/1M, reasoning/creative caps 150k).

```python
# Cell 1: Setup — sirf FRESH session me. Wahi session rerun karne pe cell 1 chhodo (skip/pull).
%cd /content
!git clone -q https://github.com/anupbth1/NovaCore.git || (cd NovaCore && git pull -q)
%cd NovaCore
from huggingface_hub import login
login(token="hf_XXXXXX")   # apna token; test ke baad ROTATE

# Cell 2: BASE — stream (~25-35 min, 2 vCPU + T4)
!python cli/main.py train-pools --dataset config/data_colab_general.json --config config/config_colab_stream.json --output NovaCoreV15 --workers 2

# Cell 3: Conversation
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_conversation.json --config config/config_colab_stream.json --workers 2

# Cell 4: Coding   (load = NovaCoreV15, V10 NAHI)
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_coding.json --config config/config_colab_stream.json --workers 2

# Cell 5: Hindi
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_hindi.json --config config/config_colab_stream.json --workers 2

# Cell 6: Reasoning
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_reasoning.json --config config/config_colab_stream.json --workers 2

# Cell 7: Creative
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_creative.json --config config/config_colab_stream.json --workers 2

# Cell 8: Science
!python cli/main.py train-pools --load NovaCoreV15 --dataset config/data_colab_domain_science.json --config config/config_colab_stream.json --workers 2
```

> **Rules:**
> - Har domain cell **ek baar hi** chalana — dobara chalane se duplicate facts/patterns inflate hote hain.
> - Config saare cells me **same** rehna chahiye (expand me dim/layers/vocab match hona zaroori hai) — bich me mat badlo.
> - `--cpu-only` **mat** lagao — GPU (torch CUDA) SVD stage accel karta hai.
> - Base ~5 min nahi hoga: stream loops pure-Python hain; `--workers 2` (multiprocessing) + GPU SVD
>   roughly 1.5–1.8x dete hain. Start pe `[NovaCore] HW: CPU=2 GPU=T4...` line se verify karo.
> - Re-run: fail hone par wahi cell dubara chalao (page new remaining cells pichle state se age).

### Big/coverage config — `config/config_colab_stream_big.json` (Colab free RAM-safe)
Higher `vocab_size` (80k), `qa_bank_cap` 200k, `pattern_vocab_cap` 400k, `pattern_sample_cap` 1M,
`reasoning/creative_extract_cap` 150k. RAKHA HAI: `dim` 1024, `reservoir_sample_size` 100k
(reservoir 250k → analytic X+Y matrices ~6GB → OOM on 12GB Colab). Bigger = zyada COVERAGE,
"smarter" nahi — exact math/transitive answers solver fast-path se aate hain (config-independent).
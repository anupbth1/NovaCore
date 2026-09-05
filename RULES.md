# NovaCore - Development Rules (ALWAYS FOLLOW)
> Ye rules is project me hamesha follow honge. Kabhi bhi inko violate mat karna.

## Rule 1: NO HARDCODED CONTENT (SABSE IMPORTANT)
- **Kabhi bhi** model responses, replies, templates ko hardcoded mat likho.
- Responses **hamesha** dataset weights se aane chahiye — matlab weights/patterns se generate hona chahiye.
- Hardcoded = data jo code me seedha likha ho (e.g. "hello", "i am fine", fixed replies).
- Data ka koi bhi hissa code me direct embed **NAHI** hona chahiye.
- Agar koi response dena hai to wo **dataset se source** hona chahiye (encoded weights se retrieve).

### Kya hardcoded hai (BAND):
- Static reply templates ("hi" -> "hello", fixed greetings)
- Seedha likha hua conversation text
- Fixed response maps/dictionaries jo data source na ho

### Kya allowed hai:
- Code logic (math, control flow, algorithms)
- Coding constants (dim, layers, seeds — ye config/config.json me hona chahiye, code me nahi)
- Dataset se aaya hua data (weights, patterns, vocab)

## Rule 2: Configuration Config File Me (config.json)
- `dim`, `layers`, `seed`, `temperature`, `max_tokens`, `vocab_size` jaisi settings
  **config/config.json** me honi chahiye — code me nahi.
- Code kabhi bhi hardcoded defaults na rakhe; config se load kare.

## Rule 3: Dataset-Centric
- Jo bhi model jaanta hai wo **dataset se** aana chahiye.
- Dataset jitna bada / relevant hoga, model utna behtar.
- Model ko inference ke time koi naya hardcoded knowledge nahi dena.

## Rule 4: Kisi Bhi Naye Change Ke Baad
- 5 files update karo: PLAN.md, STRUCTURE.md, COMMANDS.md, README.md, MASTER_PLAN.md
- Tests run karo (agar exists karein): `python -m pytest tests/ -q`

## Rule 5: Modularity
- Naya feature = naya module/class. Ek file me sab mat daalo.

## Rule 6: Yaad Rakho — NovaCore ka Goal
- Training-free, data ke mathematical representation se weights banaye = "Kaam transformer karta hai, uske bina".
- Yani each token/pattern ki value weights me store ho, code me nahi.

## Rule 7: Hugging Face Datasets
- HF datasets download/stream karne ke liye `novacore hf <command>` use karo.
- **Dataset ID me hamesha namespace/owner hona chahiye** (e.g. `rajpurkar/squad`, nahi `squad`).
  Naye `datasets` library (v5+) me bina namespace ka ID nahi chalega.
- Login: `novacore hf login --token <TOKEN>` (token: https://huggingface.co/settings/tokens)
- Bade datasets ke liye `--stream` use karo (RAM save).
- Limits ke liye `--max-rows` use karo.

## Rule 8: Model Save Location / Naming
- Saare models `weights/` folder ke andar save hote hain.
- HF se banaya model: `weights/hf_<dataset_id>` (slashes → underscores).
  e.g. `rajpurkar/squad` → `weights/hf_rajpurkar_squad`
- Local file se: `weights/<filename>` e.g. `data/sample.txt` → `weights/sample`
- Custom: `--output <path>` se koi bhi naam/dir de sakte ho.

## Rule 9: Config Se Sab Defaults
- dim, layers, max_tokens, temperature, etc. **config/config.json** se aate hain.
- CLI default values config.json ke `defaults` section se load hote hain.
- Code me koi hardcoded default Nahi hona chahiye (Rule 1+2).

## Rule 10: New Model `train` Command (Strict Config)
- Naya model banane ke liye `novacore train --dataset <data> --output <dir> --config config/config.json` use karo.
- **Strict config:** `train`/`hf run`/`encode` me `--config` ya manual flags
  (`--dim --layers --vocab-size --max-tokens --temperature`) me se KAM SE KAM EK dena zaroori hai.
  Rules (config file + flags mix ho sakte hain; flags override karte hain). Warna ERROR.
- Naya data add (expand): `train --dataset <new> --load <existing_dir> --config config/config.json`
  (`--new` aur `--load` saath me Nahi).

## Rule 11: Data aur Models ki Location (config paths)
- **Saare datasets (local)** → `data/`; HF cache/downloads → `data/hf_cache/`; HF saved text → `data/hf_data/`.
  Paths: `data_dir`, `hf_data`, `hf_cache` — `config/config.json` se aate hain (default `data/hf_data` + `data/hf_cache`).
- **Saare models** → `weights/` (`default_weights` path se).
- CLI me bare name = weights ke andar: `--output my_model` → `weights/my_model`,
  `--load my_model` / `--weights my_model` bhi `weights/` se resolve hote hain.
- Datasets/models kahin aur mat daalo; paths change karne ho to config.json badlo, code nahi.

## Rule 12: Cache Auto-Reuse + Resume (HF)
- Dataset pehle se locally cached ho to **download skip** → direct load + train
  (`already cached - reusing`). Interrupted downloads ke stale `.incomplete` markers clean
  hokar resume hote hain (`partial - resuming download`). Ye HFLoader `load()` me automatic hai.

## Rule 13: Random Rows + Accumulated Pool (max_rows / random / add_datasets)
- **`random` on/off config se control hota hai** — `config/data_config.json`:
  `hf` block defaults (`random`, `add_datasets`, `uid_key`) + per-dataset/per-subset keys;
  CLI `--random/--no-random`, `--add-datasets/--no-add-datasets` override.
  `random: true` + numeric `max_rows` = pool/DL data me se **random N** (dedupe-aware,
  approximate reservoir with bounded scan); OFF/stream = sequential/lazy.
- `mode: download` hamesha fetch + cache karta hai (bounded `max_rows` to utne hi rows
  materialize hote hain, poori repo nahi); cached hone par reuse hota hai.
- **Accumulated pool (per `(dataset, config, split)`):** `data/hf_cache/pool/<key>/pool.jsonl`
  (JSON-lines `{"uid","text"}`) + `manifest.json`. Har row UID-sealed (`uid_key` column ya
  sha1 content-hash) → **duplicates kabhi nahi add hote**.
  - `add_datasets: false` (default): pool jo hai usi pe train (pehli run bhi pool bana deti hai —
    khaali kabhi nahi); log `-> pool reuse ... add_datasets OFF, no fetch`. Random bhi OFF.
  - `add_datasets: true`: har run me `max_rows` **naye** unique rows add (`N duplicate(s) skipped`).
  - Legacy `texts-cache` files pool me **seed** ho jaate hain (data kabhi waste nahi).
- **Non-default split (`no_think`/`think`/`test` etc.)**: datasets 5.x ke eager path me
  card-YAML multi-split configs (jinke paas `dataset_infos.json` nahi) **galat split ki files
  pick kar dete hain** (e.g. `no_think` maango → `think` download). Isliye non-`train` splits
  repo me se **sahi split ke raw shards resolve** karke **real disk download** hote hain
  (`data/hf_cache/hub`, persistent + resumable) + chhota **texts-cache** (`data/hf_cache/texts/`)
  banta hai taaki doosri baar bina download reuse ho.
- `mode: stream` me rows lazily stream hote hain (RAM save); stream ke saath random/add N/A.

"""NovaCore CLI - command-line interface."""
import json
import os
import sys
import argparse
import random

# Set BLAS/vectorized thread counts BEFORE numpy is first imported so the
# whole process (and every worker) uses ~90% of available cores.
if os.environ.get('NOVACORE_NOTUNE') is None:
    try:
        _nc_cores = max(1, int(os.cpu_count() or 1) - 1)
        for _nc_k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                      'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(_nc_k, str(_nc_cores))
    except Exception:
        pass

import numpy as np
from tqdm import tqdm

# Ensure we can import novacore package even if not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novacore.core.encoder import RandomFourierEncoder, backend_info, release_memory
from novacore.core.hasher import FeatureHasher
from novacore.core.patterns import PatternExtractor
from novacore.core.reservoir import ReservoirSampler
from novacore.tokenizer.vocab import Vocabulary
from novacore.tokenizer.text_processor import TextProcessor
from novacore.storage.weight_manager import WeightManager
from novacore.inference.predictor import Predictor, PatternPredictor
from novacore.inference.chat import ChatSession
from novacore.dataset.loader import DatasetLoader
from novacore.dataset.hf_loader import HFLoader
from novacore.config import get_default, get_path, get_hf, get_cli, get_dataset, resolve_path
from novacore.auto_tuner import get_tuner
from novacore.logger import Logger

# Default weights dir from config (Rule 2 - no hardcoded paths)
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_weights = get_path('default_weights')
DEFAULT_WEIGHTS_DIR = _config_weights if os.path.isabs(_config_weights) else os.path.join(_base, _config_weights)


def _resolve_model_dir(name_or_dir, default_base=DEFAULT_WEIGHTS_DIR):
    """Resolve a model dir. Bare name (no path sep) -> <default_base>/<name>."""
    if name_or_dir is None:
        return None
    s = str(name_or_dir)
    if not os.path.isabs(s) and not os.path.dirname(s):
        return os.path.join(default_base, s)
    return s


def build_parser():
    p = argparse.ArgumentParser(prog="novacore", description="NovaCore - Training-Free LLM")
    sub = p.add_subparsers(dest="command", required=True)

    _dim = get_default('dim', None)
    _layers = get_default('layers', None)
    _tokens = get_default('max_tokens', None)
    _temp = get_default('temperature', None)

    # encode
    enc = sub.add_parser("encode", help="Encode dataset to weights (no training)")
    enc.add_argument("--input", "-i", required=True, help="Dataset file")
    enc.add_argument("--output", "-o", default=None, help="Output weights dir")
    enc.add_argument("--config", default=None, help="Config file (e.g. config/config.json) - base values")
    enc.add_argument("--dim", type=int, default=None, help="Weight dimension (overrides config)")
    enc.add_argument("--layers", type=int, default=None, help="Encoding layers (overrides config)")
    enc.add_argument("--vocab-size", type=int, default=None, help="Vocab size (overrides config)")
    enc.add_argument("--max-tokens", type=int, default=None, help="Max tokens for chat (overrides config)")
    enc.add_argument("--temperature", type=float, default=None, help="Sampling temp (overrides config)")
    enc.add_argument("--format", default=None, help="Dataset format (auto/csv/json/txt/tsv)")
    enc.add_argument("--text-column", default=None, help="Text column for CSV/JSON")

    # generate
    gen = sub.add_parser("generate", help="Generate text (next-token prediction)")
    gen.add_argument("--weights", "-w", required=True, help="Weights dir")
    gen.add_argument("--prompt", "-p", required=True, help="Prompt text")
    gen.add_argument("--tokens", "-t", type=int, default=None, help="Num tokens to generate (default: model config)")
    gen.add_argument("--temperature", type=float, default=None, help="Sampling temp (default: model config)")

    # info
    inf = sub.add_parser("info", help="Show model info")
    inf.add_argument("--weights", "-w", required=True, help="Weights dir")

    # compress
    cmp = sub.add_parser("compress", help="Compress weights")
    cmp.add_argument("--weights", "-w", required=True, help="Weights dir")
    cmp.add_argument("--ratio", "-r", type=float, default=get_cli('compress_ratio', 0.5), help="Compression ratio 0-1")

    # query
    qry = sub.add_parser("query", help="Query similar content")
    qry.add_argument("--weights", "-w", required=True, help="Weights dir")
    qry.add_argument("--text", "-t", required=True, help="Search text")

    # validate
    val = sub.add_parser("validate", help="Validate weights")
    val.add_argument("--weights", "-w", required=True, help="Weights dir")

    # list
    lst = sub.add_parser("list", help="List saved models")
    lst.add_argument("--dir", default=DEFAULT_WEIGHTS_DIR, help="Base weights dir")

    # convert (legacy -> ncw format)
    conv = sub.add_parser("convert", help="Convert legacy weights to NovaCore .ncw format")
    conv.add_argument("--weights", "-w", default=None, help="Weights dir to convert")
    conv.add_argument("--all", "-a", dest="convert_all", action="store_true",
                      help="Convert all models under default weights dir")
    conv.add_argument("--dir", default=DEFAULT_WEIGHTS_DIR, help="Base dir for --all")
    conv.set_defaults(convert_all=False)

    # chat
    chat = sub.add_parser("chat", help="Interactive chat with a saved model (like Ollama)")
    chat.add_argument("--weights", "-w", required=True, help="Weights dir")
    chat.add_argument("--system", "-s", default=None, help="Optional system prompt")
    chat.add_argument("--tokens", "-t", type=int, default=_tokens, help="Max tokens per reply")
    chat.add_argument("--temperature", type=float, default=_temp, help="Sampling temp")
    chat.add_argument("--prompt", "-p", default=None, help="Single prompt mode (non-interactive, then exit)")
    chat.add_argument("--verbose", "-v", dest="verbose", action="store_true", default=True,
                      help="Show step-by-step reasoning logs (ON by default)")
    chat.add_argument("--quiet", "-q", dest="verbose", action="store_false",
                      help="Suppress step-by-step logs")
    chat.add_argument("--cortex", dest="cortex", action="store_true",
                      help="Enable CORTEX attention recall head (training-free)")
    chat.add_argument("--cde", dest="cde", action="store_true",
                      help="Enable the unified Cognitive Discovery Engine "
                           "(merged think/reason/plan/simulate/attack/verify)")
    chat.set_defaults(remaining=[])

    # cortex - training-free attention memory net (bench + chat)
    ctx = sub.add_parser("cortex", help="Training-free linear-attention memory net (built from weights)")
    ctx.set_defaults(cortex_cmd="chat")
    ctx_sub = ctx.add_subparsers(dest="cortex_cmd")
    ctx_chat = ctx_sub.add_parser("chat", help="Chat using the attention memory net")
    ctx_chat.add_argument("--weights", "-w", required=True, help="Weights dir")
    ctx_chat.add_argument("--prompt", "-p", default=None, help="Prompt text (non-interactive)")
    ctx_chat.add_argument("--max-tokens", "-t", type=int, default=120, help="Max generated tokens")
    ctx_chat.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ctx_chat.add_argument("--dim", type=int, default=256, help="Attention embedding dim")
    ctx_chat.add_argument("--topk", type=int, default=12, help="Attention top-k memories")
    ctx_chat.add_argument("--gate", type=float, default=0.35, help="Attention gate threshold")
    ctx_chat.add_argument("--max-memories", type=int, default=25000, help="Memory snippet cap")
    ctx_bench = ctx_sub.add_parser("bench", help="Benchmark: cortex attention net vs legacy chat path")
    ctx_bench.add_argument("--weights", "-w", required=True, help="Weights dir")
    ctx_bench.add_argument("--questions", "-q", required=True, help="Comma-separated questions")
    ctx_bench.add_argument("--max-tokens", "-t", type=int, default=100, help="Max generated tokens")
    ctx_bench.add_argument("--dim", type=int, default=256, help="Attention embedding dim")
    ctx_bench.add_argument("--topk", type=int, default=12, help="Attention top-k memories")
    ctx_bench.add_argument("--gate", type=float, default=0.35, help="Attention gate threshold")
    ctx_bench.add_argument("--max-memories", type=int, default=25000, help="Memory snippet cap")

    # train - unified model creation (HF + local + multi + mixed + new/load)
    tr = sub.add_parser("train", help="Create/expand a model from HF and/or local datasets (config-based)")
    tr.add_argument("--dataset", "-d", action="append", default=None,
                    help="Dataset. HF: 'owner/repo[:split]' or local file path. Repeat for multiple/mixed. ")
    tr.add_argument("--config", default=None, help="Config file (e.g. config/config.json) - base values")
    tr.add_argument("--output", "-o", required=False, default=None, help="Output model dir (new model)")
    tr.add_argument("--new", dest="new_mode", action="store_true", help="Force create a NEW model")
    tr.add_argument("--load", dest="load_model", default=None, help="Load EXISTING model dir to expand with more data")
    tr.add_argument("--mode", choices=["download", "stream"], default=None, help="download=full, stream=lazy")
    tr.add_argument("--random", dest="random_flag", action="store_true", default=None,
                    help="Pick rows randomly (download mode; dedupe-aware, NO repeats)")
    tr.add_argument("--no-random", dest="random_flag", action="store_false",
                    help="Sequential rows (download mode)")
    tr.add_argument("--add-datasets", dest="add_flag", action="store_true", default=None,
                    help="Pull N NEW unique rows into the pool (accumulate; repeats skipped)")
    tr.add_argument("--no-add-datasets", dest="add_flag", action="store_false",
                    help="Reuse whatever is downloaded; fetch/add nothing")
    tr.add_argument("--rows", dest="max_rows", default=None, help="Max rows to use (download mode)")
    tr.add_argument("--column", default=None, help="Text column name for local CSV/JSON")
    tr.add_argument("--fields", action="append", default=None, help="Field mapping (e.g. text->text)")
    tr.add_argument("--field", dest="fields", action="append", default=None, help="Alias for --fields")

    # train-pools - train from existing pools only (no download)
    trp = sub.add_parser("train-pools", help="Train from existing pools only (no download)")
    trp.add_argument("--dataset", "-d", action="append", default=None,
                     help="Dataset spec file (e.g. config/pools_only.json)")
    trp.add_argument("--config", default=None, help="Config file (e.g. config/config.json)")
    trp.add_argument("--output", "-o", required=False, default=None, help="Output model dir")
    trp.add_argument("--new", dest="new_mode", action="store_true", help="Force create a NEW model")
    trp.add_argument("--load", dest="load_model", default=None, help="Load EXISTING model dir to expand")
    trp.add_argument("--mode", choices=["download", "stream"], default=None, help="download=full, stream=lazy")
    trp.add_argument("--random", dest="random_flag", action="store_true", default=None, help="Random rows")
    trp.add_argument("--no-random", dest="random_flag", action="store_false", help="Sequential rows")
    trp.add_argument("--add-datasets", dest="add_flag", action="store_true", default=None, help="Add new rows")
    trp.add_argument("--no-add-datasets", dest="add_flag", action="store_false", help="Reuse existing")
    trp.add_argument("--rows", dest="max_rows", default=None, help="Max rows")
    trp.add_argument("--column", default=None, help="Text column for local files")
    trp.add_argument("--fields", action="append", default=None, help="Field mapping")
    trp.add_argument("--field", dest="fields", action="append", default=None, help="Alias for --fields")
    trp.add_argument("--absorb", dest="absorb_flag", action="store_true", default=None,
                    help="Absorb ALL already-downloaded shards into the pool (zero network)")
    trp.add_argument("--max-rows", action="append", default=None,
                    help="Rows for matching --dataset (number, or 'full'/'all' for entire). Repeat per dataset.")
    trp.add_argument("--dim", type=int, default=None, help="Weight dimension (overrides config)")
    trp.add_argument("--layers", type=int, default=None, help="Encoding layers (overrides config)")
    trp.add_argument("--vocab-size", type=int, default=None, help="Vocab size (overrides config)")
    trp.add_argument("--max-tokens", type=int, default=None, help="Max tokens for chat (overrides config)")
    trp.add_argument("--temperature", type=float, default=None, help="Sampling temp (overrides config)")
    tr.add_argument("--absorb", dest="absorb_flag", action="store_true", default=None,
                    help="Absorb ALL already-downloaded shards into the pool (zero network)")
    tr.add_argument("--max-rows", action="append", default=None,
                    help="Rows for matching --dataset (number, or 'full'/'all' for entire). Repeat per dataset.")
    tr.add_argument("--dim", type=int, default=None, help="Weight dimension (overrides config)")
    tr.add_argument("--layers", type=int, default=None, help="Encoding layers (overrides config)")
    tr.add_argument("--vocab-size", type=int, default=None, help="Vocab size (overrides config)")
    tr.add_argument("--max-tokens", type=int, default=None, help="Max tokens for chat (overrides config)")
    tr.add_argument("--temperature", type=float, default=None, help="Sampling temp (overrides config)")

    # HW / speed controls (train + train-pools)
    for _parser in (tr, trp):
        _parser.add_argument("--workers", type=int, default=None,
                             help="Parallel extract worker processes (auto ~90% of cores; 1 = off)")
        _parser.add_argument("--cpu-only", action="store_true", default=False,
                             help="Force CPU path (no GPU acceleration)")

    # HF group
    hf = sub.add_parser("hf", help="Hugging Face dataset operations")
    hf_sub = hf.add_subparsers(dest="hf_command", required=True)

    hf_login = hf_sub.add_parser("login", help="Login to Hugging Face")
    hf_login.add_argument("--token", default=None, help="HF token (or paste interactively)")
    hf_login.add_argument("--write", dest="write_perm", action="store_true", help="Request write perms")

    hf_out = hf_sub.add_parser("logout", help="Logout from Hugging Face")

    hf_stats = hf_sub.add_parser("status", help="Check HF auth status")

    hf_dl = hf_sub.add_parser("run", help="Load a HF dataset and encode it into a NovaCore model")
    hf_dl.add_argument("--dataset", "-d", required=True, help="HF dataset name (e.g. 'squad', 'user/repo')")
    hf_dl.add_argument("--name", "-n", default=None, help="Config name (subset), passed to load_dataset")
    hf_dl.add_argument("--split", default=get_hf('default_split', 'train'), help="Dataset split (train/test)")
    hf_dl.add_argument("--column", default=None, help="Text column (auto-detect if omitted)")
    hf_dl.add_argument("--output", "-o", default=None, help="Output weights dir")
    hf_dl.add_argument("--max-rows", type=int, default=None, help="Limit rows to process")
    hf_dl.add_argument("--stream", dest="stream", action="store_true", help="Stream lazily (for huge datasets)")
    hf_dl.add_argument("--config", default=None, help="Config file (e.g. config/config.json) - base values")
    hf_dl.add_argument("--dim", type=int, default=None, help="Weight dimension (overrides config)")
    hf_dl.add_argument("--layers", type=int, default=None, help="Encoding layers (overrides config)")
    hf_dl.add_argument("--vocab-size", type=int, default=None, help="Vocab size (overrides config)")
    hf_dl.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides config)")
    hf_dl.add_argument("--max-tokens", dest="tokens", type=int, default=None, help="Max tokens for chat (overrides config)")
    hf_dl.add_argument("--cache", default=None, help="Cache dir (default: config hf_cache)")
    hf_dl.add_argument("--text-column", dest="text_column", default=None, help="Text column alias")

    hf_info = hf_sub.add_parser("stream", help="Stream a HF dataset and preview / save it")
    hf_info.add_argument("--dataset", "-d", required=True, help="HF dataset name")
    hf_info.add_argument("--split", default=get_hf('default_split', 'train'), help="Dataset split")
    hf_info.add_argument("--rows", type=int, default=get_cli('preview_rows', 10), help="Number of rows to preview")
    hf_info.add_argument("--column", default=None, help="Text column")
    hf_info.add_argument("--save", default=None, help="Optional: save N rows to this text file")

    return p


def _resolve_model_config(args):
    """
    Resolve model config from --config file (base) + manual flags (override).

    Rules (user requirement):
      - Config values come from --config file OR manual flags (or both mixed).
      - Manual flags override config file values (per-setting).
      - If NEITHER --config nor any manual model flag is given -> error (no defaults).
    Returns dict with: dim, layers, vocab_size, max_tokens (tokens), temperature.
    """
    model_keys = ["dim", "layers", "vocab_size", "max_tokens", "temperature"]

    cfg = {}
    provided = {}

    # 1) Load base values from --config file if given
    config_path = getattr(args, "config", None)
    if config_path:
        from novacore.config import configure
        cfg_data = configure(config_path)
        defaults = cfg_data.get("defaults", {})
        for key in model_keys:
            if key in defaults and defaults[key] is not None:
                cfg[key] = defaults[key]
        print(f"[NovaCore] Loaded base config from: {config_path}")

    # 2) Manual flags override (only if explicitly set, not None)
    for key in model_keys:
        # each command may expose max_tokens under different attr names
        cli_val = getattr(args, key, None) if hasattr(args, key) else None
        if cli_val is None and hasattr(args, "tokens"):
            cli_val = getattr(args, "tokens", None)
        if cli_val is None and hasattr(args, "max_tokens"):
            cli_val = getattr(args, "max_tokens", None)
        if cli_val is not None:
            cfg[key] = cli_val
            provided[key] = cli_val

    # 3) Strict: if no config file and no manual model flags -> error
    if not config_path and not provided:
        raise SystemExit(
            "[NovaCore] ERROR: no model config provided.\n"
            "  Provide a config file with --config <config.json>, and/or manual flags\n"
            "  (--dim --layers --vocab-size --max-tokens --temperature).\n"
            "  No built-in defaults are used."
        )

    # 4) Validate required keys using config.json defaults only as last allowed fallback
    #    (config.json IS an explicit config choice; missing keys fall back to it)
    from novacore.config import get_default
    cfg.setdefault("dim", get_default("dim", None))
    cfg.setdefault("layers", get_default("layers", None))
    cfg.setdefault("vocab_size", get_default("vocab_size", None))
    cfg.setdefault("max_tokens", get_default("max_tokens", None))
    cfg.setdefault("temperature", get_default("temperature", None))

    missing = [k for k in model_keys if cfg.get(k) is None]
    if missing:
        raise SystemExit(
            f"[NovaCore] ERROR: missing required model config values: {missing}. "
            f"Set them via --config config.json or flags (--dim --layers --vocab-size --max-tokens --temperature)."
        )

    # vocab_size may legitimately be small; keep as int
    cfg["max_tokens"] = int(cfg["max_tokens"])
    return cfg


def _resolve_rows(max_rows):
    """Return (num_rows or None, is_full). 'full'/'all'/'-' => None (whole dataset)."""
    if max_rows is None:
        return None, False
    s = str(max_rows).strip().lower()
    if s in ("full", "all", "0", "-", "none"):
        return None, True
    try:
        return int(s), False
    except ValueError:
        raise SystemExit(f"[NovaCore] ERROR: --max-rows must be a number or 'full', got: '{max_rows}'")


def cmd_train(args):
    """Unified model creation/expansion from HF and/or local datasets."""
    import re

    # --cpu-only: block GPU acceleration (SVD etc.) for reproducibility
    if getattr(args, 'cpu_only', False):
        os.environ['NOVACORE_CPU_ONLY'] = '1'

    # Resolve strict config (--config file + manual flags, else error)
    cfg = _resolve_model_config(args)
    dim = cfg["dim"]
    layers = cfg["layers"]
    vocab_size = cfg["vocab_size"]
    max_tokens = cfg["max_tokens"]
    temperature = cfg["temperature"]

    # Output model dir: --load expands existing, else --output (new).
    # Bare names are resolved under the default weights dir (config 'default_weights').
    if args.load_model:
        out_dir = _resolve_model_dir(args.load_model)
    else:
        if not args.output:
            raise SystemExit("[NovaCore] ERROR: provide --output <model dir> for a new model, or --load <existing> to expand.")
        out_dir = _resolve_model_dir(args.output)

    is_new = not os.path.isdir(out_dir)
    if args.new_mode and args.load_model:
        raise SystemExit("[NovaCore] ERROR: use --new OR --load, not both.")

    # For HF, need owner/repo
    def _is_hf(ds):
        # 'owner/repo' style or http/w3 path = HF; local file path = local
        return "/" in ds and os.path.exists(ds) is False and not ds.startswith(".") and os.path.splitext(ds)[1] == ""

    # Determine rows mode (applies per dataset via index pairing, unless overridden in each spec)
    max_rows_list = args.max_rows or []

    # Collect pool streams (HF) and local text lists separately.
    # Pool streams are file-backed and don't consume RAM; local lists are
    # typically small enough to hold in memory.
    pool_streams = []     # [(path, count, label), ...]  – file-backed
    local_all_texts = []  # for small local datasets only
    total_stream_docs = 0
    sources = []

    loader_hf = HFLoader()

    # Pre-scan: quickly check which pools already exist and are complete.
    # This avoids loading UIDs for datasets that are already fully absorbed.
    _cache_loc = loader_hf._resolve_cache(None) or "data/hf_cache"
    _skip_pool = set()  # (dataset_name, name, split) tuples already complete

    # Expand --dataset specs: a config file with "datasets" key becomes multiple specs.
    # Multiple config files are supported (e.g. a universal data_config.json plus
    # per-dataset overrides like data_config.rajpurkar_squad.json).  Later files
    # override earlier ones (per-dataset wins over universal).
    expanded_specs = []
    _loaded_cfgs = []
    if args.dataset:
        for idx, spec in enumerate(args.dataset):
            if os.path.isfile(spec) and spec.endswith('.json'):
                try:
                    with open(spec, 'r', encoding='utf-8') as f:
                        raw = f.read()
                    # Use JSONC parser so config files may include // and /* */ comments
                    from novacore.config import _parse_jsonc
                    try:
                        ds_cfg = _parse_jsonc(raw)
                    except json.JSONDecodeError as e:
                        # Fall back to standard json for non-JSONC files
                        ds_cfg = json.loads(raw)
                    if isinstance(ds_cfg, dict) and "datasets" in ds_cfg:
                        # Merge hf defaults from this file (later wins)
                        if isinstance(ds_cfg.get("hf"), dict):
                            from novacore.config import load_data_config
                            base = load_data_config()
                            base_hf = dict(base.get("hf") or {})
                            base_hf.update(ds_cfg["hf"])
                            base["hf"] = base_hf
                            from novacore.config import _data_cached
                            # re-bind cache so get_hf returns the merged value
                            import novacore.config as _cfg_mod
                            _cfg_mod._data_cached = base
                        for name, dcfg in ds_cfg["datasets"].items():
                            # Multi-config datasets (e.g. UltraData Math/Code) expand to N specs
                            subs = (dcfg or {}).get("subset") or (dcfg or {}).get("configs")
                            if subs:
                                if isinstance(subs, str):
                                    subs = [subs]  # single subset given as string
                                for s in subs:
                                    if isinstance(s, dict):
                                        # { "name", "split", "max_rows" } form
                                        sub_name = s.get("name")
                                        entry = dict(dcfg)
                                        entry.update(dict(s))
                                    else:
                                        sub_name = s
                                        entry = dict(dcfg)
                                        entry["name"] = s
                                    expanded_specs.append((f"{name}##{sub_name}", entry, spec))
                            else:
                                expanded_specs.append((name, dcfg, spec))
                        _loaded_cfgs.append(spec)
                        continue
                except Exception:
                    pass
            expanded_specs.append((spec, {}, None))

    if _loaded_cfgs:
        print(f"[NovaCore] Loaded config(s): {', '.join(_loaded_cfgs)}  ->  {len(expanded_specs)} dataset specs")

    # Quick pre-scan: check pool existence for every expanded spec
    for spec, ds_cfg_from_file, cfg_path in expanded_specs:
        spec_name = spec
        embedded_config = None
        if "##" in spec:
            spec_name, embedded_config = spec.split("##", 1)
        ds_spec = spec_name
        if _is_hf(ds_spec):
            ds_name = ds_spec
            split = get_hf('default_split', 'train')
            if "::" in ds_spec:
                ds_name, split = ds_spec.split("::", 1)
            ds_cfg = get_dataset(ds_name) or {}
            if embedded_config:
                ds_cfg = get_dataset(f"{ds_name}##{embedded_config}") or ds_cfg
            if ds_cfg_from_file:
                if ds_cfg_from_file.get("split"):
                    split = ds_cfg_from_file["split"]
                ds_cfg = {**ds_cfg, **ds_cfg_from_file}
            else:
                if ds_cfg.get("split"):
                    split = ds_cfg["split"]
            if embedded_config:
                ds_cfg["name"] = embedded_config
            # Check if pool already exists and is complete
            _pdir = loader_hf._pool_dir(_cache_loc, ds_name, ds_cfg.get("name"), split)
            _meta_path = os.path.join(_pdir, "manifest.json")
            if os.path.isfile(_meta_path):
                try:
                    with open(_meta_path, encoding="utf-8") as _fh:
                        _meta = json.load(_fh)
                    _rows = _meta.get("rows", 0)
                    _complete = _meta.get("complete", False)
                    if _rows > 0 and _complete:
                        _skip_pool.add((ds_name, ds_cfg.get("name"), split))
                        print(f"[NovaCore]   pre-scan: pool COMPLETE for {ds_name} [{ds_cfg.get('name')}] split={split} ({_rows} rows) - will reuse", flush=True)
                    elif _rows > 0:
                        print(f"[NovaCore]   pre-scan: pool INCOMPLETE for {ds_name} [{ds_cfg.get('name')}] split={split} ({_rows} rows) - will resume build", flush=True)
                except Exception:
                    pass

    for idx, (spec, ds_cfg_from_file, cfg_path) in enumerate(expanded_specs):
        try:
            # Spec may carry an embedded config name: 'repo##config'
            spec_name = spec
            embedded_config = None
            if "##" in spec:
                spec_name, embedded_config = spec.split("##", 1)
            # Each spec may carry an optional per-dataset row hint: spec#rows or spec:rows
            ds_spec = spec_name
            per_rows, per_full = None, False
            if "#" in spec_name or re.search(r":\s*\d", spec_name):
                parts = re.split(r"[#:]", spec_name, maxsplit=1)
                ds_spec = parts[0]
                per_rows, per_full = _resolve_rows(parts[1])
            else:
                # also allow 'full' embedded
                m = re.match(r"^(.*?)[:=](full|all|\d+)$", spec_name, re.IGNORECASE)
                if m:
                    ds_spec = m.group(1)
                    per_rows, per_full = _resolve_rows(m.group(2))
            # Pair with matching --max-rows (same index) if not already resolved in spec
            if per_rows is None and max_rows_list:
                if idx < len(max_rows_list):
                    per_rows, per_full = _resolve_rows(max_rows_list[idx])
                    print(f"[NovaCore]   (dataset {idx+1} max-rows={max_rows_list[idx]})")

            # Config-file dataset entries may carry per-dataset settings
            if ds_cfg_from_file:
                if per_rows is None and ds_cfg_from_file.get("max_rows") is not None:
                    per_rows, per_full = _resolve_rows(ds_cfg_from_file["max_rows"])

            # Global hf-level max_rows fallback (single source of truth):
            # if no per-spec / CLI / per-dataset override, use hf.max_rows.
            if per_rows is None:
                _hf_max = get_hf("max_rows")
                if _hf_max is not None:
                    per_rows, per_full = _resolve_rows(_hf_max)

            if _is_hf(ds_spec):
                ds_name = ds_spec
                split = get_hf('default_split', 'train')
                if "::" in ds_spec:
                    ds_name, split = ds_spec.split("::", 1)
                # per-dataset config overrides (split/column already handled inside loader)
                ds_cfg = get_dataset(ds_name) or {}
                if embedded_config:
                    ds_cfg = get_dataset(f"{ds_name}##{embedded_config}") or ds_cfg
                # merge with config-file overrides (config-file takes precedence)
                if ds_cfg_from_file:
                    if ds_cfg_from_file.get("split"):
                        split = ds_cfg_from_file["split"]
                    ds_cfg = {**ds_cfg, **ds_cfg_from_file}
                else:
                    if ds_cfg.get("split"):
                        split = ds_cfg["split"]
                # spec may embed a config name as 'repo##config'
                if embedded_config:
                    ds_cfg["name"] = embedded_config
                if per_rows is None and ds_cfg.get("max_rows") is not None:
                    per_rows, per_full = _resolve_rows(ds_cfg["max_rows"])
                # use fields from config if present
                _fields = ds_cfg.get("fields") or ds_cfg_from_file.get("fields")
                _col = args.column or ds_cfg.get("column") or ds_cfg_from_file.get("column")
                _field_des = ds_cfg.get("fields") or {}  # always defined for SKIP path
                step = idx + 1
                _step_log = Logger()
                sub_label = f" {ds_cfg.get('name')}" if ds_cfg.get("name") else ""
                print()
                print(_step_log.header(f" Step {step}/{len(expanded_specs)} : HF {ds_name}{sub_label} "))

                # Pre-scan skip: if pool already complete, skip the download/parse
                if (ds_name, ds_cfg.get("name"), split) in _skip_pool:
                    print(f"[NovaCore]   SKIP: pool already complete for {ds_name} [{ds_cfg.get('name')}] - reusing without re-download", flush=True)
                    _pdir = loader_hf._pool_dir(_cache_loc, ds_name, ds_cfg.get("name"), split)
                    _rows_path = os.path.join(_pdir, "pool.jsonl")
                    _meta = {}
                    try:
                        with open(os.path.join(_pdir, "manifest.json"), encoding="utf-8") as _fh:
                            _meta = json.load(_fh)
                    except Exception:
                        pass
                    _pcount = _meta.get("rows", 0)
                    if os.path.isfile(_rows_path) and _pcount > 0:
                        _filter_cfg = ds_cfg.get("filter") or ds_cfg_from_file.get("filter")
                        pool_streams.append((_rows_path, _pcount, ds_name, _filter_cfg))
                        total_stream_docs += _pcount
                        sources.append(f"hf:{ds_name}")
                        if _field_des:
                            _roles = ", ".join(f"{c}->{r}" for c, r in _field_des.items())
                            _step_log.ok(f"pooled {_pcount} row(s) on fields [{_roles}]")
                        _step_log.ok(f"+{_pcount} docs ready from existing pool  ({_step_log.total()})")
                    continue

                streaming = False
                random_sample = False
                # Resolve mode: CLI --mode > per-dataset config > hf default > ERROR.
                # The hf block is the single source of truth for shared flags;
                # a per-dataset entry may override the global default.
                _mode = args.mode or ds_cfg.get("mode") or ds_cfg_from_file.get("mode")
                if _mode is None:
                    _mode = get_hf("mode")
                if _mode is None:
                    raise SystemExit(
                        f"[NovaCore] ERROR: no mode specified for dataset '{ds_name}'"
                        f"{' (config=' + ds_cfg.get('name') + ')' if ds_cfg.get('name') else ''}.\n"
                        f"  Add \"mode\": \"download\" or \"mode\": \"stream\" in "
                        f"config/data_config.json (either under 'hf' for global, or "
                        f"under 'datasets.{ds_name}' for per-dataset) OR pass --mode download/stream."
                    )
                if _mode == "stream":
                    streaming = True
                    random_sample = False
                    _add = False
                    _absorb = False
                    _uid = ds_cfg.get("uid_key") or get_hf("uid_key")
                else:
                    # "download" -> ALWAYS materialize (fetch raw shards to disk
                    # + build the accumulated pool).  random / add_datasets /
                    # absorb come from CLI > per-dataset config > hf defaults
                    # in config/data_config.json (Rule 4).  Per-dataset entries
                    # may override any hf-level default; the hf block is the
                    # single source of truth for shared flags.
                    streaming = False
                    random_sample = args.random_flag
                    if random_sample is None:
                        random_sample = ds_cfg.get("random")
                    if random_sample is None:
                        random_sample = get_hf("random", False)
                    _add = args.add_flag
                    if _add is None:
                        _add = ds_cfg.get("add_datasets")
                    if _add is None:
                        _add = get_hf("add_datasets", False)
                    # absorb: CLI > per-dataset config > hf default
                    _absorb = getattr(args, "absorb_flag", None)
                    if _absorb is None:
                        _absorb = ds_cfg.get("absorb")
                    if _absorb is None:
                        _absorb = get_hf("absorb", False)
                    if _absorb:
                        _add = True
                        per_rows = None
                    _uid = ds_cfg.get("uid_key") or get_hf("uid_key")
                _cache_loc = loader_hf._resolve_cache(None) or "data/hf_cache"
                print(f"[NovaCore] mode : {_mode.upper()}   rows : {per_rows or 'full'}   random : {random_sample}   add_datasets : {_add}   absorb : {_absorb}" + (f"   uid_key : {_uid}" if _uid else ""))
                print(f"[NovaCore] cache: {_cache_loc}")
                texts = loader_hf.load(
                    ds_name, split=split, streaming=streaming,
                    text_column=_col, num_rows=per_rows, fields=_fields,
                    name=ds_cfg.get("name"), random_sample=random_sample,
                    add_datasets=_add, uid_key=_uid, local_only=_absorb,
                )
                sources.append(f"hf:{ds_name}")
                # Schema summary
                _schema = loader_hf.last_schema or {}
                _field_des = _schema.get("fields") or {}
                # Determine row count for logging (supports TextStream and list)
                _row_count = loader_hf.last_pool_count or (
                    len(texts) if hasattr(texts, '__len__') else 0
                )
                if not _row_count and hasattr(texts, '__len__'):
                    _row_count = len(texts)
                if _field_des:
                    _roles = ", ".join(f"{c}->{r}" for c, r in _field_des.items())
                    _step_log.ok(f"pooled {_row_count} row(s) on fields [{_roles}]")
                elif _schema.get("column"):
                    _step_log.ok(f"pooled {_row_count} row(s) on column [{_schema['column']}]")
                else:
                    _step_log.ok(f"pooled {_row_count} row(s)")
                if getattr(loader_hf, "_skipped", 0):
                    _step_log.skip(f"{loader_hf._skipped} empty/invalid row(s) not trained")
                if _row_count:
                    _step_log.ok(f"+{_row_count} docs added to training pool  ({_step_log.total()})")
                # Collect pool stream for streaming training (no RAM for texts)
                _ppath = getattr(loader_hf, 'last_pool_path', None)
                _pcount = getattr(loader_hf, 'last_pool_count', 0)
                if _ppath and os.path.isfile(_ppath) and _pcount:
                    _filter_cfg = ds_cfg.get("filter") or ds_cfg_from_file.get("filter")
                    pool_streams.append((_ppath, _pcount, ds_name, _filter_cfg))
                    total_stream_docs += _pcount
                elif texts and hasattr(texts, '__iter__') and not _ppath:
                    # Fallback: streaming iterable (not pooled) — write to disk pool
                    # to avoid RAM blow-up on large datasets (e.g. TinyStories 3.3M rows).
                    import tempfile, json as _json
                    _pool_dir = os.path.join("data", "hf_cache", "pool")
                    os.makedirs(_pool_dir, exist_ok=True)
                    _safe_name = ds_name.replace("/", "_").replace("\\", "_")
                    _stream_pool = os.path.join(_pool_dir, f"{_safe_name}_stream.jsonl")
                    _stream_count = 0
                    with open(_stream_pool, "w", encoding="utf-8") as _sf:
                        for _row in texts:
                            if _row:
                                _sf.write(_json.dumps({"text": _row}, ensure_ascii=False) + "\n")
                                _stream_count += 1
                                if _stream_count % 50000 == 0:
                                    print(f"    · wrote {_stream_count} rows to disk pool...", flush=True)
                    print(f"    · {_safe_name}: {_stream_count} rows -> disk pool ({_stream_pool})")
                    if _stream_count:
                        pool_streams.append((_stream_pool, _stream_count, ds_name, None))
                        total_stream_docs += _stream_count
                    del texts
            else:
                # local file
                print(f"[NovaCore] Loading local dataset: {ds_spec} (max-rows={per_rows or 'full'})")
                texts, meta = DatasetLoader.load(ds_spec, None, args.column)
                if per_rows is not None:
                    if per_rows < len(texts):
                        texts = random.sample(texts, per_rows)
                sources.append(f"local:{os.path.basename(ds_spec)}")
                print(f"[NovaCore]   -> +{len(texts)} docs")
                if texts:
                    local_all_texts.extend(texts)
                    total_stream_docs += len(texts)
        except SystemExit:
            raise  # re-raise intentional exits
        except Exception as _step_err:
            import traceback
            print(f"\n[NovaCore] ⚠ ERROR in Step {idx + 1}/{len(expanded_specs)}: {spec}", flush=True)
            print(f"[NovaCore] ⚠ {type(_step_err).__name__}: {_step_err}", flush=True)
            traceback.print_exc()
            print(f"[NovaCore] ⚠ Skipping this dataset and continuing...\n", flush=True)
            continue
        # Free memory between steps — critical for 12 GB RAM machines.
        # HF dataset objects, row dicts, and intermediate lists linger until
        # gc.collect() runs; explicit calls here prevent cumulative OOM.
        import gc
        gc.collect()

    if not pool_streams and not local_all_texts:
        raise SystemExit("[NovaCore] ERROR: no documents loaded from any dataset.")

    _agg = Logger()
    print()
    print(_agg.header(" TRAINING POOL "))
    _agg.ok(f"{total_stream_docs} document(s) total  (source: {', '.join(sources)})")

    # --- Encode / expand --------------------------------------------------
    # Use streaming training when pool files are available (most of the data
    # stays on disk).  Small local datasets are chained into the same stream.
    import gc
    gc.collect()  # free memory before training

    # RAM-safe config overrides (from auto-tuner on low-RAM machines)
    # Read once; apply only if total RAM < 16 GB as detected by auto_tuner
    try:
        from novacore.auto_tuner import get_tuner
        tuner = get_tuner()
        ram_safe = tuner.total_ram < (16 * 1024**3)
        if ram_safe:
            # Load ram_safe_overrides from config if present
            try:
                import json
                with open("config/config.json", "r", encoding="utf-8") as _f:
                    _cfg_data = json.load(_f)
                _overrides = _cfg_data.get("ram_safe_overrides", {})
                _overrides_applied = []
                for _key, _val in _overrides.items():
                    if _key in cfg:
                        old = cfg[_key]
                        cfg[_key] = _val
                        _overrides_applied.append(f"{_key}:{old}->{_val}")
                    else:
                        cfg[_key] = _val
                        _overrides_applied.append(f"{_key}:{_val}")
                if _overrides_applied:
                    print(f"[NovaCore] RAM-SAFE: applying overrides (RAM={round(tuner.total_ram/1024**3,1)}GB < 16GB): {', '.join(_overrides_applied)}", flush=True)
                # Update local vars
                dim = cfg["dim"]
                layers = cfg["layers"]
                vocab_size = cfg["vocab_size"]
            except Exception:
                pass
    except Exception:
        pass

    if pool_streams:
        # Chain: pool file streams first, then any local texts
        # Each pool stream may have a per-dataset quality filter config.
        def _chain_streams():
            from novacore.dataset.hf_loader import TextStream
            from novacore.dataset.quality import FilteredTextStream
            for ppath, pcount, plabel, pfilter in pool_streams:
                ts = TextStream(ppath, count=pcount)
                if pfilter and pfilter.get("enabled") is True:
                    ts = FilteredTextStream(ts, pfilter)
                    print(f"[NovaCore]   quality filter active for {plabel}", flush=True)
                yield from ts
            for t in local_all_texts:
                yield t

        _train_from_stream(
            _chain_streams(), out_dir, dim, layers, vocab_size, max_tokens,
            temperature, source="+".join(sources),
            merge_existing=bool(args.load_model),
            workers=args.workers,
        )
    else:
        # All local (small) – use the list-based path
        _encode_texts_to_weights(
            local_all_texts, out_dir, dim, layers, vocab_size, max_tokens,
            temperature, source="+".join(sources),
            merge_existing=bool(args.load_model),
        )

    print(f"\n[NovaCore] Model ready: {out_dir}")
    print(f"[NovaCore] Config: dim={dim} layers={layers} vocab={vocab_size} temp={temperature} max_tokens={max_tokens}")
    print(f"[NovaCore] To chat: novacore chat --weights {out_dir}")


def cmd_encode(args):
    log = Logger()
    print(f"[NovaCore] Loading dataset: {args.input}")
    texts, meta = DatasetLoader.load(args.input, args.format, args.text_column)
    print(f"[NovaCore] Loaded {len(texts)} documents")

    # Build vocabulary
    print("[NovaCore] Building vocabulary...")
    _vocab_size = get_default('vocab_size')
    _vocab_cap = get_default('pattern_vocab_cap')
    vocab = Vocabulary(vocab_size=_vocab_size)
    vocab.build(texts[:min(len(texts), _vocab_cap)])
    print(f"[NovaCore] Vocabulary size: {len(vocab)}")

    processor = TextProcessor(vocab=vocab, dim=args.dim)
    _depth = get_default('hasher_depth')
    _buckets = args.dim * get_default('hasher_buckets_multiplier')
    hasher = FeatureHasher(num_buckets=_buckets, depth=_depth)

    # Build pattern extractor
    print("[NovaCore] Extracting patterns (n-grams)...")
    _ngram = tuple(get_default('ngram_range'))
    _pat_cap = get_default('pattern_sample_cap')
    extractor = PatternExtractor(ngram_range=_ngram)
    extractor.update_many(texts[:min(len(texts), _pat_cap)], log=log)
    extractor.prune()
    print(f"[NovaCore] Patterns: {len(extractor.patterns)}")

    # Reservoir sampling for large sets
    print("[NovaCore] Sampling...")
    _res_k = get_default('reservoir_sample_size')
    sampler = ReservoirSampler(k=min(_res_k, len(texts)))
    sampler.add_many(texts, log=log)

    # Build analytic model on sample
    print("[NovaCore] Computing analytic weights...")
    sample = sampler.sample()
    if sample:
        X = np.array([processor.text_to_vector(t) for t in tqdm(sample, desc="Encoding")])
        # Target: predict vocabulary co-occurrence (bag-of-labels)
        _tgt = get_default('analytic_target_dim')
        Y = np.zeros((X.shape[0], min(len(vocab), _tgt)), dtype=np.float64)
        for i, t in enumerate(sample):
            ids = vocab.encode(t)
            for j, tid in enumerate(ids[:min(len(ids), _tgt)]):
                if tid < Y.shape[1]:
                    Y[i, tid] = 1.0
        input_dim = X.shape[1]
        encoder = RandomFourierEncoder(input_dim=input_dim, dim=args.dim, layers=args.layers)
        encoder.fit_analytic(X, Y)
        print("[NovaCore] Analytic weights computed.")

    # Output dir
    out_dir = args.output or os.path.join(
        DEFAULT_WEIGHTS_DIR,
        os.path.splitext(os.path.basename(args.input))[0]
    )

    # Save
    print("[NovaCore] Saving weights...")
    wm = WeightManager(out_dir)
    arrays = {}
    if "encoder" in locals() and encoder is not None:
        w = encoder.save_weights()
        for k, v in w.items():
            arrays[f"encoder_{k}"] = np.asarray(v)
    arrays["pattern_vec"] = extractor.encode(args.dim)
    arrays["hasher"] = hasher.vector()

    # Store raw patterns for generation
    import pickle
    patterns_bytes = pickle.dumps(dict(extractor.patterns))
    arrays["patterns_raw"] = np.frombuffer(patterns_bytes, dtype=np.uint8)

    metadata = {
        "name": os.path.basename(out_dir),
        "created": str(np.datetime64('now')),
        "num_documents": len(texts),
        "dim": args.dim,
        "layers": args.layers,
        "vocab_size": len(vocab),
        "input_format": (meta or {}).get("format"),
        "has_vocab": "encoder_beta" in arrays,
        "dataset_size_bytes": sum(len(t.encode('utf-8')) for t in texts[:10000]),
    }
    wm.save(arrays, metadata)

    # Save vocab separately
    vocab_path = os.path.join(out_dir, "vocab.json")
    vocab.save(vocab_path)

    # Report sizes
    info = wm.info()
    print("\n[NovaCore] Done!")
    print(f"[NovaCore] Saved to: {out_dir}")
    print(f"[NovaCore] Weight file size: {info.get('total_size_mb', 0)} MB")
    print(f"[NovaCore] Documents: {len(texts)}")
    print(f"[NovaCore] Vocabulary: {len(vocab)}")


def cmd_generate(args):
    data = _load_model(args.weights)
    if data is None:
        return
    processor, vocab, hasher = data["processor"], data["vocab"], data["hasher"]

    # Use pattern-based generation from saved patterns
    import pickle
    _patterns = {}
    if "patterns_raw" in data["data"].files:
        raw = data["data"]["patterns_raw"]
        try:
            _patterns = pickle.loads(raw.tobytes())
        except Exception:
            _patterns = {}
    extractor = PatternExtractor()
    extractor.patterns = dict(_patterns)
    predictor = PatternPredictor(extractor, vocab, processor)
    metadata = data.get("metadata") or {}
    n_tokens = args.tokens if args.tokens is not None else metadata.get("max_tokens") or get_default('max_tokens')
    temp = args.temperature if args.temperature is not None else metadata.get("temperature") or get_default('temperature')
    print(f"\n[NovaCore] Generating from: '{args.prompt}'")
    print("-" * 50)
    result = predictor.generate(args.prompt, n_tokens, temp)
    print(result)
    print("-" * 50)


def _load_model(weights_dir):
    wm = WeightManager(weights_dir)
    try:
        data, metadata = wm.load()
    except FileNotFoundError as e:
        print(f"[NovaCore] Error: {e}")
        return None

    # Load vocab
    vocab_path = os.path.join(weights_dir, "vocab.json")
    if os.path.exists(vocab_path):
        try:
            vocab = Vocabulary.load(vocab_path)
        except Exception:
            vocab = Vocabulary()
    else:
        vocab = Vocabulary()

    dim = (metadata or {}).get("dim") or get_default('dim')
    processor = TextProcessor(vocab=vocab, dim=dim)

    # Rebuild hasher from saved vector if present
    """Rebuild hasher from saved vector if present"""
    _depth = get_default('hasher_depth')
    hasher = FeatureHasher(num_buckets=dim * get_default('hasher_buckets_multiplier'), depth=_depth)
    hv = data["hasher"] if "hasher" in data.files else None
    if hv is not None:
        hasher.counts = hv.reshape(hasher.counts.shape)

    return {
        "processor": processor,
        "vocab": vocab,
        "hasher": hasher,
        "metadata": metadata,
        "data": data,
    }


def cmd_info(args):
    wm = WeightManager(args.weights)
    try:
        info = wm.info()
    except FileNotFoundError as e:
        print(f"[NovaCore] Error: {e}")
        return
    print(f"[NovaCore] Model: {info.get('name', 'unknown')}")
    print(f"[NovaCore] Created: {info.get('created', 'unknown')}")
    print(f"[NovaCore] Documents: {info.get('num_documents', 0)}")
    print(f"[NovaCore] Dimension: {info.get('dim', 'unknown')}")
    print(f"[NovaCore] Layers: {info.get('layers', 'unknown')}")
    print(f"[NovaCore] Vocab size: {info.get('vocab_size', 'unknown')}")
    print(f"[NovaCore] Input format: {info.get('input_format', 'unknown')}")
    print(f"[NovaCore] Weight file size: {info.get('total_size_mb', 0)} MB")
    print("[NovaCore] Arrays:")
    for key, val in info.get('arrays', {}).items():
        print(f"  - {key}: shape={val['shape']} dtype={val['dtype']} size={val['mb']} MB")


def cmd_compress(args):
    wm = WeightManager(args.weights)
    try:
        wm.info()
    except FileNotFoundError as e:
        print(f"[NovaCore] Error: {e}")
        return
    wm.compress(args.ratio)
    print(f"[NovaCore] Compressed {args.weights} (ratio={args.ratio})")


def cmd_query(args):
    data = _load_model(args.weights)
    if data is None:
        return
    processor = data["processor"]
    vec = processor.text_to_vector(args.text)
    norm = np.linalg.norm(vec)
    print(f"[NovaCore] Query vector norm: {norm:.4f}")
    print(f"[NovaCore] Query '{args.text}' processed (dim={vec.shape[0]})")
    print("[NovaCore] Note: Full similarity search requires stored corpus embeddings.")


def cmd_validate(args):
    wm = WeightManager(args.weights)
    try:
        info = wm.info()
    except FileNotFoundError as e:
        print(f"[NovaCore] Error: {e}")
        return
    ok = True
    for key, val in info.get('arrays', {}).items():
        shape = val['shape']
        # Skip scalar metadata arrays (0-dim) and small int metadata
        if len(shape) == 0:
            continue
        size = 1
        for s in shape:
            size *= s
        if size < 4:
            continue
        if np.issubdtype(np.dtype(val['dtype']), np.floating):
            continue
        if size == 0:
            print(f"[NovaCore] WARNING: empty array '{key}'")
            ok = False
    print(f"[NovaCore] Validation {'PASSED' if ok else 'FAILED'}")
    print(f"[NovaCore] {len(info.get('arrays', {}))} arrays, {info.get('total_size_mb', 0)} MB")


def cmd_list(args):
    models = WeightManager.list_models(args.dir)
    if not models:
        print(f"[NovaCore] No models found in {args.dir}")
        return
    print(f"[NovaCore] Models in {args.dir}:")
    for m in models:
        wm = WeightManager(os.path.join(args.dir, m))
        try:
            info = wm.info()
            size = info.get('total_size_mb', 0)
            docs = info.get('num_documents', '?')
            print(f"  - {m}: {size} MB, {docs} docs")
        except Exception:
            print(f"  - {m}: (unreadable)")


def _chat_banner(session, weights_dir):
    """Print the chat banner for the currently loaded model."""
    info = session.info()
    model_name = info.get('name', os.path.basename(weights_dir))
    num_docs = info.get('num_documents', '?')
    try:
        wm_size = WeightManager(weights_dir).info()
        size_mb = wm_size.get('total_size_mb', 0)
        fmt = wm_size.get('format', 'ncw')
    except Exception:
        size_mb = 0
        fmt = 'ncw'
    print("\n" + "=" * 60)
    print(f"  NovaCore Chat  |  Model: {model_name}")
    print(f"  Docs: {num_docs}   Weights: {size_mb} MB   Dim: {info.get('dim', '?')}   Layers: {info.get('layers', '?')}")
    print(f"  Format: {fmt}   Source: {info.get('source', info.get('input_format', 'local'))}")
    print("=" * 60)
    print("  Commands: /help  /models  /load <name>  /new  /info  /exit")
    print("-" * 60)


def _list_available_models(base_dir):
    """Return dict {name: full_path} of available models."""
    models = WeightManager.list_models(base_dir)
    return {m: os.path.join(base_dir, m) for m in models}


def cmd_convert(args):
    """Convert legacy models to NovaCore .ncw format."""
    if args.convert_all:
        models = WeightManager.list_models(args.dir)
        if not models:
            print(f"[NovaCore] No models found in {args.dir}")
            return
        for m in models:
            path = os.path.join(args.dir, m)
            _convert_one(path, m)
        return
    if not args.weights:
        print("[NovaCore] Specify --weights <dir> or use --all to convert everything")
        print(f"[NovaCore]   models are in: {DEFAULT_WEIGHTS_DIR}")
        return
    _convert_one(args.weights, os.path.basename(args.weights))


def _convert_one(path, name):
    """Convert a single model dir to native ncw format."""
    if WeightManager.is_native(path):
        print(f"[NovaCore] '{name}' already in .ncw format")
        return
    wm = WeightManager(path)
    try:
        converted = wm.migrate()
    except Exception as e:
        print(f"[NovaCore] Error converting '{name}': {e}")
        return
    if converted:
        print(f"[NovaCore] Converted '{name}' -> .ncw (weights.ncw + index.ncmeta)")
    else:
        print(f"[NovaCore] '{name}': nothing to convert")


def cmd_chat(args):
    """Interactive chat with a saved model - like Ollama (supports model switching)."""
    # Resolve temperature from model config if not given on CLI
    _meta_temp = None
    _meta_tokens = None
    try:
        _wm = WeightManager(args.weights)
        _d, _m = _wm.load()
        _meta_temp = (_m or {}).get("temperature")
        _meta_tokens = (_m or {}).get("max_tokens")
    except Exception:
        pass

    def _start_session(weights_dir):
        try:
            s = ChatSession(weights_dir, temperature=args.temperature if args.temperature is not None else _meta_temp)
            if hasattr(args, 'cortex') and args.cortex:
                s.enable_cortex = True
            if hasattr(args, 'cde') and args.cde:
                s.enable_cde = True
            s.load()
            return s
        except RuntimeError as e:
            print(f"[NovaCore] Error: {e}")
            return None

    session = _start_session(args.weights)
    if session is None:
        return
    current_weights = args.weights

    _chat_banner(session, current_weights)

    if args.system:
        session.history.append({"role": "system", "content": args.system})

    def _run(prompt, num_tokens):
        if num_tokens is None:
            num_tokens = ((session.metadata or {}).get("max_tokens")
                          or _meta_tokens or get_default('max_tokens'))
        try:
            return session.chat(prompt, max_tokens=num_tokens, verbose=args.verbose)
        except Exception as e:
            return f"[Error: {e}]"

    # One-shot mode
    if args.prompt:
        reply = _run(args.prompt, args.tokens)
        print(f"You: {args.prompt}")
        print(f"\033[92mNovaCore> {reply}\033[0m")
        return

    # Interactive REPL
    print("  (type your message, /models to list, /load <name> to switch, /exit to quit)")
    while True:
        try:
            nline = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[NovaCore] Goodbye!")
            break
        if not nline:
            continue
        if nline in ("/exit", "/quit", "/q"):
            print("[NovaCore] Goodbye!")
            break
        elif nline == "/new":
            session.reset()
            print("[NovaCore] Conversation reset.")
            continue
        elif nline == "/info":
            for k, v in session.info().items():
                print(f"  {k}: {v}")
            continue
        elif nline in ("/models", "/list"):
            avail = _list_available_models(DEFAULT_WEIGHTS_DIR)
            if not avail:
                print(f"[NovaCore] No models found in {DEFAULT_WEIGHTS_DIR}")
                print(f"[NovaCore]   (or use: /load <full_path>)")
            else:
                print("[NovaCore] Available models:")
                for name, path in sorted(avail.items()):
                    marker = "  <-- current" if os.path.normpath(path) == os.path.normpath(current_weights) else ""
                    try:
                        wm = WeightManager(path).info()
                        size = wm.get('total_size_mb', 0)
                        docs = wm.get('num_documents', '?')
                        line = f"  - {name}: {size} MB, {docs} docs{marker}"
                    except Exception:
                        line = f"  - {name}{marker}"
                    print(line)
                print("[NovaCore] Load: /load <model_name>")
            continue
        elif nline.startswith("/load") or nline.startswith("/use"):
            parts = nline.split(None, 1)
            target = parts[1].strip() if len(parts) > 1 else ""
            if not target:
                print("[NovaCore] Usage: /load <model_name>")
                continue
            # Resolve relative to default weights dir
            candidate = target
            if not os.path.isabs(candidate) and not os.path.exists(candidate):
                cand_in_default = os.path.join(DEFAULT_WEIGHTS_DIR, target)
                if os.path.exists(os.path.join(cand_in_default, WeightManager.WEIGHTS_FILE)) or \
                   os.path.exists(os.path.join(cand_in_default, WeightManager.LEGACY_WEIGHTS_FILE)):
                    candidate = cand_in_default
            if not os.path.exists(candidate):
                print(f"[NovaCore] Model not found: {target}")
                print(f"[NovaCore]   Try /models to list, or full path")
                continue
            new_session = _start_session(candidate)
            if new_session is None:
                continue
            session = new_session
            current_weights = candidate
            session.reset()
            print(f"[NovaCore] Switched to model ->")
            _chat_banner(session, current_weights)
            continue
        elif nline == "/help":
            print("  /models      - list available models")
            print("  /load <name> - switch to another model (also /use)")
            print("  /new         - reset conversation")
            print("  /info        - show current model info")
            print("  /exit        - quit")
            continue

        reply = _run(nline, args.tokens)
        print(f"\n\033[92mNovaCore> {reply}\033[0m")


def cmd_hf(args):
    """Hugging Face operations dispatcher."""
    loader = HFLoader()
    hf_cmd = args.hf_command

    if hf_cmd == "login":
        user = loader.login(args.token)
        print("[NovaCore] HF login successful.")
        if user:
            print(f"[NovaCore] Logged in as: {user.get('name') or user.get('fullname', 'unknown')}")
            print(f"[NovaCore] Email: {user.get('email', 'N/A')}")

    elif hf_cmd == "logout":
        loader.logout()
        print("[NovaCore] HF logout successful.")

    elif hf_cmd == "status":
        status = loader.auth_status()
        if status["logged_in"]:
            u = status["user"]
            print(f"[NovaCore] Logged in as: {u.get('name') or u.get('fullname', 'unknown')}")
        else:
            print("[NovaCore] Not logged in. Run: novacore hf login --token YOUR_TOKEN")
            print("[NovaCore]   (token from https://huggingface.co/settings/tokens)")

    elif hf_cmd == "run":
        _cmd_hf_run(args, loader)

    elif hf_cmd == "stream":
        _cmd_hf_stream(args, loader)


def _cmd_hf_run(args, loader):
    """Load a HF dataset and encode it into a NovaCore model (config-based)."""
    dataset = args.dataset
    if "/" not in dataset:
        print(f"[NovaCore] Error: dataset must include namespace/owner, e.g. 'rajpurkar/squad'.")
        print(f"[NovaCore] Got: '{dataset}'. Find full IDs on huggingface.co/datasets")
        return

    # Strict config: --config file + manual flags, else error
    cfg = _resolve_model_config(args)
    dim = cfg["dim"]
    layers = cfg["layers"]

    col = args.column or args.text_column
    ds_cfg = get_dataset(dataset) or {}
    split = args.split or ds_cfg.get("split") or get_hf('default_split', 'train')
    print(f"[NovaCore] HF: loading '{dataset}' (split={split}) streaming={args.stream}")
    texts = loader.load(
        dataset, split=split, cache_dir=args.cache,
        streaming=args.stream, text_column=col, num_rows=args.max_rows,
        name=args.name or ds_cfg.get('name') or ds_cfg.get('subset'),
    )

    # Convert generator to list if streaming (pull in memory)
    if args.stream or args.max_rows is not None:
        texts = list(texts)

    # Schema report (already printed by loader, show phase summary)
    if loader.last_schema and (loader.last_schema.get("fields") or loader.last_schema.get("column")):
        print(f"[NovaCore] Schema: {dataset} trained on fields {loader.last_schema['fields'] or loader.last_schema['column']}")

    print(f"[NovaCore] Got {len(texts)} documents from HF")

    if not texts:
        print("[NovaCore] No documents found. Check dataset/split/column.")
        return

    # Model name + output dir (bare names resolved under weights/)
    model_name = args.output or f"hf_{dataset.replace('/', '_')}"
    out_dir = _resolve_model_dir(args.output) if args.output else os.path.join(DEFAULT_WEIGHTS_DIR, model_name)

    _encode_texts_to_weights(
        texts, out_dir, dim, layers, cfg["vocab_size"],
        cfg["max_tokens"], cfg["temperature"], source=f"hf:{dataset}", column=col,
    )

    print(f"\n[NovaCore] Model saved to: {out_dir}")
    print(f"[NovaCore] Config: dim={dim} layers={layers} temp={cfg['temperature']} max_tokens={cfg['max_tokens']}")
    print(f"[NovaCore] To chat: novacore chat --weights {out_dir}")


def _cmd_hf_stream(args, loader):
    """Preview / save a HF dataset without full encode."""
    dataset = args.dataset
    if "/" not in dataset:
        print(f"[NovaCore] Error: dataset must include namespace/owner, e.g. 'rajpurkar/squad'.")
        return
    ds_cfg = get_dataset(dataset) or {}
    split = args.split or ds_cfg.get("split") or get_hf('default_split', 'train')
    print(f"[NovaCore] Streaming '{dataset}' (split={split})...")
    texts = loader.stream(dataset, split=split, text_column=args.column or ds_cfg.get("column"), num_rows=args.rows)
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            for i, t in enumerate(texts):
                f.write(t.replace("\n", " ") + "\n")
        print(f"[NovaCore] Saved {args.rows or '?'} rows to {args.save}")
        return
    print(f"[NovaCore] Preview (first {args.rows} rows):")
    for i, t in enumerate(texts):
        print(f"  [{i+1}] {t[:100]}")


_REASONING_FIELDS = ('causal_chains', 'logical_patterns', 'comparison_chains',
                     'explanation_chains', 'problem_solutions', 'analogies')
_CREATIVE_FIELDS = ('story_openings', 'story_middles', 'story_endings',
                    'poem_structures', 'metaphors', 'descriptions',
                    'dialogue_patterns', 'emotional_arcs')


def _extract_docs_worker(job):
    """Worker: (vocab_docs, pat_docs, ngram) -> (vocab_freqs, pattern_counts).
    Pure per-document extraction; driver merges Counters (identical result to
    the sequential loop, but computed across cores)."""
    from collections import Counter as _Counter
    vocab_docs, pat_docs, ngram = job
    freq = _Counter()
    pat = _Counter()
    if vocab_docs:
        from novacore.tokenizer.vocab import Vocabulary
        _vocab = Vocabulary(vocab_size=50000)
        for d in vocab_docs:
            for t in _vocab._tokenize(d):
                freq[t] += 1
    if pat_docs:
        from novacore.core.patterns import PatternExtractor
        ext = PatternExtractor(ngram_range=tuple(ngram))
        for d in pat_docs:
            ext.update(d)
        pat = ext.patterns
    return freq, pat


def _extract_reasoning_worker(docs):
    """Worker: extract reasoning patterns from a doc slice -> mergeable lists."""
    from novacore.core.reasoning_engine import ReasoningPatternExtractor
    r = ReasoningPatternExtractor()
    for d in docs:
        r.extract_from_text(d)
    return {k: getattr(r, k) for k in _REASONING_FIELDS}


def _extract_creative_worker(docs):
    """Worker: extract creative patterns from a doc slice -> mergeable lists."""
    from novacore.core.creative_engine import CreativePatternExtractor
    c = CreativePatternExtractor()
    for d in docs:
        c.extract_from_text(d)
    return {k: getattr(c, k) for k in _CREATIVE_FIELDS}


def _train_from_stream(text_iter, out_dir, dim, layers, vocab_size, max_tokens,
                       temperature, source="", column=None, merge_existing=False,
                       workers=None):
    """Single-pass streaming training – memory-efficient for millions of docs.

    Vocab building, n-gram pattern extraction, and reservoir sampling all
    happen in a **single iteration** over *text_iter* so at most ~80 K
    texts (the reservoir) are ever held in RAM simultaneously.
    """
    from novacore.config import get_default as _gd
    from tqdm import tqdm
    import pickle
    import gc
    import re

    # Reproducible builds: global numpy/random must be seeded (training_upgrades
    # uses np.random.choice / np.random.randn outside its own RandomState).
    _seed = int(_gd('reservoir_seed') or _gd('seed') or 42)
    np.random.seed(_seed)
    random.seed(_seed)
    del _seed

    log = Logger()

    from novacore.auto_tuner import get_tuner
    tuner = get_tuner()
    tuner.apply_threading()
    # Worker processes: auto (~90% cores) unless CLI --workers set.
    try:
        _workers = int(workers) if workers else tuner.workers
    except Exception:
        _workers = tuner.workers
    if _workers < 1:
        _workers = 1
    print(f"[NovaCore] HW: CPU={tuner.cpu_count} RAM={round(tuner.total_ram/1024**3,1)}GB "
          f"GPU={'YES ('+tuner.gpu_name+')' if tuner.gpu_available else 'no'} "
          f"threads={tuner.num_threads} workers={_workers}", flush=True)
    # Auto-tune config knobs based on live hardware
    if _gd('pattern_sample_cap') is not None:
        _pat_cap = min(_gd('pattern_sample_cap'), tuner.pattern_sample_cap)
    else:
        _pat_cap = tuner.pattern_sample_cap
    if _gd('pattern_vocab_cap') is not None:
        _vocab_cap = min(_gd('pattern_vocab_cap'), tuner.vocab_build_cap)
    else:
        _vocab_cap = tuner.vocab_build_cap
    _res_k = min(_gd('reservoir_sample_size'), tuner.reservoir_size)
    # Memory guard: if >90% used, reduce caps by half
    if tuner.mem_pressure() > 0.90:
        print(f"[NovaCore] AUTO-REDUCE: memory pressure {tuner.mem_pressure()*100:.0f}% > 90%; halving caps")
        _vocab_cap = max(5000, _vocab_cap // 2)
        _pat_cap = max(5000, _pat_cap // 2)
        _res_k = max(10000, _res_k // 2)
    _res_mult = _gd('reservoir_merge_multiplier')
    _depth = _gd('hasher_depth')
    _buckets_mult = _gd('hasher_buckets_multiplier')
    _ngram = tuple(_gd('ngram_range'))
    _tgt = _gd('analytic_target_dim')

    is_expand = merge_existing and os.path.isdir(out_dir)
    wm = WeightManager(out_dir)

    # --- Load existing model if expanding ---
    prev_vocab = None
    prev_extractor = None
    prev_meta = {}
    if is_expand:
        log.step(f"loading existing model: {out_dir}")
        try:
            try:
                data, metadata = wm.load()
            except Exception:
                data, metadata = None, None
            prev_meta = dict(metadata or {})
            if prev_meta.get("dim"):
                dim = int(prev_meta["dim"])
            if prev_meta.get("layers"):
                layers = int(prev_meta["layers"])
            vp = os.path.join(out_dir, "vocab.json")
            if os.path.exists(vp):
                prev_vocab = Vocabulary.load(vp)
            if data is not None and "patterns_raw" in (data.files or []):
                prev_extractor = PatternExtractor(ngram_range=_ngram)
                prev_extractor.patterns = dict(pickle.loads(data["patterns_raw"].tobytes()))
            log.ok(f"existing model loaded (dims re-used)")
        except Exception as e:
            log.warn(f"could not load existing model ({e}); building fresh")

    # --- Prepare components ---
    if prev_vocab is not None:
        vocab = prev_vocab
    else:
        vocab = Vocabulary(vocab_size=vocab_size)
    extractor = PatternExtractor(ngram_range=_ngram)
    if prev_extractor is not None:
        extractor.patterns = dict(prev_extractor.patterns)
    sampler = ReservoirSampler(k=min(_res_k * _res_mult, (_res_k * 4) + _res_k))

    # QA bank: instruction/answer pairs extracted from the stream, saved INTO
    # the model weights so chat is fully self-contained (never reads external
    # pool files at runtime).  Capped to keep RAM bounded.
    _qa_cap = min(_gd('qa_bank_cap', 100000), tuner.reservoir_size * 2)
    qa_sampler = ReservoirSampler(k=_qa_cap)
    _qa_pattern = re.compile(
        r'<instruction>(.*?)</instruction>.*?<answer>(.*?)</answer>',
        re.DOTALL,
    )
    _qa_pattern2 = re.compile(
        r'<user>(.*?)</user>.*?<assistant>(.*?)</assistant>',
        re.DOTALL,
    )

    # --- Single pass: vocab + patterns + reservoir + knowledge ---------------
    print()
    print(log.header(f" ENCODING (streaming) -> {os.path.basename(out_dir) or out_dir} "))
    log.info(f"params: dim={dim} layers={layers}  (single-pass streaming)")
    
    # Initialize Knowledge Extractor
    knowledge_extractor = None
    knowledge_path = os.path.join(out_dir, "knowledge")
    try:
        from novacore.core.knowledge_extractor import EncodingTimeLearner
        knowledge_extractor = EncodingTimeLearner()
        log.info("Knowledge extractor initialized - will learn from data during encoding")
    except Exception as e:
        log.warn(f"Could not initialize knowledge extractor: {e}")
    
    from time import time as _time
    _t0 = _time()
    count = 0

    # ---- Parallel encode support (vocab freqs + n-gram patterns) ------
    # Deterministic map-reduce: worker per chunk, driver merges Counters.
    # Fallback to the original sequential loop whenever workers fail.
    from collections import deque as _deque
    from multiprocessing import Pool as _Pool
    _use_parallel = (_workers > 1 and prev_vocab is None)
    _pool = None
    if _use_parallel:
        try:
            _pool = _Pool(processes=_workers)
        except Exception:
            _pool = None
    _chunk_rows = 20000
    _chunk = []
    _futures = _deque()
    _vocab_fed = 0
    _pat_fed = 0

    def _flush_chunk():
        nonlocal _vocab_fed, _pat_fed
        if not _chunk:
            return
        vd = _chunk[: max(0, _vocab_cap - _vocab_fed)]
        pd = _chunk[: max(0, _pat_cap - _pat_fed)]
        _vocab_fed += len(vd)
        _pat_fed += len(pd)
        job = (vd, pd, _ngram)
        if _pool is not None:
            _futures.append(_pool.apply_async(_extract_docs_worker, (job,)))
        else:
            _futures.append(_extract_docs_worker(job))
        _chunk.clear()
        # Bound outstanding work: keep <=4 promises when parallel,
        # resolve immediately in the serial fallback.
        while len(_futures) >= (4 if _pool is not None else 1):
            _resolve_one()

    def _resolve_one():
        item = _futures.popleft()
        try:
            freq, patm = item.get() if hasattr(item, 'get') else item
        except Exception as exc:
            print(f"[NovaCore] parallel worker error (continuing): {exc}",
                  file=sys.stderr, flush=True)
            freq, patm = {}, {}
        if prev_vocab is None:
            vocab.freqs.update(freq)
        extractor.patterns.update(patm)

    for text in text_iter:
        if not isinstance(text, str) or not text:
            continue
        # Knowledge extraction: feed ALL texts (stateful, kept sequential)
        if knowledge_extractor:
            knowledge_extractor.learn_from_document(text, f"doc_{count}")
        # Reservoir: feed ALL texts
        sampler.add(text)
        # QA bank: extract instruction/answer pairs for self-contained chat
        if qa_sampler and count % 2 == 0:
            m = _qa_pattern.search(text)
            if not m:
                m = _qa_pattern2.search(text)
            if m and len(m.group(1).strip()) > 3 and len(m.group(2).strip()) > 3:
                qa_sampler.add(f"<instruction>{m.group(1).strip()}</instruction>\n<answer>{m.group(2).strip()}</answer>")
        count += 1
        if count % 200000 == 0:
            el = _time() - _t0
            rate = count / max(1, el)
            print(f"    · streaming {count} docs ({el:.0f}s, ~{rate:.0f} docs/s, "
                  f"parallel={'on' if _pool else 'off'})", flush=True)
        # Vocab + patterns:
        #   parallel  -> accumulate chunk, extract in worker processes
        #   sequential-> original per-doc path (expand models / worker 1)
        if _use_parallel:
            _chunk.append(text)
            if len(_chunk) >= _chunk_rows:
                _flush_chunk()
        else:
            if count < _vocab_cap:
                if prev_vocab is not None:
                    vocab.grow([text])
                else:
                    for tok in vocab._tokenize(text):
                        vocab.freqs[tok] += 1
            if count < _pat_cap:
                extractor.update(text)
        # Free memory periodically for long streams + auto-tuner guard
        _mem_cleanup_interval = _gd('memory_cleanup_interval')
        if count % _mem_cleanup_interval == 0:
            release_memory()
            # When running near 90% RAM, shed reservoir aggressively to prevent OOM
            if tuner.mem_pressure() > 0.85:
                # Trim reservoir by half if over budget; keeps representative subset
                if len(sampler.reservoir) > tuner.reservoir_size // 2:
                    sampler.reservoir = sampler.reservoir[:tuner.reservoir_size // 2]
                    gc.collect()
            if tuner.mem_pressure() > 0.90:
                print(f"[NovaCore] AUTO-GUARD: memory at {tuner.mem_pressure()*100:.0f}%; reducing reservoir + vocab feed")
                # Force vocab cap reduction for remaining stream
                _vocab_cap = min(_vocab_cap, count + 1000)

    if _use_parallel:
        _flush_chunk()
        while _futures:
            _resolve_one()

    _elapsed = _time() - _t0
    n_docs = count

    # Save knowledge base after encoding
    if knowledge_extractor:
        try:
            knowledge_extractor.save(knowledge_path)
            # Save knowledge index for inference-time retrieval
            index_data = {
                'knowledge_store': knowledge_extractor.indexer.knowledge_store,
                'keyword_index': {k: list(v) for k, v in knowledge_extractor.indexer.keyword_index.items()},
                'category_index': {k: list(v) for k, v in knowledge_extractor.indexer.category_index.items()},
            }
            import json as _json
            idx_path = os.path.join(out_dir, "knowledge_index.json")
            with open(idx_path, 'w', encoding='utf-8') as _f:
                _json.dump(index_data, _f, ensure_ascii=False, indent=2)
            log.ok(f"Knowledge base saved: {knowledge_extractor.total_docs_processed} docs processed")
            log.ok(f"Knowledge index saved: {len(index_data['knowledge_store'])} items")
            stats = knowledge_extractor.get_knowledge_base()
            for cat, num in stats['extractor']['stats'].items():
                if num > 0:
                    log.info(f"  {cat}: {num} extracted")
        except Exception as e:
            log.warn(f"Could not save knowledge base: {e}")

    # Finalize vocab (build from freqs)
    if prev_vocab is None:
        for tok, freq in vocab.freqs.most_common():
            if freq < vocab.min_freq:
                continue
            if tok in vocab.token_to_id:
                continue
            if len(vocab.token_to_id) >= vocab.vocab_size:
                break
            tid = len(vocab.token_to_id)
            vocab.token_to_id[tok] = tid
            vocab.id_to_token[tid] = tok
    log.ok(f"stream pass done: {n_docs} docs in {_elapsed:.0f}s  ({log.total()})")
    log.ok(f"vocabulary: {len(vocab)} tokens")

    # Finalize patterns
    extractor.prune()
    log.ok(f"patterns: {len(extractor.patterns)}")

    # ===== TRAINING UPGRADES (SVD + IDF + Confidence + Multi-Head) =====
    from novacore.core.training_upgrades import TrainingUpgrader
    from novacore.config import load_config
    cfg = load_config()
    upgrader = TrainingUpgrader(vocab, config=cfg)
    if upgrader.enabled:
        log.step("Training upgrades: SVD + IDF + Confidence + Multi-Head")
        try:
            upgrader.build_from_reservoir(list(sampler.sample()))
            upgrader.compute_confidence(extractor.patterns)
            upgrader.cluster_reservoir(list(sampler.sample()))
            upgrade_dir = os.path.join(out_dir, "upgrades")
            upgrader.save(upgrade_dir)
            log.ok(f"Training upgrades saved: {upgrade_dir} ({log.total()})")
        except Exception as e:
            log.warn(f"Training upgrades failed: {e}")
    release_memory(verbose=True)

    # ===== REASONING + CREATIVE pattern extraction =====
    sample_list = list(sampler.sample()) if not hasattr(sampler, '_sampled_list') else sampler._sampled_list
    if not sample_list:
        sample_list = list(sampler.sample())

    try:
        from novacore.core.reasoning_engine import ReasoningPatternExtractor, DeepReasoner
        from novacore.core.creative_engine import CreativePatternExtractor, CreativeEngine

        import time as _time

        def _parallel_lists(worker, docs, fields):
            """Map `worker` over sliced docs, merge list-fields, keep order.
            Prints per-chunk completion so long regex phases show progress."""
            parts = []
            _slice = 10000
            _t0 = _time.time()
            _total = len(docs)
            if _pool is not None:
                _futs = []
                for _s in range(0, _total, _slice):
                    _futs.append(_pool.apply_async(worker, (docs[_s:_s + _slice],)))
                for _fi, _f in enumerate(_futs, 1):
                    try:
                        parts.append(_f.get())
                    except Exception as _exc:
                        print(f"[NovaCore] parallel extract error: {_exc}",
                              file=sys.stderr, flush=True)
                    print(f"    · {os.path.basename(getattr(worker, '__name__', 'extract'))} "
                          f"chunk {_fi}/{len(_futs)} ({_time.time() - _t0:.0f}s)",
                          flush=True)
            else:
                for _si, _s in enumerate(range(0, _total, _slice), 1):
                    parts.append(worker(docs[_s:_s + _slice]))
                    print(f"    · chunk {_si}/{( _total + _slice - 1) // _slice} "
                          f"({_time.time() - _t0:.0f}s)", flush=True)
            merged = {f: [] for f in fields}
            for p in parts:
                for f in fields:
                    merged[f].extend(p.get(f, []))
            return merged

        log.step("Extracting reasoning patterns...")
        reasoning_ext = ReasoningPatternExtractor()
        _reasoning_cap = min(_gd('reasoning_extract_cap'), len(sample_list))
        _r_t0 = _time.time()
        if _reasoning_cap and _use_parallel:
            merged = _parallel_lists(
                _extract_reasoning_worker, sample_list[:_reasoning_cap],
                _REASONING_FIELDS)
            for _f in _REASONING_FIELDS:
                setattr(reasoning_ext, _f, merged[_f])
            print(f"    · reasoning: parallel over {_reasoning_cap} docs "
                  f"({_time.time() - _r_t0:.0f}s)")
        else:
            for _ri, text in enumerate(sample_list[:_reasoning_cap]):
                reasoning_ext.extract_from_text(text)
                if (_ri + 1) % 25000 == 0:
                    print(f"    · reasoning {_ri + 1}/{_reasoning_cap} docs "
                          f"({_time.time() - _r_t0:.0f}s)", flush=True)
            _r_el = _time.time() - _r_t0
            print(f"    · reasoning: {_reasoning_cap} docs ({_r_el:.0f}s)")
        reasoner = DeepReasoner()
        reasoner.patterns = reasoning_ext
        reasoner.save_patterns(os.path.join(out_dir, "reasoning_patterns.json"))
        rs = reasoning_ext.stats()
        log.ok(f"Reasoning: {sum(rs.values())} patterns ({rs})")

        log.step("Extracting creative patterns...")
        creative_ext = CreativePatternExtractor()
        _creative_cap = min(_gd('creative_extract_cap'), len(sample_list))
        _c_t0 = _time.time()
        if _creative_cap and _use_parallel:
            merged = _parallel_lists(
                _extract_creative_worker, sample_list[:_creative_cap],
                _CREATIVE_FIELDS)
            for _f in _CREATIVE_FIELDS:
                setattr(creative_ext, _f, merged[_f])
            print(f"    · creative: parallel over {_creative_cap} docs "
                  f"({_time.time() - _c_t0:.0f}s)")
        else:
            for _ci, text in enumerate(sample_list[:_creative_cap]):
                creative_ext.extract_from_text(text)
                if (_ci + 1) % 25000 == 0:
                    print(f"    · creative {_ci + 1}/{_creative_cap} docs "
                          f"({_time.time() - _c_t0:.0f}s)", flush=True)
            _c_el = _time.time() - _c_t0
            print(f"    · creative: {_creative_cap} docs ({_c_el:.0f}s)")
        creative = CreativeEngine()
        creative.patterns = creative_ext
        creative.save_patterns(os.path.join(out_dir, "creative_patterns.json"))
        cs = creative_ext.stats()
        log.ok(f"Creative: {sum(cs.values())} patterns ({cs})")
    except Exception as e:
        log.warn(f"Reasoning/Creative extraction failed: {e}")

    # Release parallel workers before the memory-heavy analytic phase
    if _pool is not None:
        try:
            _pool.close()
            _pool.join()
        except Exception:
            pass
        _pool = None
    release_memory(verbose=True)

    # --- Analytic weights from reservoir sample --------------------------
    processor = TextProcessor(vocab=vocab, dim=dim)
    _buckets = dim * _buckets_mult
    hasher = FeatureHasher(num_buckets=_buckets, depth=_depth)
    sample = sampler.sample()
    log.ok(f"reservoir sample: {len(sample)} docs")

    encoder = None
    if sample:
        log.step("computing analytic weights")
        log.info(f"auto linalg backend: {backend_info()}")
        X = np.array([processor.text_to_vector(t) for t in tqdm(sample, desc="Encoding")])
        Y = np.zeros((X.shape[0], min(len(vocab), _tgt)), dtype=np.float64)
        for i, t in enumerate(sample):
            ids = vocab.encode(t)
            for j, tid in enumerate(ids[:min(len(ids), _tgt)]):
                if tid < Y.shape[1]:
                    Y[i, tid] = 1.0
        encoder = RandomFourierEncoder(input_dim=X.shape[1], dim=dim, layers=layers)
        encoder.fit_analytic(X, Y)
        log.ok(f"analytic weights: X={X.shape[0]}x{X.shape[1]}  ({log.total()})")
    else:
        log.warn("no sample for analytic weights - saving structural weights only")

    # Save reservoir samples for retrieval-based generation
    reservoir_data = list(sample) if sample else []
    # Truncate individual samples to save memory (max chars from config)
    _res_max_chars = _gd('reservoir_max_chars')
    reservoir_data = [s[:_res_max_chars] if isinstance(s, str) else str(s)[:_res_max_chars] for s in reservoir_data]

    # Free large intermediate objects before save
    del sampler, X, Y
    release_memory(verbose=True)

    # --- Save ------------------------------------------------------------
    arrays = {}
    if is_expand:
        try:
            prev_data, _ = wm.load()
            if prev_data is not None and hasattr(prev_data, "files"):
                for k in prev_data.files:
                    try:
                        arrays[k] = np.array(prev_data[k])
                    except Exception:
                        pass
        except Exception:
            pass
    if encoder is not None:
        for k, v in encoder.save_weights().items():
            arrays[f"encoder_{k}"] = np.asarray(v)
    arrays["pattern_vec"] = extractor.encode(dim)
    arrays["hasher"] = hasher.vector()
    arrays["patterns_raw"] = np.frombuffer(pickle.dumps(dict(extractor.patterns)), dtype=np.uint8)

    # Save reservoir samples for retrieval-based generation.
    # EXPAND: merge with the previous reservoir (up to cap) so retraining
    # ACCUMULATES memory instead of replacing it (fixes shrinking weights).
    if reservoir_data or is_expand:
        merged = list(reservoir_data)
        if is_expand and "reservoir_sample" in arrays:
            try:
                prev_res = pickle.loads(arrays["reservoir_sample"].tobytes())
                if prev_res:
                    merged = (list(prev_res) + merged)[: _res_k]
            except Exception:
                pass
        if merged:
            reservoir_bytes = pickle.dumps(merged)
            arrays["reservoir_sample"] = np.frombuffer(reservoir_bytes, dtype=np.uint8)

    # Save QA bank (self-contained chat — no external pool at runtime).
    # EXPAND: keep previous QA pairs too, up to the qa_bank cap.
    if qa_sampler is not None:
        qa_data = list(qa_sampler.sample())
        if is_expand and "qa_bank" in arrays:
            try:
                prev_qa = pickle.loads(arrays["qa_bank"].tobytes())
                if prev_qa:
                    qa_data = list(prev_qa) + list(qa_data)
            except Exception:
                pass
        if qa_data:
            qa_bytes = pickle.dumps(qa_data[:_qa_cap])
            arrays["qa_bank"] = np.frombuffer(qa_bytes, dtype=np.uint8)
            log.ok(f"QA bank baked into model: {len(qa_data[:_qa_cap])} pairs")

    metadata = {
        "name": os.path.basename(out_dir),
        "created": str(np.datetime64('now')),
        "num_documents": (prev_meta.get("num_documents", 0) if prev_meta else 0) + n_docs,
        "dim": dim,
        "layers": layers,
        "vocab_size": len(vocab),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "source": source,
        "text_column": column,
        "input_format": "train",
    }
    wm.save(arrays, metadata)
    vocab.save(os.path.join(out_dir, "vocab.json"))
    release_memory(verbose=True)

    info = wm.info()
    log.ok(f"saved weights: {info.get('total_size_mb', 0)} MB  ({log.total()})")

    # Save knowledge base for batch encoding
    if knowledge_extractor:
        try:
            knowledge_extractor.save(knowledge_path)
            log.ok(f"Knowledge base saved: {knowledge_extractor.total_docs_processed} docs processed")
            stats = knowledge_extractor.get_knowledge_base()
            for cat, num in stats['extractor']['stats'].items():
                if num > 0:
                    log.info(f"  {cat}: {num} extracted")
        except Exception as e:
            log.warn(f"Could not save knowledge base: {e}")
    
    log.done(f"model ready: {out_dir}  |  docs={metadata['num_documents']}")


def _encode_texts_to_weights(texts, out_dir, dim, layers, vocab_size, max_tokens, temperature,
                             source="", column=None, merge_existing=False):
    """Encode a list of text documents into weights.

    Config values (dim/layers/vocab_size/max_tokens/temperature) come from
    the caller (resolved via --config + flags). No built-in fallbacks here.
    If merge_existing is True, expand an existing model in out_dir.

    Accepts plain lists, generators, and ``TextStream`` objects.  When the
    input supports ``len()`` (list / TextStream), the original batched
    approach is used.  For plain generators a single-pass streaming path
    is chosen automatically.
    """
    from novacore.config import get_default as _gd
    from tqdm import tqdm
    import pickle

    # Reproducible builds: global numpy/random must be seeded (training_upgrades
    # uses np.random.choice / np.random.randn outside its own RandomState).
    _seed = int(_gd('reservoir_seed') or _gd('seed') or 42)
    np.random.seed(_seed)
    random.seed(_seed)
    del _seed

    # --- Detect if we can get a length -----------------------------------
    try:
        n_docs = len(texts)
    except TypeError:
        n_docs = None

    # For generators without a known length, use the streaming path.
    if n_docs is None:
        return _train_from_stream(
            texts, out_dir=out_dir, dim=dim, layers=layers,
            vocab_size=vocab_size, max_tokens=max_tokens,
            temperature=temperature, source=source, column=column,
            merge_existing=merge_existing,
        )

    log = Logger()
    print()
    print(log.header(f" ENCODING {n_docs} DOCS -> {os.path.basename(out_dir) or out_dir} "))
    log.info(f"params: dim={dim} layers={layers} vocab_cap={vocab_size}")

    # Structural params come from config.json (single source - no hardcoded values)
    _vocab_cap = _gd('pattern_vocab_cap')
    _pat_cap = _gd('pattern_sample_cap')
    _res_k = _gd('reservoir_sample_size')
    _res_mult = _gd('reservoir_merge_multiplier')
    _depth = _gd('hasher_depth')
    _buckets_mult = _gd('hasher_buckets_multiplier')
    _ngram = tuple(_gd('ngram_range'))
    _tgt = _gd('analytic_target_dim')

    is_expand = merge_existing and os.path.isdir(out_dir)
    wm = WeightManager(out_dir)

    # --- Load existing model if expanding ---
    prev_vocab = None
    prev_extractor = None
    prev_meta = {}
    if is_expand:
        log.step(f"loading existing model: {out_dir}")
        try:
            try:
                data, metadata = wm.load()
            except Exception:
                data, metadata = None, None
            prev_meta = dict(metadata or {})
            # reuse existing dim/layers/vocab from the saved model when expanding
            if prev_meta.get("dim"):
                dim = int(prev_meta["dim"])
            if prev_meta.get("layers"):
                layers = int(prev_meta["layers"])
            # vocab
            vp = os.path.join(out_dir, "vocab.json")
            if os.path.exists(vp):
                prev_vocab = Vocabulary.load(vp)
            # patterns
            if data is not None and "patterns_raw" in (data.files or []):
                prev_extractor = PatternExtractor(ngram_range=_ngram)
                prev_extractor.patterns = dict(pickle.loads(data["patterns_raw"].tobytes()))
            log.ok(f"existing model loaded (dims re-used)")
        except Exception as e:
            log.warn(f"could not load existing model ({e}); building fresh")

    # --- Build/merge vocabulary ---
    log.step("building vocabulary")
    if prev_vocab is not None:
        vocab = prev_vocab
        vocab.grow(texts[:min(len(texts), _vocab_cap)])
    else:
        vocab = Vocabulary(vocab_size=vocab_size)
        vocab.build(texts[:min(len(texts), _vocab_cap)])
    log.ok(f"vocabulary: {len(vocab)} tokens  ({log.total()})")

    processor = TextProcessor(vocab=vocab, dim=dim)
    _buckets = dim * _buckets_mult
    hasher = FeatureHasher(num_buckets=_buckets, depth=_depth)

    # --- Build/merge pattern extractor ---
    log.step("extracting n-gram patterns")
    extractor = PatternExtractor(ngram_range=_ngram)
    if prev_extractor is not None:
        extractor.patterns = dict(prev_extractor.patterns)
    extractor.update_many(texts[:min(len(texts), _pat_cap)], log=log)
    extractor.prune()
    log.ok(f"patterns: {len(extractor.patterns)}  ({log.total()})")

    # --- Reservoir sampling ---
    log.step("reservoir sampling")
    sampler = ReservoirSampler(k=min(_res_k * _res_mult, len(texts) + _res_k))
    sampler.add_many(texts, log=log)
    log.ok(f"sampled {len(sampler.sample())} docs  ({log.total()})")

    # --- Knowledge extraction during encoding ---
    knowledge_extractor = None
    knowledge_path = os.path.join(out_dir, "knowledge")
    try:
        from novacore.core.knowledge_extractor import EncodingTimeLearner
        knowledge_extractor = EncodingTimeLearner()
        log.step("extracting knowledge from documents")
        for i, text in enumerate(tqdm(texts[:min(len(texts), 10000)], desc="Learning")):
            if text and isinstance(text, str):
                knowledge_extractor.learn_from_document(text, f"doc_{i}")
        log.ok(f"knowledge base: {knowledge_extractor.total_docs_processed} docs processed  ({log.total()})")
    except Exception as e:
        log.warn(f"Could not initialize knowledge extractor: {e}")

    # --- Analytic weights ---
    log.step("computing analytic weights")
    log.info(f"auto linalg backend: {backend_info()}")
    sample = sampler.sample()
    encoder = None
    if sample:
        X = np.array([processor.text_to_vector(t) for t in tqdm(sample, desc="Encoding")])
        Y = np.zeros((X.shape[0], min(len(vocab), _tgt)), dtype=np.float64)
        for i, t in enumerate(sample):
            ids = vocab.encode(t)
            for j, tid in enumerate(ids[:min(len(ids), _tgt)]):
                if tid < Y.shape[1]:
                    Y[i, tid] = 1.0
        encoder = RandomFourierEncoder(input_dim=X.shape[1], dim=dim, layers=layers)
        encoder.fit_analytic(X, Y)
        log.ok(f"analytic weights: X={X.shape[0]}x{X.shape[1]}  ({log.total()})")
    else:
        log.warn("no sample for analytic weights - saving structural weights only")

    # --- Save (merge with existing arrays if expanding) ---
    arrays = {}
    if is_expand:
        try:
            prev_data, _ = wm.load()
            if prev_data is not None and hasattr(prev_data, "files"):
                for k in prev_data.files:
                    try:
                        arrays[k] = np.array(prev_data[k])
                    except Exception:
                        pass
        except Exception:
            pass
    if encoder is not None:
        for k, v in encoder.save_weights().items():
            arrays[f"encoder_{k}"] = np.asarray(v)
    arrays["pattern_vec"] = extractor.encode(dim)
    arrays["hasher"] = hasher.vector()
    arrays["patterns_raw"] = np.frombuffer(pickle.dumps(dict(extractor.patterns)), dtype=np.uint8)

    metadata = {
        "name": os.path.basename(out_dir),
        "created": str(np.datetime64('now')),
        "num_documents": (prev_meta.get("num_documents", 0) if prev_meta else 0) + len(texts),
        "dim": dim,
        "layers": layers,
        "vocab_size": len(vocab),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "source": source,
        "text_column": column,
        "input_format": "train",
    }
    wm.save(arrays, metadata)
    vocab.save(os.path.join(out_dir, "vocab.json"))
    release_memory(verbose=True)

    info = wm.info()
    log.ok(f"saved weights: {info.get('total_size_mb', 0)} MB  ({log.total()})")

    # Save knowledge base for batch encoding
    if knowledge_extractor:
        try:
            knowledge_extractor.save(knowledge_path)
            log.ok(f"Knowledge base saved: {knowledge_extractor.total_docs_processed} docs processed")
            stats = knowledge_extractor.get_knowledge_base()
            for cat, num in stats['extractor']['stats'].items():
                if num > 0:
                    log.info(f"  {cat}: {num} extracted")
        except Exception as e:
            log.warn(f"Could not save knowledge base: {e}")
    
    log.done(f"model ready: {out_dir}  |  docs={metadata['num_documents']}")


def cmd_cortex(args):
    """Training-free attention memory net: interactive chat / benchmark."""

    def _load_vocab(dir_):
        from novacore.tokenizer.vocab import Vocabulary
        vp = os.path.join(dir_, "vocab.json")
        if os.path.exists(vp):
            try:
                return Vocabulary.load(vp)
            except Exception:
                pass
        return Vocabulary()

    if args.cortex_cmd == "bench":
        from novacore.inference.memory_transformer import build_from_weights
        from novacore.inference.chat import ChatSession
        import time as _time

        t0 = _time.time()
        print(f"[cortex] building memory net from: {args.weights}", flush=True)
        vocab = _load_vocab(args.weights)
        mt, info = build_from_weights(
            args.weights, vocab,
            dim=args.dim, top_k=args.topk, gate_threshold=args.gate,
            max_memories=args.max_memories,
        )
        print(f"[cortex] built in {_time.time()-t0:.1f}s: {info}", flush=True)

        session = ChatSession(args.weights)
        session.load()
        questions = [q.strip() for q in args.questions.split(",") if q.strip()]
        for q in questions:
            print("\n" + "=" * 64)
            print(f"Q: {q}")
            t1 = _time.time()
            try:
                ans_a = mt.generate(q, max_tokens=args.max_tokens, temperature=0.7)
            except Exception as e:
                ans_a = f"[err: {e}]"
            ta = _time.time() - t1
            t2 = _time.time()
            try:
                ans_b = session.chat(q, max_tokens=args.max_tokens, verbose=False)
            except Exception as e:
                ans_b = f"[err: {e}]"
            tb = _time.time() - t2
            print(f"\n\033[94m[CORTEX attention net ({ta:.2f}s)]\033[0m")
            print(str(ans_a).strip())
            print(f"\n\033[92m[LEGACY chat path ({tb:.2f}s)]\033[0m")
            print(str(ans_b).strip())
        return

    # cortex chat (default)
    from novacore.inference.memory_transformer import build_from_weights

    print(f"[cortex] building memory net from: {args.weights}", flush=True)
    vocab = _load_vocab(args.weights)
    mt, info = build_from_weights(
        args.weights, vocab,
        dim=args.dim, top_k=args.topk, gate_threshold=args.gate,
        max_memories=args.max_memories,
    )
    print(f"[cortex] ready: {info}", flush=True)

    def _run(prompt):
        return mt.generate(prompt, max_tokens=args.max_tokens,
                           temperature=args.temperature)

    if args.prompt:
        print(f"You: {args.prompt}")
        print(f"\033[94mCortex> {_run(args.prompt)}\033[0m")
        return

    print("  (cortex chat - training-free attention memory net; /exit to quit)")
    while True:
        try:
            nline = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Cortex] Goodbye!")
            break
        if not nline:
            continue
        if nline in ("/exit", "/quit", "/q"):
            print("[Cortex] Goodbye!")
            break
        print(f"\033[94mCortex> {_run(nline)}\033[0m")


def main():
    # Force UTF-8 stdout so Unicode dataset content doesn't crash on Windows cp1252
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    args = build_parser().parse_args()
    # Resolve bare --weights names under the default weights dir (config 'default_weights')
    if hasattr(args, 'weights') and args.weights:
        args.weights = _resolve_model_dir(args.weights)
    handlers = {
        "train": cmd_train,
        "train-pools": cmd_train,
        "encode": cmd_encode,
        "generate": cmd_generate,
        "info": cmd_info,
        "compress": cmd_compress,
        "query": cmd_query,
        "validate": cmd_validate,
        "list": cmd_list,
        "convert": cmd_convert,
        "chat": cmd_chat,
        "cortex": cmd_cortex,
        "hf": cmd_hf,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()

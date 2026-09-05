"""NovaCore config loader - split single source of truth.

Model config : config/config.json      (defaults, paths, cli)
Data config  : config/data_config.json (hf, datasets - native schema)
A custom model config path can be passed via `configure(path)` (e.g. from --config flag).
"""
import json
import os

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "config.json"
)

_DEFAULT_DATA_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "data_config.json"
)

_REQUIRED_DEFAULTS = [
    "dim", "layers", "vocab_size", "max_tokens", "temperature",
    "max_history", "seed", "hasher_depth", "hasher_buckets_multiplier",
    "reservoir_sample_size", "pattern_vocab_cap", "pattern_sample_cap",
    "analytic_target_dim", "ngram_range",
]

_reported_missing = set()


def _validate_defaults(cfg):
    """Warn (once per key) if core defaults are missing from config.json."""
    global _reported_missing
    defaults = cfg.get("defaults", {})
    for key in _REQUIRED_DEFAULTS:
        if key not in defaults or defaults.get(key) is None:
            if key not in _reported_missing:
                _reported_missing.add(key)
                import sys
                print(
                    f"[NovaCore] WARNING: config.json missing default '{key}'. "
                    f"Add it under 'defaults' - no built-in fallback is used.",
                    file=sys.stderr,
                )

_cached = None
_custom_path = None
_data_cached = None
_data_custom_path = None


def configure(path):
    """Point the config loader at a custom config.json file and reload."""
    global _custom_path, _cached
    _custom_path = path
    _cached = None
    return load_config()


def configure_data(path):
    """Point the data config loader at a custom data_config.json and reload."""
    global _data_custom_path, _data_cached
    _data_custom_path = path
    _data_cached = None
    return load_data_config()


def _resolve_path():
    return _custom_path if _custom_path else _DEFAULT_CONFIG_PATH


def _resolve_data_path():
    return _data_custom_path if _data_custom_path else _DEFAULT_DATA_CONFIG_PATH


def reset():
    """Reset to the default config files (clears custom paths + caches)."""
    global _custom_path, _cached, _data_custom_path, _data_cached
    _custom_path = None
    _cached = None
    _data_custom_path = None
    _data_cached = None


def load_config(path=None):
    """Load config.json (cached). If path given, use it (and store as custom).

    Accepts JSONC (JSON with // and /* */ comments + trailing commas) so
    users can comment out blocks for reference (consistent with data_config).
    """
    global _cached
    if path is not None and path != _custom_path:
        return configure(path)
    if _cached is not None:
        return _cached
    cfg_path = _resolve_path()
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        try:
            _cached = _parse_jsonc(raw)
        except json.JSONDecodeError:
            _cached = json.loads(raw)
        _validate_defaults(_cached)
    else:
        _cached = {"defaults": {}, "paths": {}}
        _validate_defaults(_cached)
    return _cached


def _parse_jsonc(text):
    """Parse JSON with C-style comments (// line, /* */ block) and trailing
    commas.  Strips them out first, then hands clean text to json.loads.
    Used for data_config.json so users can leave helpful comments / disable
    a block by commenting it out (similar to tsconfig / VSCode settings).
    """
    import re
    # remove block comments /* ... */ (non-greedy, multi-line)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # remove // line comments (anything from // to end of line, but not inside strings)
    # Simple approach: walk char by char tracking string state
    out = []
    i = 0
    in_str = False
    quote = ""
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_str = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue
        # detect //
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # skip to end of line
            j = text.find("\n", i)
            if j == -1:
                break
            i = j  # keep the newline
            continue
        out.append(ch)
        i += 1
    cleaned = "".join(out)
    # remove trailing commas (", }" or ", ]")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return json.loads(cleaned)


def load_data_config(path=None):
    """Load data_config.json (cached). If path given, use it (and store as custom).

    Accepts JSONC (JSON with // and /* */ comments + trailing commas) so
    users can comment out blocks for reference.
    """
    global _data_cached
    if path is not None and path != _data_custom_path:
        return configure_data(path)
    if _data_cached is not None:
        return _data_cached
    cfg_path = _resolve_data_path()
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        try:
            _data_cached = _parse_jsonc(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"[NovaCore] ERROR: failed to parse {cfg_path}\n"
                f"  {e}\n"
                f"  Remove the offending line or escape it as a string."
            )
    else:
        _data_cached = {"hf": {}, "datasets": {}}
    return _data_cached


def get(section, key, default=None):
    """Get a config value: get('defaults', 'dim', 512)."""
    cfg = load_config()
    val = cfg.get(section, {}).get(key, default)
    return val


def get_default(key, default=None):
    """Get a default setting."""
    return get('defaults', key, default)


def get_path(key, default=None):
    """Get a path setting."""
    return get('paths', key, default)


def get_hf(key, default=None):
    """Get an HF integration setting (from data_config.json 'hf' section)."""
    data = load_data_config()
    return data.get('hf', {}).get(key, default)


def get_data(key, default=None):
    """Get a top-level data config value."""
    return load_data_config().get(key, default)


def get_cli(key, default=None):
    """Get a CLI setting (from config.json 'cli' section)."""
    return get('cli', key, default)


def get_dataset(name, default=None):
    """Get per-dataset override from data_config.json 'datasets' (keyed by HF id)."""
    data = load_data_config()
    return data.get('datasets', {}).get(name, default)


def resolve_path(key, default=None):
    """Resolve a config path to an absolute path under the project root."""
    raw = get_path(key, default)
    if raw is None:
        return None
    if os.path.isabs(raw):
        return raw
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, raw)

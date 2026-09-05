"""NovaCore console logger - clean, professional status lines with timing.

Provides a tiny, dependency-free logging layer used across the CLI so that
every phase (download / stream / cached / skip / train) prints a consistent,
readable, timestamped line with an elapsed-time badge. Colors are only used
when stdout is a TTY and are otherwise stripped automatically.
"""
import os
import sys
import time
from datetime import datetime


class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


# Color only when attached to a real terminal (not when piped / redirected).
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code, text):
    return f"{code}{text}{_Color.RESET}" if _USE_COLOR else text


def _init():
    """Force UTF-8 output on Windows so Unicode dataset content does not crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def col(text, code):
    return _c(code, str(text))


def badge(text, color=_Color.GREY):
    """Draw a compact [] status badge, e.g. [ OK ]."""
    return _c(color, f"[{text}]")


def _header_text(text, width=72, char="="):
    """Draw a prominent section divider, e.g. system tool header."""
    pad = max(0, width - len(text) - 4)
    left = pad // 2
    right = pad - left
    line = (char * left) + "  " + text + "  " + (char * right)
    return _c(_Color.BOLD, line)


def header(text, width=72, char="="):
    return _header_text(text, width=width, char=char)


def _fmt_sec(sec):
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Logger:
    """A simple, structured logger with elapsed-time and status badges.

    Callers create one logger per phase. `phase` is a short label (e.g. the
    dataset name + config) shown once, followed by progress lines.
    """

    def __init__(self):
        _init()
        self._start = time.time()
        self._last = self._start
        self._section_done = False

    def reset(self):
        self._last = time.time()

    def header(self, text, width=72, char="="):
        """Prominent section divider, e.g. step headers."""
        return _header_text(text, width=width, char=char)

    def elapsed(self):
        return _fmt_sec(time.time() - self._last)

    def total(self):
        return _fmt_sec(time.time() - self._start)

    def step(self, text):
        """Dim, un-badged progress line (e.g. 'checking cache')."""
        print(f"    {_c(_Color.DIM, '·')} {text}")

    def info(self, text):
        print(f"[NovaCore] {text}")

    def ok(self, text):
        print(f"{badge(' OK ', _Color.GREEN)} {text}")

    def warn(self, text):
        print(f"{badge('WARN', _Color.YELLOW)} {text}")

    def skip(self, text):
        print(f"{badge('SKIP', _Color.BLUE)} {text}")

    def done(self, text, elapsed=True):
        msg = f"{text}"
        if elapsed:
            msg += _c(_Color.DIM, f"  ({self.total()})")
        print(f"{badge('DONE', _Color.GREEN)} {msg}")

    def error(self, text):
        print(f"{badge('FAIL', _Color.RED)} {text}")


_default = Logger()


def info(text):
    _default.info(text)


def ok(text):
    _default.ok(text)


def warn(text):
    _default.warn(text)


def skip(text):
    _default.skip(text)


def error(text):
    _default.error(text)


def done(text, elapsed=True):
    _default.done(text, elapsed)

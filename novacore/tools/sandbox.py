"""tools/sandbox.py — NovaCore code-execution sandbox (single code runner).

The "python terminal" is implemented as a SANDBOX ORCHESTRATOR: Python is the
common controller that spawns the target language's runtime (Python, Go,
JS/Node, C/C++, or any installed runtime) to run/test user code or model code
before it is allowed to leave as an answer.

Security caps (always ON):
  - AST blocklist for Python (no os/sys/network/file-writing/ctypes...)
  - wall-clock timeout for every run
  - output captured and capped (no unbounded streams)
  - other runtimes launched via subprocess with timeout; runtime must be
    installed (`shutil.which`) or the run reports "runtime unavailable".
"""

import ast
import io
import re
import shutil
import subprocess
import threading
import contextlib

_OUTPUT_CAP = 4000          # max characters of stdout kept
_DEFAULT_TIMEOUT_S = 6.0    # wall-clock timeout per run


# ---------------------------------------------------------------------------
# Python blocklist (AST-based) — deny everything that could touch the world.
# ---------------------------------------------------------------------------
_BLOCKED_IMPORTS = frozenset([
    'os', 'sys', 'subprocess', 'socket', 'ctypes', 'importlib', 'shutil',
    'urllib', 'requests', 'http', 'json' and 'ssl', 'pickle', 'shelve',
    'tempfile', 'multiprocessing', 'threading', 'signal', 'fcntl',
])
_BLOCKED_CALLS = frozenset([
    'open', 'exec', 'eval', 'compile', 'input', 'breakpoint', 'exit', 'quit',
    'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
    '__import__', 'help',
])
_BLOCKED_ATTR = frozenset([
    '__subclasses__', '__globals__', '__builtins__', '__import__',
    '__getattribute__', '__setattr__', '__delattr__', '__reduce__',
])


def _scan_python_ast(code):
    """Return (ok, reason). Rejected code returns False + the offending node."""
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split('.')[0]
                if root in _BLOCKED_IMPORTS:
                    return False, f"import of '{root}' blocked (sandbox)"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if root in _BLOCKED_IMPORTS:
                return False, f"import of '{root}' blocked (sandbox)"
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _BLOCKED_CALLS:
                return False, f"call to '{fn.id}()' blocked (sandbox)"
            if isinstance(fn, ast.Attribute) and fn.attr in _BLOCKED_ATTR:
                return False, f"attribute '{fn.attr}' blocked (sandbox)"
    return True, None


# Restricted builtins exposed to sandboxed Python code.
def _safe_builtins():
    import math as _math
    import random as _random
    import statistics as _stats
    b = {
        'print': print, 'len': len, 'range': range, 'int': int,
        'float': float, 'str': str, 'bool': bool, 'list': list,
        'dict': dict, 'set': set, 'tuple': tuple, 'abs': abs, 'min': min,
        'max': max, 'sum': sum, 'round': round, 'sorted': sorted,
        'enumerate': enumerate, 'zip': zip, 'reversed': reversed,
        'chr': chr, 'ord': ord, 'pow': pow, 'divmod': divmod,
        'isinstance': isinstance, 'hasattr': hasattr, 'type': type,
        'all': all, 'any': any, 'repr': repr, 'format': format,
        'True': True, 'False': False, 'None': None,
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'IndexError': IndexError,
        'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError,
        'RuntimeError': RuntimeError, 'OverflowError': OverflowError,
        'math': _math, 'random': _random, 'statistics': _stats,
    }
    try:
        import numpy as _np
        b['np'] = _np
    except Exception:
        pass
    try:
        import fractions as _frac
        b['Fraction'] = _frac.Fraction
    except Exception:
        pass
    return b


class CodeSandbox:
    """Single execution controller — Python is the default runner, any other
    installed runtime (go/node/gcc...) is spawned by name on request."""

    def __init__(self, timeout=_DEFAULT_TIMEOUT_S, output_cap=_OUTPUT_CAP):
        self.timeout = float(timeout)
        self.output_cap = int(output_cap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, code, language='python'):
        """Run `code` in the requested language. Returns a result dict:
        {success, output, error, runtime}. Never raises."""
        lang = (language or 'python').lower().strip()
        if lang in ('py', 'python', 'python3'):
            return self.run_python(code)
        runtime = {
            'go': 'go', 'golang': 'go',
            'js': 'node', 'javascript': 'node', 'node': 'node',
            'ts': 'node', 'typescript': 'node',
            'c': 'gcc', 'cpp': 'g++', 'c++': 'g++', 'cc': 'g++',
            'bash': 'bash', 'sh': 'bash', 'shell': 'bash',
        }.get(lang)
        if runtime is None:
            return {'success': False, 'output': '',
                    'error': f"no runner configured for '{lang}'",
                    'runtime': lang}
        return self.run_external(runtime, code, lang)

    def run_python(self, code):
        """Restricted, timeout-guarded exec of Python code."""
        if not code or not isinstance(code, str):
            return {'success': False, 'output': '', 'error': 'empty code',
                    'runtime': 'python'}
        ok, reason = _scan_python_ast(code)
        if not ok:
            return {'success': False, 'output': '', 'error': reason,
                    'runtime': 'python'}
        result = {'success': False, 'output': '', 'error': None,
                  'runtime': 'python'}
        out_buf = io.StringIO()
        ns = {'__name__': '__novacore_sandbox__',
              '__builtins__': _safe_builtins()}

        def _exec():
            try:
                with contextlib.redirect_stdout(out_buf):
                    exec(compile(code, '<novacore-sandbox>', 'exec'), ns)
                result['success'] = True
            except Exception as e:
                result['error'] = f"{type(e).__name__}: {e}"
                result['success'] = False

        t = threading.Thread(target=_exec, daemon=True)
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            result['error'] = (f"Timeout after {self.timeout:.1f}s "
                               "(possible infinite loop)")
            result['success'] = False
        out = out_buf.getvalue()
        result['output'] = out[:_OUTPUT_CAP]
        if result['success'] and not result['output'].strip():
            result['output'] = "Code executed successfully."
        return result

    def run_external(self, runtime, code, lang):
        """Spawn an installed runtime (go/node/gcc/g++/bash) with timeouts."""
        exe = shutil.which(runtime)
        if exe is None:
            return {'success': False, 'output': '',
                    'error': f"'{runtime}' runtime not installed",
                    'runtime': lang}
        try:
            if runtime in ('gcc', 'g++'):
                import tempfile
                suffix = '.c' if runtime == 'gcc' else '.cpp'
                with tempfile.NamedTemporaryFile(suffix=suffix,
                                                 delete=False) as f:
                    f.write(code.encode('utf-8'))
                    src = f.name
                outexe = src + '.exe'
                proc = subprocess.run(
                    [exe, src, '-o', outexe], capture_output=True, text=True,
                    timeout=self.timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if proc.returncode != 0:
                    return {'success': False,
                            'error': (proc.stderr or 'compile failed')[:500],
                            'output': '', 'runtime': lang}
                proc = subprocess.run(
                    [outexe] if outexe.endswith('.exe') else [outexe],
                    capture_output=True, text=True, timeout=self.timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                return {'success': proc.returncode == 0,
                        'output': proc.stdout[:_OUTPUT_CAP],
                        'error': proc.stderr[:500] if proc.returncode else None,
                        'runtime': lang}
            proc = subprocess.run(
                [exe], input=code, capture_output=True, text=True,
                timeout=self.timeout,
                cwd=None,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            return {'success': proc.returncode == 0,
                    'output': proc.stdout[:_OUTPUT_CAP],
                    'error': proc.stderr[:500] if proc.returncode else None,
                    'runtime': lang}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f"Timeout after {self.timeout:.1f}s",
                    'output': '', 'runtime': lang}
        except Exception as e:
            return {'success': False, 'error': f"{type(e).__name__}: {e}",
                    'output': '', 'runtime': lang}

    # ------------------------------------------------------------------
    # Safe math — a tiny restricted evaluator for numeric expressions.
    # ------------------------------------------------------------------
    def compute_expr(self, expression):
        """Evaluate a numeric expression safely. Returns float/str or None."""
        if not expression or not isinstance(expression, str):
            return None
        expr = re.sub(r'(?i)^(?:what is|whats|what is the value of|what does|'
                      r'what do|calculate|compute|evaluate|solve|find|the '
                      r'value of)\s*', '', expression.rstrip('?')).strip()
        if expr.startswith('='):
            expr = expr[1:].strip()
        if not re.fullmatch(r'[\d\s+\-*/().,%x×÷]+', expr):
            return None
        expr = expr.replace('x', '*').replace('×', '*').replace('÷', '/')
        if '(' in expr and ')' not in expr:
            return None
        try:
            r = eval(expr, {'__builtins__': {}}, {})
        except Exception:
            return None
        if isinstance(r, bool):
            return None
        if isinstance(r, (int, float)):
            return float(r) if isinstance(r, float) else r
        return None
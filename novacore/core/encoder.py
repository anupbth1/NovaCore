"""Core: Random Fourier Features encoder - training-free weight generation."""
import os

import numpy as np


def auto_backend():
    """Auto-select numeric backend: 'cuda' when torch + GPU + free VRAM is
    available (unless NOVACORE_CPU_ONLY=1), else 'cpu'. Everything callable,
    even without torch installed."""
    if os.environ.get('NOVACORE_CPU_ONLY') == '1':
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
                if free >= 256 * 1024 * 1024:
                    return 'cuda'
            except Exception:
                return 'cuda'
        return 'cpu'
    except Exception:
        return 'cpu'


def backend_info():
    """Human-readable auto backend, e.g. 'GPU (Tesla T4, 11.5/15.0 GB free)'."""
    if auto_backend() == 'cuda':
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            return (f"GPU ({torch.cuda.get_device_name(0)}, "
                    f"{free/1024**3:.1f}/{total/1024**3:.1f} GB free)")
        except Exception:
            return "GPU (torch-CUDA)"
    return "CPU (numpy/BLAS, auto threads)"


def release_memory(verbose=False):
    """Auto-free what can be freed right now:
      - CPU: Python GC cycle (returns heap pages to the OS where possible)
      - GPU: torch CUDA reserved-VRAM cache (empty_cache) when available
    Always safe to call (never raises). With verbose=True prints a short
    `[Mem] RAM ... | VRAM ...` line. Returns (ram_gb, vram_gb): current RSS /
    freed VRAM in GB (None when unknown/unavailable)."""
    ram_gb = None
    vram_gb = None
    vram_free = None

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    # Free CUDA's cached blocks (nn libs keep up to 100% of VRAM as cache).
    if os.environ.get('NOVACORE_CPU_ONLY') != '1':
        try:
            import torch
            if torch.cuda.is_available():
                before = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_reserved() / 1024**3
                freed = before - after
                if freed > 0:
                    vram_gb = freed
                try:
                    free, total = torch.cuda.mem_get_info()
                    vram_free = (free / 1024**3, total / 1024**3)
                except Exception:
                    vram_free = None
        except Exception:
            pass

    if verbose:
        try:
            if os.name == 'posix':
                with open('/proc/self/status') as fh:
                    for ln in fh:
                        if ln.startswith('VmRSS:'):
                            ram_gb = int(ln.split()[1]) / 1024**2
                            break
            else:
                try:
                    import psutil
                    ram_gb = psutil.Process().memory_info().rss / 1024**3
                except Exception:
                    ram_gb = None
        except Exception:
            ram_gb = None
        msg = []
        if ram_gb is not None:
            msg.append(f"RAM {ram_gb:.1f}GB")
        if vram_free is not None:
            f, t = vram_free
            msg.append(f"VRAM {f:.1f}/{t:.1f}GB free")
        elif vram_gb is not None:
            msg.append(f"VRAM freed {vram_gb:.2f}GB")
        if msg:
            print("[Mem] " + " | ".join(msg), flush=True)
    return ram_gb, vram_gb


class RandomFourierEncoder:
    """
    Random Fourier Features (RFF) encoder with analytical solution.

    Maps input data to a high-dimensional feature space using random
    Fourier features, then solves for weights analytically via
    pseudo-inverse - NO gradient descent, NO backprop, NO training loop.
    """

    def __init__(self, input_dim=None, dim=None, layers=None, seed=None):
        from ..config import get_default
        self.dim = dim if dim is not None else get_default('dim')
        self.layers = layers if layers is not None else get_default('layers')
        self.seed = seed if seed is not None else get_default('seed')
        if self.seed is None:
            self.seed = 42
        self.input_dim = input_dim if input_dim is not None else get_default('encoder_input_dim')
        self._rng = np.random.RandomState(self.seed)
        self.W = None
        self.b = None
        self.beta = None
        self._init_random_weights()

    def _init_random_weights(self):
        """Initialize random projection weights (the only randomness)."""
        scale = np.sqrt(1.0 / self.input_dim)
        self.W = self._rng.randn(self.input_dim, self.dim) * scale
        self.b = self._rng.uniform(0, 2 * np.pi, size=(1, self.dim))
        self.beta = None

    def encode(self, X):
        """Apply random Fourier feature mapping (non-linear), numpy/BLAS."""
        return self._encode_impl(X, 'cpu')

    def _encode_impl(self, X, backend):
        """Shared RFF mapping for both backends. backend in {'cpu','cuda'}.
        The RandomState draw order is identical on both paths so W/b/layer
        weights stay reproducible regardless of where the matmul runs."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_dim:
            X = self._pad(X)
        X32 = X.astype(np.float32, copy=False)

        if backend == 'cuda':
            import torch
            Xt = torch.as_tensor(X32, device='cuda')
            Wt = torch.as_tensor(self.W.astype(np.float32, copy=False), device='cuda')
            bt = torch.as_tensor(self.b.astype(np.float32, copy=False), device='cuda')
            Z = torch.cos(Xt @ Wt + bt)
            if self.layers > 1:
                scale = np.float32(np.sqrt(1.0 / self.dim))
                W_layer = (self._rng.randn(self.dim, self.dim) * scale).astype(np.float32)
                b_layer = self._rng.uniform(0, 2 * np.pi, size=(1, self.dim)).astype(np.float32)
                Wt2 = torch.as_tensor(W_layer, device='cuda')
                bt2 = torch.as_tensor(b_layer, device='cuda')
                for _ in range(self.layers - 1):
                    Z = torch.cos(Z @ Wt2 + bt2)
            return Z

        W32 = self.W.astype(np.float32, copy=False)
        b32 = self.b.astype(np.float32, copy=False)
        Z = np.cos(X32 @ W32 + b32)
        if self.layers > 1:
            scale = np.float32(np.sqrt(1.0 / self.dim))
            W_layer = (self._rng.randn(self.dim, self.dim) * scale).astype(np.float32)
            b_layer = self._rng.uniform(0, 2 * np.pi, size=(1, self.dim)).astype(np.float32)
            for _ in range(self.layers - 1):
                Z = np.cos(Z @ W_layer + b_layer)
        return Z

    def _fit_analytic_cuda(self, X, Y, lam):
        """Analytic fit on CUDA when a GPU is free (float32, like the CPU path)."""
        import torch
        Z = self._encode_impl(X, 'cuda')
        n, d = Z.shape
        Yf = torch.as_tensor(np.asarray(Y, dtype=np.float32), device='cuda')
        ZtZ = Z.mT @ Z
        ZtZ = ZtZ + lam * torch.eye(d, dtype=torch.float32, device='cuda')
        try:
            L = torch.linalg.cholesky(ZtZ)
            ZtY = Z.mT @ Yf
            beta = torch.linalg.solve(L.mT, torch.linalg.solve(L, ZtY))
        except RuntimeError:
            beta = torch.linalg.solve(ZtZ, Z.mT @ Yf)
        del Z, ZtZ, ZtY
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return beta.cpu().numpy().astype(np.float64)

    def fit_analytic(self, X, Y):
        """
        Fit weights analytically using least-squares (pseudo-inverse).

        beta = (Z^T Z + lambda I)^-1 Z^T Y
        This is the closed-form solution - no iterative training.

        Backend is auto-selected: CUDA (torch) when a GPU + free VRAM is
        present, otherwise numpy/BLAS. Falls back to CPU on any CUDA error.
        """
        from ..config import get_default
        lam = get_default('ridge_lambda')
        if auto_backend() == 'cuda':
            try:
                self.beta = self._fit_analytic_cuda(X, Y, lam)
                return self.beta
            except Exception:
                pass  # graceful CPU fallback (no error spam)
        Z = self._encode_impl(X, 'cpu')
        n, d = Z.shape
        # Use float32 for the heavy matmul (4x faster, plenty of precision)
        Zf = Z.astype(np.float32, copy=False)
        Yf = Y.astype(np.float32, copy=False)
        # Z^T Z: n×d @ d×n -> d×d (but we accumulate via d×n @ n×d for memory)
        ZtZ = Zf.T @ Zf
        ZtZ += lam * np.eye(d, dtype=np.float32)
        # Cholesky-based solve (much faster than np.linalg.solve for SPD)
        try:
            L = np.linalg.cholesky(ZtZ)
            ZtY = Zf.T @ Yf
            self.beta = np.linalg.solve(L.T, np.linalg.solve(L, ZtY))
        except np.linalg.LinAlgError:
            # Fall back if not positive-definite for some numerical reason
            self.beta = np.linalg.solve(ZtZ, Zf.T @ Yf)
        # Cast back to float64 to match downstream code
        self.beta = self.beta.astype(np.float64)
        return self.beta

    def predict(self, X):
        """Forward pass using learned beta."""
        if self.beta is None:
            raise ValueError("Model not fitted. Call fit_analytic() first.")
        Z = self.encode(X)
        return Z @ self.beta

    def _pad(self, X):
        """Pad or truncate input to input_dim."""
        if X.shape[1] < self.input_dim:
            pad = np.zeros((X.shape[0], self.input_dim - X.shape[1]))
            return np.hstack([X, pad])
        return X[:, :self.input_dim]

    def save_weights(self):
        """Return all weights for storage."""
        return {
            "W": self.W,
            "b": self.b,
            "beta": self.beta if self.beta is not None else np.zeros((self.dim, 1)),
            "input_dim": self.input_dim,
            "dim": self.dim,
            "layers": self.layers,
            "seed": self.seed,
        }

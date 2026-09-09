"""Core: Random Fourier Features encoder - training-free weight generation."""
import numpy as np


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
        """Apply random Fourier feature mapping (non-linear)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_dim:
            X = self._pad(X)
        # Use float32 for the heavy matmul (much faster on most CPUs/GPUs)
        X32 = X.astype(np.float32, copy=False)
        W32 = self.W.astype(np.float32, copy=False)
        b32 = self.b.astype(np.float32, copy=False)
        Z = np.cos(X32 @ W32 + b32)
        # For additional layers, use a square random matrix transform
        if self.layers > 1:
            scale = np.float32(np.sqrt(1.0 / self.dim))
            # Reuse same W/b for determinism but cast to float32
            W_layer = (self._rng.randn(self.dim, self.dim) * scale).astype(np.float32)
            b_layer = self._rng.uniform(0, 2 * np.pi, size=(1, self.dim)).astype(np.float32)
            for _ in range(self.layers - 1):
                Z = np.cos(Z @ W_layer + b_layer)
        return Z

    def fit_analytic(self, X, Y):
        """
        Fit weights analytically using least-squares (pseudo-inverse).

        beta = (Z^T Z + lambda I)^-1 Z^T Y
        This is the closed-form solution - no iterative training.

        Uses float32 for the heavy matmul (Z^T Z) and Cholesky decomposition
        for the symmetric positive-definite system (much faster than
        np.linalg.solve for large d).
        """
        Z = self.encode(X)
        from ..config import get_default
        lam = get_default('ridge_lambda')
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

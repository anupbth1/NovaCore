"""Tests for core encoder."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from novacore.core.encoder import RandomFourierEncoder


def test_encoder_shape():
    enc = RandomFourierEncoder(input_dim=32, dim=128, layers=2)
    X = np.random.randn(10, 32)
    Z = enc.encode(X)
    assert Z.shape == (10, 128), f"Expected (10,128), got {Z.shape}"


def test_analytic_fit():
    enc = RandomFourierEncoder(input_dim=32, dim=128, layers=2)
    X = np.random.randn(50, 32)
    Y = X @ np.random.randn(32, 5)
    beta = enc.fit_analytic(X, Y)
    assert beta.shape == (128, 5)
    pred = enc.predict(X[:5])
    assert pred.shape == (5, 5)


def test_deterministic_seed():
    a = RandomFourierEncoder(input_dim=16, dim=64, seed=42)
    b = RandomFourierEncoder(input_dim=16, dim=64, seed=42)
    X = np.random.randn(3, 16)
    assert np.allclose(a.encode(X), b.encode(X)), "Same seed should give same encoding"

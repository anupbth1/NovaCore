"""Tests for tokenizer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from novacore.tokenizer.vocab import Vocabulary
from novacore.tokenizer.text_processor import TextProcessor


def test_vocab_build():
    v = Vocabulary(vocab_size=100)
    n = v.build(["the cat sat", "the dog ran"])
    assert n >= 5, f"Expected at least 5 tokens, got {n}"
    assert "the" in v.token_to_id
    assert v.token_to_id["the"] == v.token_to_id["the"]


def test_encode_decode():
    v = Vocabulary(vocab_size=100)
    v.build(["hello world"])
    ids = v.encode("hello")
    assert ids[0] == v.BOS
    assert ids[-1] == v.EOS
    text = v.decode(ids)
    assert "hello" in text


def test_text_to_vector():
    v = Vocabulary(vocab_size=100)
    v.build(["the cat sat on mat"])
    processor = TextProcessor(vocab=v, dim=256)
    vec = processor.text_to_vector("the cat")
    assert vec.shape == (256,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6  # normalized


import numpy as np

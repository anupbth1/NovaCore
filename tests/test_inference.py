"""Tests for inference."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from novacore.core.patterns import PatternExtractor
from novacore.inference.predictor import PatternPredictor
from novacore.tokenizer.vocab import Vocabulary
from novacore.tokenizer.text_processor import TextProcessor


def test_pattern_extract():
    p = PatternExtractor(ngram_range=(2, 3))
    p.update("the cat sat on the mat")
    p.prune()
    assert len(p.patterns) > 0
    top = p.top_patterns(1)
    # Most frequent n-gram should be "the" repeated patterns
    assert top[0][1] >= 1


def test_pattern_predictor():
    p = PatternExtractor(ngram_range=(2, 3))
    news = [
        "the cat sat on the mat",
        "the cat and the dog",
        "the cat purred",
        "a cat is an animal",
    ]
    p.update_many(news)
    p.prune()
    v = Vocabulary()
    v.build(news)
    proc = TextProcessor(vocab=v, dim=128)
    predictor = PatternPredictor(p, v, proc)
    result = predictor.generate("the cat", max_tokens=5)
    # Should produce some output (words appended)
    assert len(result) > len("the cat")
    assert "cat" in result

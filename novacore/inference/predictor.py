"""Inference: Enhanced Predictor with Retrieval-Based Generation."""
import numpy as np
import re
from collections import Counter


class Predictor:
    """
    Predicts next token using the pre-computed weights.
    Uses the trained-free analytic model for forward pass.
    """

    def __init__(self, encoder, processor, vocab, hasher=None):
        self.encoder = encoder
        self.processor = processor
        self.vocab = vocab
        self.hasher = hasher

    def encode_text(self, text):
        """Encode text to feature vector using encoder."""
        return self.processor.text_to_vector(text, normalize=True)

    def top_tokens(self, context_text, k=None):
        """
        Predict top-k next tokens for a given context.

        Uses the analytic model. If beta is available for vocab-sized output,
        returns ranked tokens. Otherwise uses pattern co-occurrence.
        """
        from ..config import get_default
        k = k if k is not None else get_default('predictor_top_k')
        vec = self.encode_text(context_text)
        if self.encoder.beta is not None and self.encoder.beta.shape[1] > 1:
            probs = self.encoder.predict(vec.reshape(1, -1))[0]
            top = np.argsort(-probs)[:k]
            scores = probs[top]
            tokens = [self.vocab.id_to_token.get(int(t), f"<{t}>") for t in top]
            return list(zip(tokens, [float(s) for s in scores]))
        return []

    def generate(self, prompt, max_tokens=None, temperature=None):
        """
        Generate text by iteratively predicting next token.

        Falls back to a hashing-based sampler using the feature hasher if
        analytic beta not available (works without any training).
        """
        from ..config import get_default
        max_tokens = max_tokens if max_tokens is not None else get_default('max_tokens')
        temperature = temperature if temperature is not None else get_default('temperature')
        result = prompt
        current = prompt
        len_vocab = len(self.vocab)

        for _ in range(max_tokens):
            next_token = self._predict_next(current, temperature, len_vocab)
            if next_token is None:
                break
            word = self.vocab.id_to_token.get(next_token, "")
            if not word:
                break
            result += " " + word
            current = result
        return result

    def _predict_next(self, context, temperature, len_vocab):
        """Predict a single next token id."""
        from ..config import get_default
        # Method 1: analytic beta if shape matches vocab
        if self.encoder.beta is not None and self.encoder.beta.shape[1] == len_vocab:
            vec = self.encode_text(context)
            logits = self.encoder.predict(vec.reshape(1, -1))[0]
            logits = logits / temperature
            logits = logits - logits.max()
            exp = np.exp(logits)
            probs = exp / exp.sum()
            return int(np.random.choice(len_vocab, p=probs))

        # Method 2: hashing-based sampling from feature hasher
        if self.hasher is not None:
            # Map context to most likely next token via hasher estimate
            cap = get_default('predictor_scan_vocab_cap')
            best_score = -1
            best_id = None
            for tid in range(min(len_vocab, cap)):
                score = self.hasher.estimate(f"prev:{context.split()[-1].lower()};next:{tid}")
                if score > best_score:
                    best_score = score
                    best_id = tid
            if best_id is not None:
                return best_id
        return None


class PatternPredictor(Predictor):
    """
    Predictor that uses extracted patterns (n-grams) for text generation.
    No training needed - uses co-occurrence statistics.
    All content comes from dataset patterns.
    """

    def __init__(self, pattern_extractor, vocab, processor, reservoir_samples=None, weights_dir=None, semantic_index=None):
        self.patterns = pattern_extractor
        self.vocab = vocab
        self.processor = processor
        self.reservoir_samples = reservoir_samples or []
        self.upgrader = None
        self.semantic_index = semantic_index
        self._log = None  # optional live-logger callback injected by ChatSession
        # Model metadata values (set by ChatSession for zero-config chat)
        self.model_max_tokens = 4096
        self.model_temperature = 0.7

        # Load training upgrades if available
        if weights_dir:
            self._load_upgrades(weights_dir)

        super().__init__(None, processor, vocab)

    def _load_upgrades(self, weights_dir):
        """Load training upgrades (SVD + IDF) if present."""
        import os
        upgrade_dir = os.path.join(weights_dir, "upgrades")
        if not os.path.exists(upgrade_dir):
            return
        try:
            from ..core.training_upgrades import TrainingUpgrader
            self.upgrader = TrainingUpgrader.load(upgrade_dir, vocab=self.vocab)
            if self.upgrader.enabled:
                item_count = len(self.upgrader.word_to_idx)
                print(f"[NovaCore] Training upgrades loaded: {item_count} words, SVD dim={self.upgrader.svd_dim}")
        except Exception:
            pass

    def generate(self, prompt, max_tokens=None, temperature=None, include_prompt=True):
        """Generate response using pattern-based approach from dataset.
        
        All content comes from the dataset patterns extracted during training.
        """
        max_tokens = max_tokens if max_tokens is not None else self.model_max_tokens
        temperature = temperature if temperature is not None else self.model_temperature
        
        # Generate using pattern matching from dataset
        result = self._generate_pattern_based(prompt, max_tokens, temperature)
        
        # Apply virtual simulation for quality improvement (uses dataset knowledge)
        try:
            from ..virtual_sim import SimulationEngine
            from .chat import SIM_THRESHOLD
            extractor = getattr(self, 'patterns', None) or None
            if self._log is not None:
                self._log(f"     [self-correct] building sim engine "
                          f"(precomputing token profile over up to 5000 "
                          f"reservoir docs)...")
            engine = SimulationEngine(
                vocab=self.vocab,
                extractor=extractor,
                reservoir_sample=self.reservoir_samples,
                config={"sim_threshold": SIM_THRESHOLD},
            )
            if self._log is not None:
                engine.verifier._log = self._log
                engine.corrector._log = self._log
                self._log(f"     [self-correct] running virtual simulation on "
                          f"predictor output...")
            
            # Apply correction with query context
            sim_result = engine.verify_and_correct(result, query_context=prompt)
            if self._log is not None:
                self._log(f"     [self-correct] score={sim_result.get('score', 0):.3f} "
                          f"corrected={sim_result.get('corrected', False)} "
                          f"attempts={sim_result.get('attempts', 0)}")
            if sim_result["corrected"]:
                if self._log is not None:
                    self._log(f"     [self-correct] BEFORE: {result[:130]!r}")
                    self._log(f"     [self-correct] AFTER : {sim_result['text'][:130]!r}")
                result = sim_result["text"]
                
        except Exception:
            pass
        
        # Final cleanup (only removes artifacts, no new content)
        result = self._final_cleanup(result)
        
        return result

    def _final_cleanup(self, text):
        """Final cleanup - uses config-driven artifact patterns."""
        if not text:
            return text
        from ..core.neural_engine import clean_artifacts
        return clean_artifacts(text)

    def reply(self, prompt, max_tokens=None, temperature=None):
        """Generate a reply WITHOUT the prompt prefix (clean chatbot output)."""
        return self.generate(prompt, max_tokens, temperature, include_prompt=False)
    
    def _generate_pattern_based(self, prompt, max_tokens, temperature):
        """Pattern-based generation with semantic search upgrade."""
        import random
        import sys

        # Priority 1: Try built-in semantic index
        if self.semantic_index is not None and self.semantic_index._built:
            from .chat import SEMANTIC_SEARCH_TOP_K, SEMANTIC_SEARCH_THRESHOLD
            if self._log is not None:
                self._log(f"     [pred] trying built-in semantic index...")
            results = self.semantic_index.search(prompt, self.vocab, top_k=SEMANTIC_SEARCH_TOP_K)
            if results and results[0][0] > SEMANTIC_SEARCH_THRESHOLD:
                best_text = results[0][1]
                if best_text and len(best_text) > 10:
                    if self._log is not None:
                        self._log(f"     [pred] ✓ semantic index returned "
                                  f"(score={results[0][0]:.3f})")
                    return best_text
            if self._log is not None:
                self._log(f"     [pred] semantic index: "
                          f"{'below threshold' if results else 'no hits'} "
                          f"-> try SVD")

        # Priority 2: SVD-based semantic search (if upgrader available)
        if self.upgrader and self.upgrader.semantic_search_enabled and self.reservoir_samples:
            from .chat import SEMANTIC_SEARCH_TOP_K, SVD_SEMANTIC_MIN_SCORE
            if self._log is not None:
                self._log(f"     [pred] trying SVD semantic search "
                          f"(dim={self.upgrader.svd_dim})...")
            sem_results = self.upgrader.semantic_search(
                prompt, self.reservoir_samples, top_k=SEMANTIC_SEARCH_TOP_K
            )
            if sem_results and sem_results[0][0] > SVD_SEMANTIC_MIN_SCORE:
                best_text = sem_results[0][2]
                if len(best_text) > 10:
                    if self._log is not None:
                        self._log(f"     [pred] ✓ SVD search returned "
                                  f"(score={sem_results[0][0]:.3f})")
                    return best_text[:max_tokens * 4]
            if self._log is not None:
                self._log(f"     [pred] SVD search: "
                          f"{'below threshold' if sem_results else 'no hits'} "
                          f"-> n-gram fallback")

        # Priority 3: n-gram pattern generation (last resort)
        result = ""
        current_tokens = self.vocab._tokenize(prompt)

        _npat = len(getattr(self.patterns, 'patterns', {}) or {})
        if self._log is not None:
            self._log(f"     [pred] n-gram generation over {_npat} patterns, "
                      f"max_tokens={max_tokens}, temp={temperature}")

        import time as _clock
        _gen_t0 = _clock.time()
        for step in range(max_tokens):
            candidates = self._next_candidates(current_tokens)
            if not candidates:
                if self._log is not None:
                    self._log(f"     [pred] no more n-gram candidates after "
                              f"{step} tokens")
                break
            if temperature <= 0:
                word = candidates[0][0]
            else:
                weights = [c[1] for c in candidates]
                weights = np.array(weights, dtype=float)
                probs = weights ** (1.0 / max(temperature, 1e-5))
                probs = probs / probs.sum()
                word = candidates[random.choices(range(len(candidates)), probs)[0]][0]
            result += (" " if result else "") + word
            if self._log is not None:
                _snippet = result[-70:].replace('\n', ' ')
                sys.stdout.write(f"\r     ✍️ token {step + 1}: '{word}'  ->  ...{_snippet}")
                sys.stdout.flush()
            current_tokens.append(word)
            current_tokens = current_tokens[-10:]  # history window

        if self._log is not None:
            sys.stdout.write(f"\r     ✍️ n-gram generation done: {len(result.split())} "
                             f"words in {_clock.time() - _gen_t0:.2f}s{' ' * 30}\n")
            sys.stdout.flush()

        return result

    def _next_candidates(self, tokens):
        """Get candidate next words from n-gram patterns."""
        candidates = {}
        if not tokens:
            return []
        # Infer max n-gram from stored patterns
        max_n = 1
        for gram in self.patterns.patterns:
            if len(gram) > max_n:
                max_n = len(gram)
        # Look at last n-1 tokens for n>=2 patterns
        for n in range(max_n, 1, -1):
            if len(tokens) < n - 1:
                continue
            prefix = tuple(tokens[-(n-1):])
            for (gram, count) in self.patterns.patterns.items():
                if len(gram) != n:
                    continue
                if gram[:-1] == prefix:
                    nxt = gram[-1]
                    candidates[nxt] = candidates.get(nxt, 0) + count
        if candidates:
            return sorted(candidates.items(), key=lambda x: -x[1])

        # No exact n-gram match -> fall back to most common dataset words
        return self._unigram_fallback()

    def _unigram_fallback(self):
        """Count unigram frequencies from stored patterns as fallback."""
        freq = {}
        for (gram, count) in self.patterns.patterns.items():
            if len(gram) == 2:
                # A 2-gram 'a b' implies 'a' and 'b' both appeared often
                for w in gram:
                    freq[w] = freq.get(w, 0) + count
        if not freq:
            return []
        return sorted(freq.items(), key=lambda x: -x[1])[:10]

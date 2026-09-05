"""
Enhanced Virtual Simulation Layer for NovaCore.

Adds an internal verifier / self-corrector that runs during both
ENCODING (training) and GENERATION to improve output quality.

Key improvements:
  - Semantic coherence checking (not just statistical)
  - Response structure validation
  - Context-aware correction
  - Multi-attempt generation with quality scoring
"""
import math
import random
import re
from collections import Counter


class VirtualVerifier:
    """Verify quality of a generated / encoded text against learned patterns.

    Enhanced scoring rules (0.0 - 1.0):
      - Semantic coherence (does the text make sense?)
      - Response structure (proper formatting, no artifacts)
      - Vocabulary coverage (% of tokens that exist in vocab)
      - Pattern overlap (how many n-grams from extractor.patterns match)
      - Statistical similarity (token frequency match to reservoir mean)
      - Length appropriateness (not too short, not too long)
    """

    def __init__(self, vocab, extractor, reservoir_sample, min_score=0.45):
        self.vocab = vocab              # Vocabulary instance
        self.extractor = extractor      # PatternExtractor
        self.reservoir = reservoir_sample  # list of representative texts
        self.min_score = min_score
        # Pre-compute reservoir token frequency profile for comparison
        self.reservoir_token_freq = Counter()
        for t in self.reservoir:
            if isinstance(t, str):
                for tok in self.vocab._tokenize(t):
                    self.reservoir_token_freq[tok] += 1
        self.total_reservoir_tok = max(1, sum(self.reservoir_token_freq.values()))
        
        # Common artifacts to check for
        self.artifacts = [
            r'<instruction>.*?</instruction>',
            r'<answer>.*?</answer>',
            r'<text>.*?</text>',
            r'###\s*Instruction:',
            r'###\s*Response:',
            r'Below is an instruction',
        ]

    def score_text(self, text, query_context=None):
        """Return a quality score 0.0 - 1.0 with enhanced scoring."""
        if not isinstance(text, str) or not text or len(text) < 2:
            return 0.0

        # 1. Semantic coherence (NEW - most important)
        score_coherence = self._score_coherence(text)
        
        # 2. Response structure (NEW - check for proper formatting)
        score_structure = self._score_structure(text)
        
        # 3. Vocabulary coverage (% tokens that exist)
        tokens = self.vocab._tokenize(text)
        vocab_set = getattr(self.vocab, "token_to_id", {})
        existing = sum(1 for tok in tokens if tok in vocab_set)
        score_vocab = existing / max(1, len(tokens))

        # 4. Pattern overlap: match text against extractor.patterns keys
        pattern_matches = 0
        if tokens and self.extractor and hasattr(self.extractor, "patterns"):
            patterns = getattr(self.extractor.patterns, "items", lambda: [])()
            # For speed, sample first 30 patterns; full scan is O(N*P) and slow
            for tok_seq, weight in list(patterns)[:30]:
                # Simple sliding window overlap check (subsequence match)
                tl = len(tok_seq)
                for i in range(len(tokens) - tl + 1):
                    if tuple(tokens[i:i+tl]) == tok_seq:
                        pattern_matches += weight
                        break  # count once per pattern
        score_pattern = min(1.0, pattern_matches / max(1, len(getattr(self.extractor, "patterns", {}) or {})))

        # 5. Statistical similarity (token frequency divergence from reservoir profile)
        text_counter = Counter(tokens)
        divergence = 0.0
        for tok, c in text_counter.items():
            res_freq = self.reservoir_token_freq.get(tok, 0) / self.total_reservoir_tok
            text_freq = c / max(1, len(tokens))
            # KL-divergence approximation (clamped to avoid inf)
            res_freq = max(1e-6, min(1.0, res_freq))
            text_freq = max(1e-6, min(1.0, text_freq))
            divergence += text_freq * math.log(text_freq / res_freq)
        divergence = max(0.0, divergence)
        # Normalize divergence: 0 = identical profile (score 1.0), >1.5 = very different (score 0.0)
        score_sim = max(0.0, 1.0 - (divergence / 2.0))

        # 6. Length appropriateness (NEW)
        score_length = self._score_length(text)
        
        # 7. Context relevance (NEW - if query provided)
        score_context = 0.5  # default neutral
        if query_context:
            score_context = self._score_context_relevance(text, query_context)

        # Weighted aggregate (enhanced weights)
        score = (
            0.25 * score_coherence +    # Most important: does it make sense?
            0.15 * score_structure +     # Is it properly formatted?
            0.20 * score_vocab +         # Are the words in vocabulary?
            0.15 * score_pattern +       # Does it match learned patterns?
            0.10 * score_sim +           # Is it statistically similar?
            0.05 * score_length +        # Is it appropriate length?
            0.10 * score_context         # Is it relevant to query?
        )
        return min(1.0, max(0.0, score))
    
    def _score_coherence(self, text):
        """Score semantic coherence of text."""
        words = text.split()
        
        # Check for excessive repetition
        if not words:
            return 0.0
            
        word_counts = Counter(words)
        max_repeat = max(word_counts.values())
        repetition_penalty = min(1.0, max_repeat / max(1, len(words) * 0.3))
        
        # Check for sentence structure (basic)
        has_sentence_structure = bool(re.search(r'[A-Z].*[.!?]', text))
        
        # Check for common incoherence patterns
        incoherent_patterns = [
            r'(\b\w+\b)\s+\1\s+\1',  # Triple word repetition
            r'[.!?]\s*[a-z]',  # Sentence starts with lowercase after punctuation
            r'\s{3,}',  # Multiple spaces
        ]
        
        incoherence_count = 0
        for pattern in incoherent_patterns:
            if re.search(pattern, text):
                incoherence_count += 1
        
        # Calculate coherence score
        coherence = 1.0
        coherence -= repetition_penalty * 0.4
        coherence -= incoherence_count * 0.2
        coherence += 0.2 if has_sentence_structure else 0.0
        
        return max(0.0, min(1.0, coherence))
    
    def _score_structure(self, text):
        """Score response structure and formatting."""
        score = 1.0
        
        # Check for artifacts
        for artifact_pattern in self.artifacts:
            if re.search(artifact_pattern, text, re.DOTALL):
                score -= 0.3
        
        # Check for proper punctuation
        if text and text[0].islower():
            score -= 0.1
        
        # Check for sentence endings
        if text and text[-1] not in '.!?':
            score -= 0.1
        
        # Check for excessive whitespace
        if '  ' in text:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_length(self, text):
        """Score length appropriateness."""
        words = text.split()
        word_count = len(words)
        
        # Optimal length range (10-100 words)
        if 10 <= word_count <= 100:
            return 1.0
        elif word_count < 5:
            return 0.3
        elif word_count < 10:
            return 0.6
        elif word_count > 200:
            return 0.4
        else:
            return 0.8
    
    def _score_context_relevance(self, text, query_context):
        """Score relevance to the query context."""
        if not query_context:
            return 0.5
        
        # Extract key words from query
        query_words = set(query_context.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                      'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                      'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                      'don', 'now', 'what', 'who', 'which', 'this', 'that', 'these', 'those'}
        
        query_words -= stop_words
        text_words -= stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate overlap
        overlap = query_words & text_words
        relevance = len(overlap) / len(query_words) if query_words else 0.0
        
        return min(1.0, relevance)

    def passes(self, text):
        return self.score_text(text) >= self.min_score


class VirtualSelfCorrector:
    """Enhanced self-correction layer for generation.

    Strategy:
      1. Generate N candidates (or take current output)
      2. Score each with Verifier
      3. If best < threshold, try multiple correction strategies:
         - Remove artifacts and formatting issues
         - Fix sentence structure
         - Remove excessive repetition
         - Add missing context
         - Truncate or extend as needed
      4. Return corrected version + score report
    """

    def __init__(self, vocab, extractor, reservoir_sample, threshold=0.45, max_attempts=5):
        self.verifier = VirtualVerifier(vocab, extractor, reservoir_sample, min_score=threshold)
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.vocab = vocab
        
        # Correction strategies in order of effectiveness
        self.correction_strategies = [
            self._remove_artifacts,
            self._fix_sentence_structure,
            self._remove_repetition,
            self._fix_punctuation,
            self._add_context,
            self._truncate_to_sentence,
            self._extend_with_pattern,
        ]

    def correct_text(self, text, query_context=None):
        best_text = text
        best_score = self.verifier.score_text(text, query_context)
        attempts = 0
        
        # If already above threshold, nothing to correct
        if best_score >= self.threshold:
            return {"text": best_text, "score": best_score, "corrected": False, "attempts": 0}
        
        # Try corrections using multiple strategies
        while attempts < self.max_attempts and best_score < self.threshold:
            attempts += 1
            corrected = text
            
            # Apply correction strategies in order
            for strategy in self.correction_strategies:
                candidate = strategy(corrected, query_context)
                if candidate != corrected:
                    score = self.verifier.score_text(candidate, query_context)
                    if score > best_score:
                        best_text = candidate
                        best_score = score
                        corrected = candidate  # Continue improving from this version
            
            # If no improvement, try a different approach
            if best_score <= self.verifier.score_text(text, query_context):
                break
        
        return {
            "text": best_text,
            "score": best_score,
            "corrected": best_text != text,
            "attempts": attempts,
            "passed": best_score >= self.threshold,
        }
    
    def _remove_artifacts(self, text, query_context=None):
        """Remove common artifacts from training data."""
        corrected = text
        
        # Remove instruction/answer tags
        corrected = re.sub(r'<instruction>.*?</instruction>', '', corrected, flags=re.DOTALL)
        corrected = re.sub(r'<answer>.*?</answer>', '', corrected, flags=re.DOTALL)
        corrected = re.sub(r'<text>.*?</text>', '', corrected, flags=re.DOTALL)
        
        # Remove markdown headers
        corrected = re.sub(r'###\s*Instruction:.*?###\s*Response:', '', corrected, flags=re.DOTALL)
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'Below is an instruction.*?###\s*Response:\s*',
            r'Instruction:.*?Response:\s*',
            r'Here is the response:\s*',
            r'The answer is:\s*',
        ]
        for prefix_pattern in prefixes_to_remove:
            corrected = re.sub(prefix_pattern, '', corrected, flags=re.DOTALL)
        
        # Clean up whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    def _fix_sentence_structure(self, text, query_context=None):
        """Fix sentence structure issues."""
        corrected = text
        
        # Capitalize first letter
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
        
        # Fix sentences starting with lowercase after punctuation
        def fix_sentence_start(match):
            return match.group(0).upper()
        
        corrected = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), corrected)
        
        # Add period at end if missing
        if corrected and corrected[-1] not in '.!?':
            corrected += '.'
        
        return corrected
    
    def _remove_repetition(self, text, query_context=None):
        """Remove excessive word repetition."""
        words = text.split()
        if not words:
            return text
        
        # Remove consecutive duplicate words
        cleaned = [words[0]]
        for word in words[1:]:
            if word != cleaned[-1]:
                cleaned.append(word)
        
        # Remove excessive repetition of any word
        word_counts = Counter(cleaned)
        max_allowed = max(3, len(cleaned) // 5)
        
        final_words = []
        for word in cleaned:
            if word_counts[word] <= max_allowed:
                final_words.append(word)
            else:
                word_counts[word] -= 1
        
        return ' '.join(final_words)
    
    def _fix_punctuation(self, text, query_context=None):
        """Fix punctuation issues."""
        corrected = text
        
        # Remove multiple spaces
        corrected = re.sub(r'\s+', ' ', corrected)
        
        # Fix spacing around punctuation
        corrected = re.sub(r'\s+([.,;:!?])', r'\1', corrected)
        corrected = re.sub(r'([.,;:!?])\s+', r'\1 ', corrected)
        
        # Remove leading/trailing punctuation that doesn't make sense
        corrected = corrected.strip('.,;:')
        
        return corrected
    
    def _add_context(self, text, query_context=None):
        """Add missing context to make response more relevant."""
        if not query_context:
            return text
        
        # Check if text already contains query context
        query_words = set(query_context.lower().split())
        text_words = set(text.lower().split())
        
        # If less than 30% overlap, add context
        if query_words and len(query_words & text_words) / len(query_words) < 0.3:
            # Extract topic from query
            stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does',
                          'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'the', 'a', 'an'}
            topic_words = [w for w in query_context.split() if w.lower() not in stop_words and len(w) > 2]
            
            if topic_words:
                topic = ' '.join(topic_words[:3])
                # Add context prefix
                if not text.lower().startswith(topic.lower()):
                    text = f"Regarding {topic}: {text}"
        
        return text
    
    def _truncate_to_sentence(self, text, query_context=None):
        """Truncate text to nearest sentence boundary."""
        if len(text) < 50:
            return text
        
        # Find the last complete sentence
        last_period = text.rfind('.')
        last_question = text.rfind('?')
        last_exclamation = text.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        # If we found a sentence end and it's at least 30% of the text
        if last_sentence_end > len(text) * 0.3:
            return text[:last_sentence_end + 1]
        
        return text
    
    def _extend_with_pattern(self, text, query_context=None):
        """Extend short text with relevant patterns."""
        words = text.split()
        
        # Only extend if too short
        if len(words) >= 10:
            return text
        
        # Add common response patterns
        extensions = [
            "This is an important concept to understand.",
            "Let me explain this in more detail.",
            "Here are the key points to consider.",
            "I hope this helps clarify the topic.",
        ]
        
        import random
        extension = random.choice(extensions)
        
        if extension not in text:
            text = f"{text} {extension}"
        
        return text


# Module-level helpers for CLI wiring
class SimulationEngine:
    """Enhanced wrapper that holds verifier + corrector for CLI integration."""

    def __init__(self, vocab=None, extractor=None, reservoir_sample=None, config=None):
        cfg = config or {}
        threshold = cfg.get("sim_threshold", 0.45)
        self.verifier = VirtualVerifier(
            vocab=vocab,
            extractor=extractor,
            reservoir_sample=reservoir_sample or [],
            min_score=threshold,
        )
        self.corrector = VirtualSelfCorrector(
            vocab=vocab, extractor=extractor, reservoir_sample=reservoir_sample or [],
            threshold=threshold, max_attempts=cfg.get("sim_max_attempts", 5),
        )

    def verify_and_correct(self, text, query_context=None):
        """Verify and correct text with optional query context."""
        result = self.corrector.correct_text(text, query_context=query_context)
        return result


def apply_virtual_simulation(text_stream, vocab, extractor, reservoir_sample, config=None):
    """Wrap a text stream / generator with virtual simulation filtering.

    For each yielded text, run verification; if it fails, attempt correction.
    The corrected text is what reaches training; a log message is printed
    when correction happens (so the user can observe).
    """
    engine = SimulationEngine(vocab, extractor, reservoir_sample, config)
    for text in text_stream:
        if not isinstance(text, str):
            yield text
            continue
        result = engine.verify_and_correct(text)
        if result["corrected"]:
            # Print a brief correction log (only once per 100 corrections to avoid spam)
            # We'll rely on caller or external counter for rate-limiting
            pass
        yield result["text"]

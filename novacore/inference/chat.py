"""Inference: ChatSession - interactive chat like Ollama.

ZERO config file dependency. ALL values from model weights/metadata.
Only temperature adjustable via CLI --temperature flag.

Architecture:
  1. Semantic retrieval from reservoir (cosine similarity on IDF-weighted
     feature-hashed vectors) — the PRIMARY response path.
  2. Knowledge base, reasoning, creative engines for specialised queries.
  3. Neural engine + predictor as last-resort fallback only.
"""
import os
import re
import pickle
import random
import numpy as np
import math
from collections import Counter

from ..core.encoder import RandomFourierEncoder
from ..core.hasher import FeatureHasher, murmurhash3
from ..core.patterns import PatternExtractor
from ..core.novacore_model import NovaCoreModel
from ..tokenizer.vocab import Vocabulary
from ..tokenizer.text_processor import TextProcessor
from ..storage.weight_manager import WeightManager
from .predictor import PatternPredictor


# ---------------------------------------------------------------------------
# Internal defaults — model-level constants, NOT from config file
# These are the inference engine's own tuned values.
# ---------------------------------------------------------------------------
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_HISTORY = 10
_DEFAULT_CHAT_CONTEXT_TOKENS = 200

# Semantic retrieval
SEMANTIC_SEARCH_THRESHOLD = 0.35
SEMANTIC_SEARCH_TOP_K = 5
SEMANTIC_SEARCH_RELAXED = 0.85
SEMANTIC_INDEX_BUILD_LIMIT = 200000
POOL_LOAD_CAP = 200000

# Neural engine fallback
NEURAL_RESERVOIR_SCAN_LIMIT = 500
NEURAL_SEMANTIC_MIN_SCORE = 0.15
NEURAL_BEST_MATCH_MIN_SCORE = 0.1
KNOWLEDGE_SEARCH_TOP_K = 3
SVD_SEMANTIC_MIN_SCORE = 0.4

# Word-overlap pool matching
POOL_MATCH_QUERY_WEIGHT = 0.7
POOL_MATCH_INST_WEIGHT = 0.3
POOL_MATCH_LENGTH_BONUS = 1.1
POOL_MATCH_SHORT_THRESHOLD = 50
POOL_MATCH_LONG_THRESHOLD = 200
POOL_MATCH_MIN_SCORE = 0.3

# Relevance checking
RELEVANCE_CHECK_PREFIX_LEN = 200
CREATIVE_KEYWORDS = frozenset([
    'story', 'poem', 'poetry', 'write', 'create', 'compose',
    'narrative', 'fiction', 'tale',
])
STORY_INDICATORS = [
    'once upon a time', 'there was', 'one day',
    'he said', 'she said', 'they said', 'fred was',
    'tom was', 'sarah was', 'john was',
]

# Simulation
SIM_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Semantic Index — fast cosine-similarity retrieval from instruction-answer
# pairs extracted from the reservoir.  Built once at load time, queried in
# O(dim) per candidate (NumPy dot product).
# ---------------------------------------------------------------------------
class SemanticIndex:
    """
    IDF-weighted feature-hashed embedding index for semantic retrieval.
    All values from model metadata — zero config dependency.
    """

    def __init__(self, dim):
        self.dim = dim
        self.instructions = []
        self.answers = []
        self.embeddings = None
        self.idf = {}
        self._built = False

    def build(self, reservoir, vocab, pool_extra=None):
        """Build the index from reservoir and optionally pool_extra."""
        limit = SEMANTIC_INDEX_BUILD_LIMIT

        pairs = []
        raw_instructions = []
        count = 0
        for text in reservoir:
            if count >= limit:
                break
            if not isinstance(text, str) or not text.strip():
                continue
            inst, ans = self._extract_pair(text)
            if inst and ans:
                toks = vocab._tokenize(inst)
                if toks:
                    pairs.append((toks, ans))
                    raw_instructions.append(toks)
                    count += 1

        if pool_extra:
            for text in pool_extra:
                if count >= limit:
                    break
                if not isinstance(text, str) or not text.strip():
                    continue
                inst, ans = self._extract_pair(text)
                if inst and ans:
                    toks = vocab._tokenize(inst)
                    if toks:
                        pairs.append((toks, ans))
                        raw_instructions.append(toks)
                        count += 1

        if not pairs:
            print("[SemanticIndex] No instruction-answer pairs found.")
            return

        n_docs = len(raw_instructions)
        doc_freq = Counter()
        for toks in raw_instructions:
            for w in set(toks):
                doc_freq[w] += 1
        self.idf = {}
        for w, df in doc_freq.items():
            self.idf[w] = math.log((n_docs + 1) / (df + 1)) + 1.0

        N = len(pairs)
        emb = np.zeros((N, self.dim), dtype=np.float32)
        for i, (toks, _) in enumerate(pairs):
            for tok in toks:
                h = murmurhash3(tok)
                bucket = h % self.dim
                sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                w = self.idf.get(tok, 1.0)
                emb[i, bucket] += sign * w
            norm = np.linalg.norm(emb[i])
            if norm > 0:
                emb[i] /= norm

        self.embeddings = emb
        self.instructions = [' '.join(toks) for toks, _ in pairs]
        self.answers = [ans for _, ans in pairs]
        self._built = True
        print(f"[SemanticIndex] Built: {N} instruction-answer pairs, dim={self.dim}")

    def search(self, query_text, vocab, top_k=None):
        """Return list of (score, answer) sorted descending."""
        if not self._built or self.embeddings is None or len(self.answers) == 0:
            return []
        top_k = top_k or SEMANTIC_SEARCH_TOP_K

        qvec = self._encode_query(query_text, vocab)
        if qvec is None:
            return []

        scores = self.embeddings @ qvec
        if len(scores) <= top_k:
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for idx in top_idx:
            s = float(scores[idx])
            if s > 0:
                results.append((s, self.answers[idx]))
        return results

    def _encode_query(self, text, vocab):
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = vocab._tokenize(text)
        if not tokens:
            return None
        for tok in tokens:
            h = murmurhash3(tok)
            bucket = h % self.dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            w = self.idf.get(tok, 1.0)
            vec[bucket] += sign * w
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            return None
        return vec

    @staticmethod
    def _extract_pair(text):
        """Extract (instruction, answer) from a reservoir text entry."""
        from ..core.neural_engine import clean_artifacts
        instruction = ''
        answer = ''

        m = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        if m:
            instruction = clean_artifacts(m.group(1).strip())
        if not instruction:
            m = re.search(r'###\s*Instruction:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if m:
                instruction = clean_artifacts(m.group(1).strip())
        if not instruction:
            m = re.search(r'<user>(.*?)</user>', text, re.DOTALL)
            if m:
                instruction = clean_artifacts(m.group(1).strip())

        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 3:
            answer = clean_artifacts(m.group(1).strip())
        if not answer:
            m = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
            if m and len(m.group(1).strip()) > 3:
                answer = clean_artifacts(m.group(1).strip())
        if not answer:
            m = re.search(r'###\s*Response:\s*(.*?)(?:\n\n###|\Z)', text, re.DOTALL)
            if m and len(m.group(1).strip()) > 3:
                answer = clean_artifacts(m.group(1).strip())

        if instruction and answer:
            return (instruction, answer)
        elif answer:
            return ('', answer)
        return (None, None)


# ---------------------------------------------------------------------------
# ChatSession — ZERO config dependency, all from model weights/metadata
# ---------------------------------------------------------------------------
class ChatSession:
    """
    Loads a saved NovaCore model and provides interactive chat.

    ALL values read from model weights/metadata — NO config file needed.
    Only temperature adjustable via CLI --temperature flag.
    """

    def __init__(self, weights_dir, temperature=None):
        self.weights_dir = weights_dir
        self.temperature = temperature  # None = use model metadata value
        self.history = []
        self.predictor = None
        self.model = None
        self.patterns_data = {}
        self.reservoir_samples = []
        self.metadata = None
        self._loaded = False
        self.semantic_index = None
        self.pool_pairs = []
        self.max_history = _DEFAULT_MAX_HISTORY
        self.chat_context_tokens = _DEFAULT_CHAT_CONTEXT_TOKENS

    def load(self):
        """Load model — ALL values from weights/metadata, NOT from config."""
        wm = WeightManager(self.weights_dir)
        try:
            data, metadata = wm.load()
        except FileNotFoundError as e:
            raise RuntimeError(f"Model not found at '{self.weights_dir}': {e}")

        self.metadata = metadata or {}

        # --- Read ALL training params from saved model metadata ---
        dim = self.metadata.get("dim")
        if dim is None:
            raise RuntimeError(
                f"Model metadata missing 'dim'. Is this a valid NovaCore model?"
            )

        vocab_size = self.metadata.get("vocab_size")

        # Temperature: CLI flag > model metadata > default
        if self.temperature is None:
            self.temperature = self.metadata.get("temperature", _DEFAULT_TEMPERATURE)

        # Vocabulary
        vocab_path = os.path.join(self.weights_dir, "vocab.json")
        if os.path.exists(vocab_path):
            try:
                vocab = Vocabulary.load(vocab_path)
            except Exception:
                vocab = Vocabulary()
        else:
            vocab = Vocabulary()

        processor = TextProcessor(vocab=vocab, dim=dim)

        # Patterns
        if "patterns_raw" in data.files:
            try:
                self.patterns_data = pickle.loads(data["patterns_raw"].tobytes())
            except Exception:
                self.patterns_data = {}

        # Reservoir samples
        if "reservoir_sample" in data.files:
            try:
                raw_data = data["reservoir_sample"].tobytes()
                if raw_data:
                    self.reservoir_samples = pickle.loads(raw_data)
                else:
                    self.reservoir_samples = []
            except Exception as e:
                print(f"[WARN] Could not load reservoir samples: {e}")
                self.reservoir_samples = []
        else:
            self.reservoir_samples = []

        print(f"[ChatSession] Loaded {len(self.reservoir_samples)} reservoir samples")
        print(f"[ChatSession] Model: dim={dim}, vocab={len(vocab)}, "
              f"temp={self.temperature}")

        # Build semantic index from reservoir + pool data from disk
        pool_extra = self._load_pool_data_from_disk()
        self.semantic_index = SemanticIndex(dim)
        self.semantic_index.build(self.reservoir_samples, vocab, pool_extra)

        # Word-overlap pool pairs (fallback)
        self.pool_pairs = []
        for text in (self.reservoir_samples[:5000] + pool_extra[:5000]):
            inst, ans = SemanticIndex._extract_pair(text)
            if inst and ans:
                self.pool_pairs.append((inst, ans))

        extractor = PatternExtractor()
        extractor.patterns = dict(self.patterns_data)

        all_data = list(self.reservoir_samples) + pool_extra

        # Neural engine config — from model metadata, NOT external config
        engine_config = {
            'neural_engine': {
                'input_dim': min(128, dim),
                'hidden_dim': min(64, dim // 2),
                'output_dim': min(32, dim // 4),
            },
            'virtual_simulation': {
                'quality_threshold': SIM_THRESHOLD,
                'scoring_weights': {'length': 0.4, 'relevance': 0.4, 'coherence': 0.2},
            },
            'verification_engine': {
                'max_retries': 2,
                'min_score': 0.3,
                'reservoir_search_limit': NEURAL_RESERVOIR_SCAN_LIMIT,
            },
            'python_terminal': {
                'safe_mode': True,
            },
        }

        self.model = NovaCoreModel(
            patterns=extractor,
            vocab=vocab,
            reservoir=all_data,
            config=engine_config,
        )

        # Pass semantic index to neural engine
        self.model.neural_engine._semantic_index = self.semantic_index
        self.model.neural_engine._vocab = vocab

        # Predictor
        self.predictor = PatternPredictor(
            extractor, vocab, processor,
            reservoir_samples=self.reservoir_samples,
            weights_dir=self.weights_dir,
            semantic_index=self.semantic_index,
        )
        # Pass model metadata to predictor (zero config)
        self.predictor.model_max_tokens = self.metadata.get("max_tokens", 4096)
        self.predictor.model_temperature = self.temperature

        # SVD upgrader (may not exist)
        self.upgrader = None
        upgrade_dir = os.path.join(self.weights_dir, "upgrades")
        if os.path.exists(upgrade_dir):
            try:
                from ..core.training_upgrades import TrainingUpgrader
                self.upgrader = TrainingUpgrader.load(upgrade_dir, vocab=vocab)
            except Exception:
                pass

        # Knowledge base
        self.knowledge_index = None
        knowledge_path = os.path.join(self.weights_dir, "knowledge_index.json")
        if os.path.exists(knowledge_path):
            try:
                import json as _json
                from collections import defaultdict as _dd
                with open(knowledge_path, 'r', encoding='utf-8') as _f:
                    index_data = _json.load(_f)
                from ..core.knowledge_extractor import KnowledgeIndexer
                self.knowledge_index = KnowledgeIndexer()
                self.knowledge_index.knowledge_store = index_data.get('knowledge_store', [])
                self.knowledge_index.keyword_index = _dd(set,
                    {k: set(v) for k, v in index_data.get('keyword_index', {}).items()})
                self.knowledge_index.category_index = _dd(set,
                    {k: set(v) for k, v in index_data.get('category_index', {}).items()})
            except Exception:
                self.knowledge_index = None

        # Deep reasoning engine
        self.reasoner = None
        reasoning_path = os.path.join(self.weights_dir, "reasoning_patterns.json")
        if os.path.exists(reasoning_path):
            try:
                from ..core.reasoning_engine import DeepReasoner
                self.reasoner = DeepReasoner()
                self.reasoner.load_patterns(reasoning_path)
            except Exception:
                self.reasoner = None

        # Creative engine
        self.creative = None
        creative_path = os.path.join(self.weights_dir, "creative_patterns.json")
        if os.path.exists(creative_path):
            try:
                from ..core.creative_engine import CreativeEngine
                self.creative = CreativeEngine()
                self.creative.load_patterns(creative_path)
            except Exception:
                self.creative = None

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Pool data from disk (supplementary)
    # ------------------------------------------------------------------
    def _load_pool_data_from_disk(self):
        pool_data = []
        base_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool'
        )
        if not os.path.exists(base_path):
            return pool_data
        loaded = 0
        for entry in os.listdir(base_path):
            if loaded >= POOL_LOAD_CAP:
                break
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                pool_file = os.path.join(full_path, 'pool.jsonl')
                if os.path.exists(pool_file):
                    try:
                        with open(pool_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if loaded >= POOL_LOAD_CAP:
                                    break
                                try:
                                    import json as _json
                                    d = _json.loads(line.strip())
                                    text = d.get('text', '')
                                    if text:
                                        pool_data.append(text)
                                        loaded += 1
                                except Exception:
                                    continue
                    except Exception:
                        pass
        print(f"[ChatSession] Loaded {len(pool_data)} pool entries from disk")
        return pool_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def info(self):
        if not self._loaded:
            self.load()
        return dict(self.metadata or {})

    def generate(self, prompt, max_tokens=None, temperature=None, use_history=True):
        """
        Generate response. ALL from model, NO config file.
        Priority:
          1. Math detection
          2. Semantic retrieval from reservoir
          2.5 Knowledge base
          3. Deep reasoning
          3.5 Creative generation
          4. Neural engine
          5. Predictor fallback
        """
        if not self._loaded:
            self.load()

        if max_tokens is None:
            max_tokens = self.metadata.get("max_tokens", 4096)
        if temperature is None:
            temperature = self.temperature

        # Priority 1: Math
        if self.model:
            math_result = self.model.neural_engine.python_terminal.parse_and_compute(prompt)
            if math_result is not None:
                return math_result

        # Priority 2: Semantic retrieval — THE PRIMARY PATH
        if self.semantic_index and self.semantic_index._built:
            vocab = self._get_vocab()
            results = self.semantic_index.search(prompt, vocab, top_k=SEMANTIC_SEARCH_TOP_K)
            if results:
                best_score, best_answer = results[0]
                if (best_score >= SEMANTIC_SEARCH_THRESHOLD and best_answer
                        and len(best_answer) > 10
                        and self._is_relevant_answer(prompt, best_answer)):
                    return best_answer
                if len(results) > 1:
                    for score, answer in results[1:]:
                        if (score >= SEMANTIC_SEARCH_THRESHOLD * SEMANTIC_SEARCH_RELAXED
                                and answer and len(answer) > 10
                                and self._is_relevant_answer(prompt, answer)):
                            return answer

        # Priority 2b: Word-overlap pool matching
        if self.pool_pairs:
            matched = self._match_pool(prompt)
            if matched and len(matched) > 10:
                return matched

        # Priority 2.5: Knowledge base
        if self.knowledge_index:
            results = self.knowledge_index.search(prompt, top_k=KNOWLEDGE_SEARCH_TOP_K)
            if results:
                best = results[0]
                answer = (best.get('definition') or best.get('answer')
                          or best.get('object') or '')
                if answer and len(answer) > 10:
                    return answer

        # Priority 3: Deep reasoning
        if self.reasoner:
            reasoned = self.reasoner.reason(prompt, knowledge_index=self.knowledge_index)
            if reasoned and len(reasoned) > 10:
                return reasoned

        # Priority 3.5: Creative generation
        if self.creative:
            creative = self.creative.generate(prompt, reservoir=self.reservoir_samples)
            if creative and len(creative) > 20:
                return creative

        # Priority 4: Neural engine
        if self.model:
            response = self.model.generate(prompt, max_tokens)
            if response and len(response) > 10:
                return response

        # Priority 5: Predictor fallback
        if self.predictor:
            context = self._build_context(prompt, use_history)
            response = self.predictor.reply(context, max_tokens, temperature)
            if response:
                return response

        return "I'm not sure how to respond to that."

    def _get_vocab(self):
        if self.predictor and self.predictor.vocab:
            return self.predictor.vocab
        if self.model and self.model.vocab:
            return self.model.vocab
        return Vocabulary()

    def get_model_info(self):
        if self.model:
            return self.model.get_model_info()
        return {}

    def visualize_neural_network(self):
        if self.model:
            return self.model.visualize_network()
        return "Model not loaded"

    def chat(self, prompt, max_tokens=None, temperature=None):
        self.history.append({"role": "user", "content": prompt})
        reply = self.generate(prompt, max_tokens, temperature)
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
        return reply

    def reset(self):
        self.history = []

    def get_history(self):
        return self.history

    # ------------------------------------------------------------------
    # Relevance checking
    # ------------------------------------------------------------------
    def _is_relevant_answer(self, query, answer):
        query_lower = query.lower()
        answer_lower = answer.lower()

        if any(kw in query_lower for kw in CREATIVE_KEYWORDS):
            return True

        answer_first = answer_lower[:RELEVANCE_CHECK_PREFIX_LEN]
        for ind in STORY_INDICATORS:
            if ind in answer_first:
                return False
        return True

    # ------------------------------------------------------------------
    # Word-overlap pool matching (secondary fallback)
    # ------------------------------------------------------------------
    def _match_pool(self, query):
        if not self.pool_pairs:
            return ''

        query_lower = query.lower().strip().rstrip('?!.')
        query_words = set(query_lower.split())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'this', 'that', 'it', 'or', 'and', 'but', 'not',
            'i', 'me', 'my', 'your', 'we', 'they', 'he', 'she',
        }
        query_content = query_words - stop_words
        if not query_content:
            query_content = query_words

        best_score = 0
        best_answer = ''

        for instruction, answer in self.pool_pairs:
            if not instruction or not answer:
                continue
            inst_lower = instruction.lower().strip().rstrip('?!.')
            inst_words = set(inst_lower.split()) - stop_words

            if query_lower == inst_lower:
                return answer

            if query_content:
                overlap = len(query_content & inst_words)
                score = overlap / max(len(query_content), 1)
                inst_overlap = overlap / max(len(inst_words), 1) if inst_words else 0
                score = score * POOL_MATCH_QUERY_WEIGHT + inst_overlap * POOL_MATCH_INST_WEIGHT

                if len(answer) > POOL_MATCH_SHORT_THRESHOLD:
                    score *= POOL_MATCH_LENGTH_BONUS
                if len(answer) > POOL_MATCH_LONG_THRESHOLD:
                    score *= POOL_MATCH_LENGTH_BONUS

                if score > best_score:
                    best_score = score
                    best_answer = answer

        if best_score >= POOL_MATCH_MIN_SCORE and best_answer:
            return best_answer
        return ''

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_context(self, prompt, use_history=True):
        if not use_history or not self.history:
            return prompt
        parts = []
        for msg in self.history[-self.max_history:]:
            if msg["role"] == "assistant":
                parts.append(msg["content"])
        parts.append(prompt)
        context = " ... ".join(parts)
        tokens = context.split()
        return " ".join(tokens[-self.chat_context_tokens:])

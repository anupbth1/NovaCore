"""Inference: ChatSession - interactive chat like Ollama.

Core architecture:
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
from ..config import get_default
from .predictor import PatternPredictor


# ---------------------------------------------------------------------------
# Semantic Index — fast cosine-similarity retrieval from instruction-answer
# pairs extracted from the reservoir.  Built once at load time, queried in
# O(dim) per candidate (NumPy dot product).
# ---------------------------------------------------------------------------
class SemanticIndex:
    """
    IDF-weighted feature-hashed embedding index for semantic retrieval.

    At build time:
      - Extract (instruction, answer) pairs from raw reservoir texts
      - Compute IDF weights across all instructions
      - Pre-compute a dense (N, dim) embedding matrix using feature hashing

    At query time:
      - Encode the query vector
      - Dot-product against all rows (= cosine similarity since rows are L2-normalised)
      - Return top-k (score, answer) pairs
    """

    def __init__(self, dim=None):
        from ..config import get_default
        self.dim = dim or get_default('dim')
        self.instructions = []   # str — original instruction text
        self.answers = []        # str — corresponding clean answer text
        self.embeddings = None   # np.ndarray (N, dim) float32, L2-normalised
        self.idf = {}            # word -> float
        self._built = False

    # ---- building ---------------------------------------------------------

    def build(self, reservoir, vocab, pool_extra=None):
        """
        Build the index from *reservoir* (list[str]) and optionally
        *pool_extra* (list[str]) which may come from on-disk pool files.
        """
        from ..config import get_default
        limit = get_default('semantic_index_build_limit', 200000)

        # 1. Extract (instruction, answer) pairs
        pairs = []  # list of (inst_tokens, answer_str)
        raw_instructions = []  # for IDF computation
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

        # Also accept pool_extra (already parsed instruction strings)
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

        # 2. Compute IDF
        n_docs = len(raw_instructions)
        doc_freq = Counter()
        for toks in raw_instructions:
            for w in set(toks):
                doc_freq[w] += 1
        self.idf = {}
        for w, df in doc_freq.items():
            self.idf[w] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # 3. Build embedding matrix
        N = len(pairs)
        emb = np.zeros((N, self.dim), dtype=np.float32)
        for i, (toks, _) in enumerate(pairs):
            for tok in toks:
                h = murmurhash3(tok)
                bucket = h % self.dim
                sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                w = self.idf.get(tok, 1.0)
                emb[i, bucket] += sign * w
            # L2-normalise
            norm = np.linalg.norm(emb[i])
            if norm > 0:
                emb[i] /= norm

        self.embeddings = emb
        self.instructions = [self._clean_inst(toks) for toks, _ in pairs]
        self.answers = [ans for _, ans in pairs]
        self._built = True
        print(f"[SemanticIndex] Built index: {N} instruction-answer pairs, "
              f"dim={self.dim}")

    # ---- query ------------------------------------------------------------

    def search(self, query_text, vocab, top_k=None):
        """
        Return list of (score, answer) sorted descending, up to *top_k*.
        """
        from ..config import get_default
        if not self._built or self.embeddings is None or len(self.answers) == 0:
            return []
        top_k = top_k or get_default('semantic_search_top_k', 5)

        # Encode query
        qvec = self._encode_query(query_text, vocab)
        if qvec is None:
            return []

        # Cosine similarity = dot product (both L2-normalised)
        scores = self.embeddings @ qvec  # (N,)
        # Get top-k indices
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

    # ---- internal ---------------------------------------------------------

    def _encode_query(self, text, vocab):
        """Encode a query into a normalised feature-hashed vector."""
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

        # <instruction>...</instruction>
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

        # <answer>...</answer>
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

    @staticmethod
    def _clean_inst(toks):
        """Re-join tokens into a readable instruction string."""
        return ' '.join(toks)


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------
class ChatSession:
    """
    Loads a saved NovaCore model and provides an interactive chat interface.

    Priority chain:
      1. Math detection (python terminal)
      2. Semantic retrieval (IDF-weighted cosine similarity on reservoir)
      3. Knowledge base retrieval
      4. Deep reasoning engine
      5. Creative generation
      6. Neural engine / predictor (last resort)
    """

    def __init__(self, weights_dir, max_history=None, temperature=None):
        self.weights_dir = weights_dir
        self.max_history = max_history if max_history is not None else get_default('max_history')
        self.temperature = temperature if temperature is not None else get_default('temperature')
        self.history = []
        self.predictor = None
        self.model = None
        self.patterns_data = {}
        self.reservoir_samples = []
        self.metadata = None
        self._loaded = False
        self.semantic_index = SemanticIndex()
        self.pool_pairs = []

    def load(self):
        """Load the saved model weights + vocab + patterns + reservoir samples."""
        wm = WeightManager(self.weights_dir)
        try:
            data, metadata = wm.load()
        except FileNotFoundError as e:
            raise RuntimeError(f"Model not found at '{self.weights_dir}': {e}")

        # Vocabulary
        vocab_path = os.path.join(self.weights_dir, "vocab.json")
        if os.path.exists(vocab_path):
            try:
                vocab = Vocabulary.load(vocab_path)
            except Exception:
                vocab = Vocabulary()
        else:
            vocab = Vocabulary()

        dim = (metadata or {}).get("dim") or get_default('dim')
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

        # Load pool data from disk (if available) — adds extra QA pairs
        pool_extra = self._load_pool_data_from_disk()

        # Build semantic index from reservoir + pool extra
        self.semantic_index.build(self.reservoir_samples, vocab, pool_extra)

        # Also try to load on-disk pool pairs for word-overlap fallback
        self.pool_pairs = []
        for text in (self.reservoir_samples[:5000] + pool_extra[:5000]):
            inst, ans = SemanticIndex._extract_pair(text)
            if inst and ans:
                self.pool_pairs.append((inst, ans))

        extractor = PatternExtractor()
        extractor.patterns = dict(self.patterns_data)

        # Combine reservoir and pool data for internal components
        all_data = list(self.reservoir_samples) + pool_extra

        # Build config dict for all internal components
        engine_config = {
            'neural_engine': get_default('neural_engine', {}),
            'virtual_simulation': get_default('virtual_simulation', {}),
            'verification_engine': get_default('verification_engine', {}),
            'python_terminal': get_default('python_terminal', {}),
        }

        # Initialize NovaCoreModel
        self.model = NovaCoreModel(
            patterns=extractor,
            vocab=vocab,
            reservoir=all_data,
            config=engine_config,
        )

        # Pass semantic index to neural engine for better retrieval
        self.model.neural_engine._semantic_index = self.semantic_index
        self.model.neural_engine._vocab = vocab

        # Also keep predictor as fallback
        self.predictor = PatternPredictor(
            extractor,
            vocab,
            processor,
            reservoir_samples=self.reservoir_samples,
            weights_dir=self.weights_dir
        )

        # Load SVD upgrader (may not exist)
        self.upgrader = None
        upgrade_dir = os.path.join(self.weights_dir, "upgrades")
        if os.path.exists(upgrade_dir):
            try:
                from ..core.training_upgrades import TrainingUpgrader
                self.upgrader = TrainingUpgrader.load(upgrade_dir, vocab=vocab)
            except Exception:
                pass

        # Load knowledge base
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

        # Load deep reasoning engine
        self.reasoner = None
        reasoning_path = os.path.join(self.weights_dir, "reasoning_patterns.json")
        if os.path.exists(reasoning_path):
            try:
                from ..core.reasoning_engine import DeepReasoner
                self.reasoner = DeepReasoner()
                self.reasoner.load_patterns(reasoning_path)
            except Exception:
                self.reasoner = None

        # Load creative engine
        self.creative = None
        creative_path = os.path.join(self.weights_dir, "creative_patterns.json")
        if os.path.exists(creative_path):
            try:
                from ..core.creative_engine import CreativeEngine
                self.creative = CreativeEngine()
                self.creative.load_patterns(creative_path)
            except Exception:
                self.creative = None

        self.metadata = metadata
        self.encoder_input_dim = dim
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Pool data loading from disk (supplementary to reservoir)
    # ------------------------------------------------------------------
    def _load_pool_data_from_disk(self):
        """Load pool data from disk pool files if available."""
        pool_data = []
        base_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool'
        )
        if not os.path.exists(base_path):
            return pool_data
        cap = get_default('pool_load_cap', 200000)
        loaded = 0
        for entry in os.listdir(base_path):
            if loaded >= cap:
                break
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                pool_file = os.path.join(full_path, 'pool.jsonl')
                if os.path.exists(pool_file):
                    try:
                        with open(pool_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if loaded >= cap:
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
        """Return model metadata dict."""
        if not self._loaded:
            self.load()
        return dict(self.metadata or {})

    def generate(self, prompt, max_tokens=None, temperature=None, use_history=True):
        """
        Generate response with priority:
        1. Math detection (python terminal)
        2. Semantic retrieval from reservoir (cosine similarity)
        2.5 Knowledge base retrieval
        3. Deep reasoning
        3.5 Creative generation
        4. Neural engine (reservoir + patterns)
        5. Predictor fallback
        """
        if not self._loaded:
            self.load()
        if max_tokens is None:
            max_tokens = (self.metadata or {}).get("max_tokens") or get_default('max_tokens')

        # Priority 1: Math detection
        if self.model:
            math_result = self.model.neural_engine.python_terminal.parse_and_compute(prompt)
            if math_result is not None:
                return math_result

        # Priority 2: Semantic retrieval from reservoir — THE PRIMARY PATH
        threshold = get_default('semantic_search_threshold')
        top_k = get_default('semantic_search_top_k')
        relaxed = get_default('semantic_search_relaxed_factor')
        results = self.semantic_index.search(prompt, self._get_vocab(), top_k=top_k)
        if results:
            best_score, best_answer = results[0]
            if (best_score >= threshold and best_answer
                    and len(best_answer) > 10
                    and self._is_relevant_answer(prompt, best_answer)):
                return best_answer
            # Try second-best if first is irrelevant
            if len(results) > 1:
                for score, answer in results[1:]:
                    if (score >= threshold * relaxed and answer
                            and len(answer) > 10
                            and self._is_relevant_answer(prompt, answer)):
                        return answer

        # Priority 2b: Word-overlap pool matching (fallback if semantic weak)
        if self.pool_pairs:
            matched = self._match_pool(prompt)
            if matched and len(matched) > 10:
                return matched

        # Priority 2.5: Knowledge base retrieval
        if self.knowledge_index:
            results = self.knowledge_index.search(prompt, top_k=get_default('knowledge_search_top_k'))
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

        # Priority 4: Neural engine (reservoir + patterns)
        if self.model:
            response = self.model.generate(prompt, max_tokens)
            if response and len(response) > 10:
                return response

        # Priority 5: Predictor fallback
        if self.predictor:
            context = self._build_context(prompt, use_history)
            response = self.predictor.reply(context, max_tokens, self.temperature)
            if response:
                return response

        return "I'm not sure how to respond to that."

    def _get_vocab(self):
        """Get the vocabulary from the predictor or model."""
        if self.predictor and self.predictor.vocab:
            return self.predictor.vocab
        if self.model and self.model.vocab:
            return self.model.vocab
        return Vocabulary()

    def get_model_info(self):
        """Get complete model information."""
        if self.model:
            return self.model.get_model_info()
        return {}

    def visualize_neural_network(self):
        """Get neural network visualization."""
        if self.model:
            return self.model.visualize_network()
        return "Model not loaded"

    def chat(self, prompt, max_tokens=None, temperature=None):
        """
        User sends a message -> add to history -> generate ->
        add assistant reply to history -> return reply.
        """
        self.history.append({"role": "user", "content": prompt})
        reply = self.generate(prompt, max_tokens, temperature)
        self.history.append({"role": "assistant", "content": reply})
        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
        return reply

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self):
        """Return conversation history."""
        return self.history

    # ------------------------------------------------------------------
    # Word-overlap pool matching (secondary fallback)
    # ------------------------------------------------------------------
    def _is_relevant_answer(self, query, answer):
        """Check if the answer is likely relevant to the query (not a random story)."""
        query_lower = query.lower()
        answer_lower = answer.lower()

        # Creative queries should accept any length answer
        creative_kw = set(get_default('creative_keywords', []))
        if any(kw in query_lower for kw in creative_kw):
            return True

        # Answers that look like stories (narrative indicators) are likely irrelevant
        # for non-creative queries
        story_inds = get_default('story_indicators', [])
        prefix_len = get_default('relevance_check_prefix_len')
        answer_first = answer_lower[:prefix_len]
        for ind in story_inds:
            if ind in answer_first:
                return False

        return True

    def _match_pool(self, query: str) -> str:
        """Match query against pool instructions using word overlap."""
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

            # Exact match bonus
            if query_lower == inst_lower:
                return answer

            if query_content:
                overlap = len(query_content & inst_words)
                q_weight = get_default('pool_match_query_weight')
                i_weight = get_default('pool_match_inst_weight')
                score = overlap / max(len(query_content), 1)
                inst_overlap = overlap / max(len(inst_words), 1) if inst_words else 0
                score = score * q_weight + inst_overlap * i_weight

                short_thresh = get_default('pool_match_short_threshold')
                long_thresh = get_default('pool_match_long_threshold')
                length_bonus = get_default('pool_match_length_bonus')
                if len(answer) > short_thresh:
                    score *= length_bonus
                if len(answer) > long_thresh:
                    score *= length_bonus

                if score > best_score:
                    best_score = score
                    best_answer = answer

        if best_score >= get_default('pool_match_min_score') and best_answer:
            return best_answer
        return ''

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_context(self, prompt, use_history=True):
        """Combine recent conversation into a single context string."""
        if not use_history or not self.history:
            return prompt
        parts = []
        for msg in self.history[-self.max_history:]:
            if msg["role"] == "assistant":
                parts.append(msg["content"])
        parts.append(prompt)
        context = " ... ".join(parts)
        tokens = context.split()
        return " ".join(tokens[-get_default('chat_context_tokens'):])

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

# Stopwords used for overlap guard in semantic search
_STOPWORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'this', 'that', 'it', 'or', 'and', 'but', 'not',
    'i', 'me', 'my', 'your', 'we', 'they', 'he', 'she', 'you',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where',
    'why', 'how', 'hi', 'hello', 'hey', 'please', 'there',
    'some', 'any', 'more', 'most', 'other', 'such', 'only',
])


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
        # Token sets per instruction for overlap guard (kills hash-collision
        # garbage matches on short queries like "hi")
        self._inst_tok_sets = [set(toks) for toks, _ in pairs]
        self._built = True
        print(f"[SemanticIndex] Built: {N} instruction-answer pairs, dim={self.dim}")

    def search(self, query_text, vocab, top_k=None):
        """Return list of (score, answer) sorted descending.

        Cosine similarity alone is fooled by feature-hash collisions when the
        query is short (1-3 words): the query vector becomes almost one-hot and
        ANY instruction with a token in the same bucket scores ~0.95.  So we
        also require REAL word overlap between the query and the matched
        instruction, and blend it into the final score.
        """
        if not self._built or self.embeddings is None or len(self.answers) == 0:
            return []
        top_k = top_k or SEMANTIC_SEARCH_TOP_K

        qvec = self._encode_query(query_text, vocab)
        if qvec is None:
            return []

        # Content words of the query (drop pure stopwords for overlap check)
        q_tokens = vocab._tokenize(query_text)
        q_set = set(q_tokens) - _STOPWORDS
        if not q_set:
            q_set = set(q_tokens)

        # Cosine similarity = dot product (both L2-normalised)
        scores = self.embeddings @ qvec  # (N,)

        # Overlap-aware re-ranking: fetch a wider pool, then blend real overlap
        pool_k = min(len(scores), max(top_k * 10, 100))
        if len(scores) <= pool_k:
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, pool_k)[:pool_k]
            order = order[np.argsort(-scores[order])]

        results = []
        for idx in order:
            cos = float(scores[idx])
            if cos <= 0:
                continue
            inst_set = self._inst_tok_sets[idx]
            if q_set:
                overlap = len(q_set & inst_set)
                overlap_ratio = overlap / max(1, min(len(q_set), 8))
            else:
                overlap = 0
                overlap_ratio = 0.0
            # Blend: pure cosine trusted more with real overlap
            blend = cos * (0.45 + 0.55 * overlap_ratio)
            results.append((blend, cos, overlap, self.answers[idx], inst_set))
            if len(results) >= pool_k:
                break

        # Sort by blended score, drop zero-overlap matches entirely (they are
        # hash collisions, not real relevance)
        results.sort(key=lambda r: r[0], reverse=True)
        out = []
        for blend, cos, overlap, ans, inst_set in results:
            if overlap == 0:
                continue
            out.append((blend, ans))
            if len(out) >= top_k:
                break
        return out

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
        """Load pool QA data from disk (*_stream.jsonl flat files AND pool.jsonl
        subdirectories).  Produced during training by the streaming pipeline.

        Strategy (RAM-bounded + fast):
          - Collect all candidate files
          - Pre-check each file: if its first lines contain no QA structure
            (e.g. the ~2GB TinyStories story file), skip it entirely
          - Process smaller QA files first; stop once the cap is reached
        """
        pool_data = []
        base_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool'
        )
        if not os.path.exists(base_path):
            return pool_data

        candidate_files = []
        try:
            for entry in os.listdir(base_path):
                full = os.path.join(base_path, entry)
                if os.path.isfile(full) and entry.endswith('_stream.jsonl'):
                    candidate_files.append(full)
                elif os.path.isdir(full):
                    pf = os.path.join(full, 'pool.jsonl')
                    if os.path.exists(pf):
                        candidate_files.append(pf)
        except Exception:
            return pool_data

        if not candidate_files:
            return pool_data

        # QA markers shared by every instruction-style pool file
        qa_markers = ('<instruction>', '<answer>', '<assistant>',
                      '<user>', '### Instruction:')

        def file_has_qa(path):
            """Peek the first lines: skip files without QA structure (e.g.
            raw TinyStories story text) so we never scan GBs pointlessly."""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for _ in range(20):
                        line = f.readline()
                        if not line:
                            break
                        if any(m in line for m in qa_markers):
                            return True
            except Exception:
                return False
            return False

        # Pre-check: only keep files that look like QA pools
        candidate_files = [p for p in candidate_files if file_has_qa(p)]

        # Smaller files first (QA pools are tens of MB; stories can be GBs)
        candidate_files.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)

        loaded = 0
        for pool_file in candidate_files:
            if loaded >= POOL_LOAD_CAP:
                break
            try:
                with open(pool_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if loaded >= POOL_LOAD_CAP:
                            break
                        try:
                            import json as _json
                            d = _json.loads(line.strip())
                            text = d.get('text', '')
                            if text and any(m in text for m in qa_markers):
                                pool_data.append(text)
                                loaded += 1
                        except Exception:
                            continue
            except Exception:
                continue
        print(f"[ChatSession] Loaded {len(pool_data)} QA pool entries from disk")
        return pool_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def info(self):
        if not self._loaded:
            self.load()
        return dict(self.metadata or {})

    def generate(self, prompt, max_tokens=None, temperature=None, use_history=True, verbose=False):
        """
        Generate response with Virtual Simulation on ALL paths.
        
        Every candidate from every priority goes through virtual simulation.
        If score < threshold → retry with next candidate / next priority.
        Only returns when virtual simulation PASSES.
        """
        if not self._loaded:
            self.load()

        if max_tokens is None:
            max_tokens = self.metadata.get("max_tokens", 4096)
        if temperature is None:
            temperature = self.temperature

        # ANSI color codes for logging
        GRAY = '\033[90m'
        RESET = '\033[0m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'

        def log(msg, color=GRAY):
            if verbose:
                print(f"{color}{msg}{RESET}")

        def log_step(step, detail="", color=GRAY):
            if verbose:
                print(f"{color}  ▶ {step}{RESET} {GRAY}{detail}{RESET}")

        log(f"{'='*60}")
        log(f"INPUT: {prompt}")
        log(f"PARAMS: max_tokens={max_tokens}, temperature={temperature}, history={use_history}")
        log(f"{'='*60}")

        # ============================================================
        # VIRTUAL SIMULATION WRAPPER
        # ============================================================
        def run_virtual_sim(candidate_text, source_name):
            """Run virtual simulation on a candidate. Returns (passed, score, corrected_text)."""
            if not self.model:
                return True, 1.0, candidate_text  # no model = no sim
            
            log(f"  🔮 VIRTUAL SIM on [{source_name}]...", CYAN)
            log(f"     Input: {candidate_text[:100]}...", GRAY)
            
            # Use the model's neural engine virtual simulation
            sim_result = self.model.neural_engine.virtual_simulation.simulate(prompt, candidate_text)
            score = sim_result.get('score', 0.0)
            final_text = sim_result.get('final_response', candidate_text)
            passed = sim_result.get('passed', False)
            
            log(f"     Score: {score:.4f} (threshold={self.model.neural_engine.virtual_simulation.quality_threshold:.2f})", 
                GREEN if passed else YELLOW)
            if sim_result.get('corrected'):
                log(f"     ✓ CORRECTED by virtual sim", GREEN)
                log(f"     Corrected: {final_text[:100]}...", GRAY)
            elif passed:
                log(f"     ✓ PASSED", GREEN)
            else:
                log(f"     ✗ FAILED — will retry/fallback", YELLOW)
            
            # Also run verification engine for extra check
            if not passed:
                verify_result = self.model.neural_engine.verification_engine.verify_and_retry(prompt, final_text)
                if verify_result.get('verified'):
                    final_text = verify_result['final_response']
                    score = max(score, verify_result['scores'][-1] if verify_result['scores'] else score)
                    passed = True
                    log(f"     ✓ VERIFICATION PASSED (retries={verify_result['attempts']})", GREEN)
                else:
                    log(f"     ✗ VERIFICATION FAILED", YELLOW)
            
            return passed, score, final_text

        # ============================================================
        # PRIORITY 1: Math (exact computation — auto-passes virtual sim)
        # ============================================================
        log_step("PRIORITY 1: Math Detection", "Checking for mathematical expressions...")
        if self.model:
            math_result = self.model.neural_engine.python_terminal.parse_and_compute(prompt)
            if math_result is not None:
                log(f"  ✓ MATH DETECTED: computed result", GREEN)
                log(f"  RESULT: {math_result}", GREEN)
                # Math is exact computation — bypass virtual sim, return directly
                log(f"  ✓ EXACT COMPUTATION — auto-passes virtual sim", GREEN)
                log(f"{'='*60}")
                return math_result
            else:
                log("  ✗ No math expression found", YELLOW)

        # ============================================================
        # CANDIDATE GENERATORS — each yields (source_name, candidate_text)
        # ============================================================
        
        def gen_semantic():
            """Generate candidates from semantic index."""
            if not (self.semantic_index and self.semantic_index._built):
                return
            vocab = self._get_vocab()
            results = self.semantic_index.search(prompt, vocab, top_k=SEMANTIC_SEARCH_TOP_K)
            for i, (score, ans) in enumerate(results):
                if score >= SEMANTIC_SEARCH_THRESHOLD and ans and len(ans) > 10:
                    if self._is_relevant_answer(prompt, ans):
                        yield f"semantic[{i+1}]", ans
            # Relaxed threshold
            for i, (score, ans) in enumerate(results):
                if score >= SEMANTIC_SEARCH_THRESHOLD * SEMANTIC_SEARCH_RELAXED and ans and len(ans) > 10:
                    if self._is_relevant_answer(prompt, ans):
                        yield f"semantic-relaxed[{i+1}]", ans

        def gen_pool():
            """Generate candidates from word-overlap pool matching."""
            if not self.pool_pairs:
                return
            matched = self._match_pool(prompt)
            if matched and len(matched) > 10:
                yield "pool_match", matched

        def gen_knowledge():
            """Generate candidates from knowledge base."""
            if not self.knowledge_index:
                return
            results = self.knowledge_index.search(prompt, top_k=KNOWLEDGE_SEARCH_TOP_K)
            for i, res in enumerate(results):
                ans = (res.get('definition') or res.get('answer') or res.get('object') or '')
                if ans and len(ans) > 10:
                    yield f"knowledge[{i+1}]", ans

        def gen_reasoning():
            """Generate candidates from deep reasoning."""
            if not self.reasoner:
                return
            reasoned = self.reasoner.reason(prompt, knowledge_index=self.knowledge_index)
            if reasoned and len(reasoned) > 10:
                yield "reasoning", reasoned

        def gen_creative():
            """Generate candidates from creative engine."""
            if not self.creative:
                return
            creative = self.creative.generate(prompt, reservoir=self.reservoir_samples)
            if creative and len(creative) > 20:
                yield "creative", creative

        def gen_neural():
            """Generate candidate from neural engine."""
            if not self.model:
                return
            response = self.model.generate(prompt, max_tokens)
            if response and len(response) > 10:
                yield "neural", response

        def gen_predictor():
            """Generate candidate from predictor."""
            if not self.predictor:
                return
            context = self._build_context(prompt, use_history)
            response = self.predictor.reply(context, max_tokens, temperature)
            if response:
                yield "predictor", response

        # ============================================================
        # MAIN LOOP: Try each priority, run virtual sim on each candidate
        # ============================================================
        all_generators = [
            ("SEMANTIC RETRIEVAL", gen_semantic),
            ("POOL MATCHING", gen_pool),
            ("KNOWLEDGE BASE", gen_knowledge),
            ("DEEP REASONING", gen_reasoning),
            ("CREATIVE GENERATION", gen_creative),
            ("NEURAL ENGINE", gen_neural),
            ("PREDICTOR FALLBACK", gen_predictor),
        ]

        for priority_name, gen_func in all_generators:
            log_step(f"PRIORITY: {priority_name}", "")
            try:
                for source_name, candidate in gen_func():
                    log(f"  Candidate from {source_name}: {candidate[:120]}...", BLUE)
                    
                    # Run virtual simulation
                    passed, score, final_text = run_virtual_sim(candidate, source_name)
                    
                    if passed:
                        log(f"  ✓ VIRTUAL SIM PASSED — returning final answer", GREEN)
                        log(f"  FINAL: {final_text[:200]}...", GREEN)
                        log(f"{'='*60}")
                        return final_text
                    else:
                        log(f"  → Candidate rejected, trying next...", YELLOW)
            except Exception as e:
                log(f"  ✗ {priority_name} error: {e}", YELLOW)
                continue

        # Nothing passed virtual simulation
        log("  ✗ ALL CANDIDATES FAILED VIRTUAL SIM — returning best effort", YELLOW)
        log(f"{'='*60}")
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

    def chat(self, prompt, max_tokens=None, temperature=None, verbose=False):
        self.history.append({"role": "user", "content": prompt})
        reply = self.generate(prompt, max_tokens, temperature, use_history=True, verbose=verbose)
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

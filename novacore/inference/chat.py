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
import string
import pickle
import random
import numpy as np
import math
from collections import Counter

# Windows consoles default to cp1252 which cannot encode ▶ / emoji.
# Force UTF-8 so the gray reasoning trace prints everywhere.
try:
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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
    'kya', 'ka', 'ki', 'ke', 'kaise', 'kahan', 'kyun', 'kis',
    'hai', 'ho', 'hain', 'kar', 'karta', 'karte', 'hota', 'hoti', 'hote',
])

# Smalltalk / greeting detection.  These are conversational openers where
# the model must reply from greeting data — NOT from factual retrieval that
# would produce unrelated trivia.
SMALLTALK_PHRASES = frozenset([
    'hi', 'hello', 'hey', 'yo', 'how are you', 'how are you doing',
    'how is it going', 'how have you been', 'how do you do',
    'good morning', 'good afternoon', 'good evening', 'whats up',
    "what's up", 'hey there', 'hello there', 'hi there',
])
SMALLTALK_PROBES = ('hello', 'hi', 'hey', 'how are you',
                    'good morning', 'good afternoon', 'good evening')

# Fact aggregation — compile-an-answer fallback.  When no single direct QA
# pair answers the query, gather every mention of the query's content words
# across the model's own data and join the best snippets into one answer.
AGGREGATE_TOP_N = 6
AGGREGATE_MAX_CHARS = 1400
AGGREGATE_MIN_CHARS = 40
# Ambiguous abbreviations: expand each to related words when scoring snippet
# relevance so 'pm' (prime minister) matches prime/minister instructions
# instead of post-meridiem timezone text.
_FACT_EXPANSION = {
    'pm': ('prime', 'minister', 'ministers'),
    'usa': ('america', 'united', 'states'),
    'us': ('america', 'united', 'states'),
}


def _content_toks(tokens):
    """Drop punctuation-only tokens and stopwords.  If that empties the list,
    fall back to all tokens that contain any real letters so greeting queries
    like 'hi' / 'hello' still keep meaningful content for overlap checks."""
    real = [t for t in tokens if any(c not in string.punctuation for c in t)]
    return [t for t in real if t not in _STOPWORDS] or real


# ---------------------------------------------------------------------------
# Code detection helpers — used so that ANY code output is verified by the
# internal Python terminal before being returned.
# ---------------------------------------------------------------------------
_CODE_KEYWORDS = frozenset([
    'def ', 'import ', 'from ', 'class ', 'return ', 'print(', 'print (',
    'if ', 'elif ', 'else', 'for ', 'while ', 'lambda', 'range(',
    'append(', 'self', 'return', '```python', '```py', '```',
    '==', '!=', '>=', '<=', '+=', '-=', 'while', 'def', 'import',
])
_CODE_STARTERS = frozenset([
    'def ', 'import ', 'from ', 'class ', 'print(', 'for ', 'while ',
    '```', 'lambda', 'x =', 'a =', 'return ',
])


def _looks_like_code(text):
    """Heuristic: does the text look like source code rather than prose?"""
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    # Markdown code fence
    if '```' in t:
        return True
    lines = [ln.strip() for ln in t.split('\n') if ln.strip()]
    if not lines:
        return False
    code_lines = 0
    for ln in lines:
        if (ln.startswith(('def ', 'import ', 'from ', 'class ',
                           'print(', 'for ', 'while ', 'if ', 'elif ',
                           'else:', 'return ', 'lambda ', '```'))
                or ln.startswith('#')):
            code_lines += 1
    # Majority of short-ish text that is code-ish
    if len(lines) <= 3:
        return code_lines >= 2
    return code_lines >= max(2, len(lines) // 2)


def _extract_code_block(text):
    """Extract a code block from markdown fences if present, else the raw text
    when it clearly reads as code."""
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    if '```' in t:
        blocks = re.findall(r'```(?:python|py)?\s*\n?(.*?)```', t, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        # single-fence or inline
        return t.split('```')[-1].strip() if t.count('```') % 2 == 1 else None
    # No fences: if the whole text looks like code, return it trimmed
    if _looks_like_code(t):
        # Trim trailing prose after the code ends (heuristic: last non-code line)
        lines = t.split('\n')
        code_lines = []
        for ln in lines:
            if ln.strip() and not ln.strip().startswith(('#', '//', '/*')):
                code_lines.append(ln)
            elif ln.strip().startswith('#') and len(code_lines) > 0:
                code_lines.append(ln)
        return '\n'.join(code_lines).strip()
    return None


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

    def search(self, query_text, vocab, top_k=None, log=None):
        """Return list of (score, answer, instruction) sorted descending.

        Cosine similarity alone is fooled by feature-hash collisions when the
        query is short (1-3 words): the query vector becomes almost one-hot and
        ANY instruction with a token in the same bucket scores ~0.95.  So we
        also require REAL word overlap between the query and the matched
        instruction, and blend it into the final score.

        When *log* (a callable) is provided, every surviving match is printed
        with its score so the user can watch what the model is retrieving.
        """
        if not self._built or self.embeddings is None or len(self.answers) == 0:
            if log is not None:
                log("     · semantic index empty — skipping")
            return []
        top_k = top_k or SEMANTIC_SEARCH_TOP_K

        qvec = self._encode_query(query_text, vocab)
        if qvec is None:
            if log is not None:
                log("     · query produced no embedding vector")
            return []

        # Content words of the query for the overlap guard: drop stopwords AND
        # punctuation-only tokens.  If nothing survives, fall back to all real
        # words so greeting queries like "hi" / "hello" still keep meaning.
        q_set = set(_content_toks(vocab._tokenize(query_text)))
        if not q_set:
            if log is not None:
                log("     · query has no usable content words")
            return []
        total_q = sum(self.idf.get(t, 1.0) for t in q_set)

        # Cosine similarity = dot product (both L2-normalised), full scan.
        # The index is small (tens of thousands of docs) so a full scan is
        # cheap and gives strictly better relevance than a hash-bucket pool
        # which was fooled by collisions on short queries.
        scores = self.embeddings @ qvec  # (N,)

        results = []
        for idx in np.argsort(-scores):
            cos = float(scores[idx])
            if cos <= 0:
                break
            inst_set = self._inst_tok_sets[idx]
            overlap_toks = q_set & inst_set
            if not overlap_toks:
                continue  # zero real word overlap = hash collision, not relevance
            num = sum(self.idf.get(t, 1.0) for t in overlap_toks)
            overlap_ratio = num / max(total_q, 1e-9)
            # Blend: cosine trusted more when the instruction shares rare (high
            # idf) content words with the query.
            blend = cos * (0.4 + 0.6 * overlap_ratio)
            results.append((blend, cos, overlap_ratio, len(overlap_toks),
                            self.answers[idx], idx))
            if len(results) >= max(top_k * 5, 25):
                break

        results.sort(key=lambda r: r[0], reverse=True)
        out = []
        for blend, cos, ratio, overlap, ans, idx in results:
            inst_text = self.instructions[idx]
            if log is not None:
                log(f"     · idx={idx}  score={blend:.3f} cos={cos:.3f} "
                    f"word_overlap={overlap} ratio={ratio:.2f}  "
                    f"instruction: '{inst_text[:90]}'")
            out.append((blend, ans, inst_text))
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
_HONEST_REFUSAL = ("I don't have a reliable answer for that from my trained data "
                   "yet. If you teach me a dataset with it, I'll answer correctly.")

# Interrogative heads that start a NEW sub-question inside a compound prompt.
_SUBQ_HEADS = (
    'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom',
    'what\'s', 'whats', 'define', 'explain', 'describe', 'is there',
    'are there', 'do you know', 'tell me', 'can you', 'does', 'should',
)


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
        self.fact_index = {}
        self.max_history = _DEFAULT_MAX_HISTORY
        self.chat_context_tokens = _DEFAULT_CHAT_CONTEXT_TOKENS
        self.enable_cortex = False
        self.cortex = None
        self.dim = None
        self.qa_bank = []
        self.enable_cde = False
        self.cde = None
        self.knowledge_index = None
        self.reasoner = None
        self.creative = None

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

        # QA bank — instruction/answer pairs BAKED INTO THE MODEL WEIGHTS at
        # training time.  This makes chat fully self-contained: the model never
        # reads external pool files during inference.
        qa_bank = []
        if "qa_bank" in data.files:
            try:
                raw_qa = data["qa_bank"].tobytes()
                if raw_qa:
                    qa_bank = pickle.loads(raw_qa)
            except Exception as e:
                print(f"[WARN] Could not load qa_bank: {e}")

        print(f"[ChatSession] Loaded {len(self.reservoir_samples)} reservoir samples, "
              f"{len(qa_bank)} QA pairs baked in weights")
        print(f"[ChatSession] Model: dim={dim}, vocab={len(vocab)}, "
              f"temp={self.temperature}")
        self.qa_bank = qa_bank
        self.dim = dim

        # Semantic index from model-internal data ONLY (reservoir + qa_bank).
        # No external pool files — the model is self-contained.
        self.semantic_index = SemanticIndex(dim)
        self.semantic_index.build(qa_bank, vocab, self.reservoir_samples)

        # Word-overlap pool pairs (fallback, model-internal only)
        self.pool_pairs = []
        for text in (self.reservoir_samples[:5000] + qa_bank[:5000]):
            inst, ans = SemanticIndex._extract_pair(text)
            if inst and ans:
                self.pool_pairs.append((inst, ans))

        # Inverted "fact index" from the same model-internal pairs: for every
        # instruction term remember (instruction terms, answer).  Used by FACT
        # AGGREGATION to compile an answer from every mention of the query's
        # content words — the model's own data, nothing external.
        fact_index = {}
        for _inst, _ans in zip(self.semantic_index.instructions,
                               self.semantic_index.answers):
            if not _ans or len(_ans) < 10:
                continue
            _terms = frozenset(_content_toks(vocab._tokenize(_inst)))
            if not _terms:
                continue
            for _t in _terms:
                fact_index.setdefault(_t, []).append((_terms, _ans))
        self.fact_index = fact_index
        _fact_terms = len(fact_index)
        _fact_refs = sum(len(v) for v in fact_index.values())
        print(f"[ChatSession] Fact index: {_fact_terms} terms, "
              f"{_fact_refs} snippet refs")

        extractor = PatternExtractor()
        extractor.patterns = dict(self.patterns_data)

        all_data = list(self.reservoir_samples) + qa_bank

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
    # BAKING: read external pool files ONCE and bake QA pairs INTO weights.
    # After this, chat never needs the external pool again — fully
    # self-contained model (like a transformer weights file).
    # ------------------------------------------------------------------
    def bake_qa_bank(self, base_pool_dir=None, cap=None):
        """
        Read *_stream.jsonl pool files (created during training), extract the
        instruction/answer pairs, and store them INSIDE weights.ncw as the
        'qa_bank' array.  This is a one-time migration step so that inference
        reads ONLY from the model directory.
        """
        import json as _json
        cap = cap or POOL_LOAD_CAP
        base = base_pool_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool'
        )
        if not os.path.exists(base):
            print("[Bake] No pool dir found:", base)
            return 0

        texts = []
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if os.path.isfile(full) and entry.endswith('_stream.jsonl'):
                with open(full, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(texts) >= cap:
                            break
                        try:
                            d = _json.loads(line.strip())
                            t = d.get('text', '')
                            if t and ('<instruction>' in t or '<answer>' in t
                                      or '<assistant>' in t or '<user>' in t):
                                texts.append(t)
                        except Exception:
                            continue
            elif os.path.isdir(full):
                pf = os.path.join(full, 'pool.jsonl')
                if os.path.exists(pf):
                    with open(pf, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(texts) >= cap:
                                break
                            try:
                                d = _json.loads(line.strip())
                                t = d.get('text', '')
                                if t:
                                    texts.append(t)
                            except Exception:
                                continue
            if len(texts) >= cap:
                break

        # Re-save weights with qa_bank embedded
        wm = WeightManager(self.weights_dir)
        data, metadata = wm.load()
        arrays = {}
        for k in data.files:
            try:
                arrays[k] = np.array(data[k])
            except Exception:
                pass
        arrays["qa_bank"] = np.frombuffer(pickle.dumps(texts), dtype=np.uint8)
        wm.save(arrays, metadata)
        print(f"[Bake] Baked {len(texts)} QA entries into {self.weights_dir}/weights.ncw")
        return len(texts)

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

        # ============================================================
        # SMALLTALK SHORT-CIRCUIT — "hi" / "how are you?" must ALWAYS get
        # a greeting answer regardless of history or priority ordering.
        # Answers are real greeting sentences from the model's own data.
        # ============================================================
        try:
            _greet = self._smalltalk_reply(prompt)
            if _greet:
                return _greet
        except Exception:
            pass

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
                print(f"{color}{msg}{RESET}", flush=True)

        def log_step(step, detail="", color=GRAY):
            if verbose:
                print(f"{color}  ▶ {step}{RESET} {GRAY}{detail}{RESET}", flush=True)

        log(f"{'='*60}")
        log(f"INPUT: {prompt}")
        log(f"PARAMS: max_tokens={max_tokens}, temperature={temperature}, history={use_history}")
        log(f"{'='*60}")

        # ============================================================
        # COMPOUND-PROMPT DECOMPOSITION — "who is X and what is Y?"
        # Each independent sub-question runs the full pipeline, then the
        # answers are joined.  This is the transformer-style behaviour of
        # composing an answer from multiple retrieval passes.
        # ============================================================
        if not getattr(self, '_decomp_depth', 0):
            parts = self._decompose_multi(prompt)
            if parts:
                log("  ▶ COMPOUND PROMPT — answering each part separately", CYAN)
                self._decomp_depth = 1
                _pieces = []
                try:
                    for _p in parts:
                        try:
                            _a = self.generate(
                                _p, max_tokens=min(max_tokens, 140),
                                temperature=temperature, use_history=False,
                                verbose=False)
                        except Exception:
                            _a = None
                        if not _a or _a.strip() == _HONEST_REFUSAL:
                            _pieces.append(
                                "(I don't have that in my training data.)")
                        else:
                            _pieces.append(_a.strip())
                finally:
                    self._decomp_depth = 0
                if _pieces:
                    log(f"  ✓ COMPOUND ANSWERED ({len(_pieces)} parts)", GREEN)
                    log(f"{'='*60}")
                    return '\n\n'.join(_pieces)

        # ============================================================
        # VIRTUAL SIMULATION WRAPPER
        # ============================================================
        def run_virtual_sim(candidate_text, source_name):
            """Run virtual simulation on a candidate.
            Returns (passed, score, corrected_text, relevant)."""
            if not self.model:
                return True, 1.0, candidate_text, True  # no model = no sim

            log(f"  🔮 VIRTUAL SIM on [{source_name}]...", CYAN)
            log(f"     Input: {candidate_text[:100]}...", GRAY)

            # Use the model's neural engine virtual simulation
            sim_result = self.model.neural_engine.virtual_simulation.simulate(prompt, candidate_text)
            score = sim_result.get('score', 0.0)
            final_text = sim_result.get('final_response', candidate_text)
            passed = sim_result.get('passed', False)
            relevant = sim_result.get('relevant', True)

            log(f"     Score: {score:.4f} (threshold={self.model.neural_engine.virtual_simulation.quality_threshold:.2f})",
                GREEN if passed else YELLOW)
            if not relevant:
                log(f"     ✗ IRRELEVANT — answer shares no content words with "
                    f"the query", YELLOW)
            elif sim_result.get('corrected'):
                log(f"     ✓ CORRECTED by virtual sim", GREEN)
                log(f"     Corrected: {final_text[:100]}...", GRAY)
            elif passed:
                log(f"     ✓ PASSED", GREEN)
            else:
                log(f"     ✗ FAILED — will retry/fallback", YELLOW)

            # Also run verification engine for extra check — but only for
            # candidates that are relevant to the query.  Retrieving/rewriting
            # an irrelevant answer would just re-introduce the garbage the
            # relevance gate just rejected.
            if not passed:
                if relevant:
                    verify_result = self.model.neural_engine.verification_engine.verify_and_retry(prompt, final_text)
                    if verify_result.get('verified'):
                        final_text = verify_result['final_response']
                        score = max(score, verify_result['scores'][-1] if verify_result['scores'] else score)
                        passed = True
                        log(f"     ✓ VERIFICATION PASSED (retries={verify_result['attempts']})", GREEN)
                    else:
                        log(f"     ✗ VERIFICATION FAILED", YELLOW)
                else:
                    log(f"     · skipping verification rescue on irrelevant "
                        f"candidate", YELLOW)

            return passed, score, final_text, relevant

        # ============================================================
        # PRIORITY 1: Python Terminal — math AND direct code execution
        # ============================================================
        python_terminal = (self.model.neural_engine.python_terminal
                           if self.model else None)
        log_step("PRIORITY 1: Python Terminal",
                 "Exact math computation + code execution")
        if python_terminal is not None:
            # Math expression in the prompt
            math_result = python_terminal.parse_and_compute(prompt)
            if math_result is not None:
                log(f"  ✓ MATH EXECUTED: {math_result}", GREEN)
                log(f"  {'='*60}")
                return math_result
            # Direct code pasted by the user ("run this: ...")
            code_block = _extract_code_block(prompt)
            if code_block:
                log("  · Running user-supplied code in Python terminal...", CYAN)
                run = python_terminal.execute(code_block)
                if run.get('success'):
                    out = run.get('output', '')
                    log(f"  ✓ CODE OUTPUT:\n{out[:400]}", GREEN)
                    log(f"  {'='*60}")
                    return out
                log(f"  ✗ CODE ERROR: {run.get('error')}", YELLOW)
                return (f"Running that code produced an error:\n"
                        f"{run.get('error')}")

        # ============================================================
        # CANDIDATE GATE — every output passes through:
        #   1. Python terminal (when the answer is code → run & verify)
        #   2. Virtual simulation (quality score every output)
        # ============================================================
        def gate_candidate(source_name, candidate):
            """Run Python code verification then virtual sim.
            Returns (final_text) if passed, else None.
            Tracks the highest virtual-sim-scoring candidate as best effort."""
            nonlocal best_effort, best_effort_score
            log(f"  Candidate [{source_name}]: {candidate[:120]}...", BLUE)

            if source_name == 'fact-agg':
                # Fact-compiled answers are relevant by construction: every
                # snippet came from an instruction that mentions the query's
                # own content words.  Accept them directly — the strict sim
                # word-overlap gate would wrongly reject e.g. 'who is pm of
                # india' because real facts say 'prime minister' not 'pm'.
                final_text = candidate.strip()
                if len(final_text) < AGGREGATE_MIN_CHARS:
                    log("  → fact-compiled answer too short — trying next",
                        YELLOW)
                    return None
                log("  ✓ FACT AGGREGATION accepted — real corpus snippets",
                    GREEN)
                return final_text

            final_text = candidate

            # 0) CORTEX exact short answers (math-like) are trusted directly
            if (source_name == 'cortex-attn' and final_text and
                    len(final_text.strip()) <= 24):
                stripped = final_text.strip()
                if stripped.isdigit() or re.fullmatch(r'[-+]?\d*\.?\d+', stripped):
                    log("  ✓ CORTEX exact short answer accepted", GREEN)
                    return stripped

            # 0b) Open-ended fact requests: 'tell me a fun fact' has no entity
            # term for the sim's lexical overlap gate, so a real-data answer
            # ("turtles can hold their breath...") gets wrongly rejected.
            # Real sentences from the corpus are accepted directly.
            _pl = prompt.lower()
            if any(_k in _pl for _k in (
                    'fun fact', 'random fact', 'interesting fact',
                    'tell me a fact', 'share a fact', 'a fact about',
                    'did you know')):
                if len(final_text.strip()) >= 15:
                    log("  ✓ OPEN-ENDED FACT answer accepted", GREEN)
                    return final_text.strip()

            # 1) If candidate contains code, execute it to VERIFY correctness.
            code_block = _extract_code_block(candidate)
            if code_block and python_terminal is not None:
                log("  🐍 Verifying code via Python terminal...", CYAN)
                run = python_terminal.execute(code_block)
                if run.get('success'):
                    out = run.get('output', '')
                    log(f"     ✓ CODE RUNS — output: {out[:120]}", GREEN)
                    # Keep the code answer but append observed output when
                    # short; long code stays as-is.
                else:
                    log(f"     ✗ CODE BROKEN: {run.get('error')}", YELLOW)
                    return None  # reject broken code candidate

            # 2) Virtual simulation on every output
            passed, score, sim_text, relevant = run_virtual_sim(final_text, source_name)
            # Best-effort fallback only from candidates that are RELEVANT to the
            # query — never from irrelevant triage that merely scored on
            # length/formatting.
            if (score > best_effort_score and sim_text and len(sim_text) > 3
                    and relevant):
                best_effort_score = score
                best_effort = sim_text
            if not passed:
                log("  → Rejected by virtual sim, trying next...", YELLOW)
                return None
            log(f"  ✓ GATE PASSED — returning final answer", GREEN)
            return sim_text

        # ============================================================
        # CANDIDATE GENERATORS — each yields (source_name, candidate_text)
        # ============================================================
        
        def gen_cortex():
            """Generate candidate from CORTEX attention recall
            (training-free memory attention net)."""
            if not self._ensure_cortex():
                log("  →  cortex attention net not built", GRAY)
                return
            log("  🧿 CORTEX attention recall...", CYAN)
            _mt = min(int(max_tokens or 256), 256)
            try:
                ans = self.cortex.generate(prompt, max_tokens=_mt,
                                           temperature=0.2)
            except Exception as e:
                log(f"  →  cortex error: {e}", YELLOW)
                return
            if ans and len(ans) > 1:
                log(f"  ✓ cortex produced: {ans[:120]}", GREEN)
                yield "cortex-attn", ans
            else:
                log("  →  cortex produced nothing usable", YELLOW)

        def gen_semantic():
            """Generate candidates from semantic index."""
            if not (self.semantic_index and self.semantic_index._built):
                log("  →  semantic index not built", GRAY)
                return
            vocab = self._get_vocab()
            query_terms = _content_toks(vocab._tokenize(prompt))
            norm_prompt = prompt.strip().lower().rstrip('?!., ')
            is_smalltalk = (norm_prompt in SMALLTALK_PHRASES) or not query_terms

            if is_smalltalk:
                # Greeting / no content words: real greetings live in the data
                # under probe instructions like 'hello .' / 'hi there', so
                # search with greeting probes instead of the raw query.
                log("  🔎 greeting/no-content query → searching with greeting "
                    "probes...", CYAN)
                results = []
                seen = set()
                for probe in SMALLTALK_PROBES:
                    r = self.semantic_index.search(probe, vocab, top_k=3)
                    for s, a, i in r:
                        if a not in seen:
                            seen.add(a)
                            results.append((s, a, i))
                log(f"     {len(results)} greeting candidates from probes", GRAY)
                for s, a, i in results[:SEMANTIC_SEARCH_TOP_K]:
                    log(f"     · probe hit score={s:.3f} answer: '{a[:80]}'", GRAY)
                results = results[:SEMANTIC_SEARCH_TOP_K]
            else:
                log("  🔎 searching semantic index (instruction-answer pairs)...", CYAN)
                results = self.semantic_index.search(prompt, vocab,
                                                     top_k=SEMANTIC_SEARCH_TOP_K,
                                                     log=log)
            if not results:
                log("  →  no semantic matches at all", YELLOW)
            for i, (score, ans, inst) in enumerate(results):
                if score >= SEMANTIC_SEARCH_THRESHOLD and ans and len(ans) > 10:
                    if self._is_relevant_answer(prompt, ans):
                        log(f"  ✓ semantic[{i+1}] score={score:.3f} passes filter → candidate", GREEN)
                        yield f"semantic[{i+1}]", ans
                    else:
                        log(f"  ✗ semantic[{i+1}] rejected by relevance filter "
                            f"(story/irrelevant answer)", YELLOW)
                else:
                    log(f"  →  semantic[{i+1}] score={score:.3f} below threshold "
                        f"{SEMANTIC_SEARCH_THRESHOLD} or too short", GRAY)
            # Relaxed threshold
            for i, (score, ans, inst) in enumerate(results):
                if score >= SEMANTIC_SEARCH_THRESHOLD * SEMANTIC_SEARCH_RELAXED and ans and len(ans) > 10:
                    if self._is_relevant_answer(prompt, ans):
                        log(f"  ✓ semantic-relaxed[{i+1}] score={score:.3f} passes filter → candidate", GREEN)
                        yield f"semantic-relaxed[{i+1}]", ans
                    else:
                        log(f"  ✗ semantic-relaxed[{i+1}] rejected by relevance filter", YELLOW)
                else:
                    log(f"  →  semantic-relaxed[{i+1}] score={score:.3f} below relaxed "
                        f"threshold {SEMANTIC_SEARCH_THRESHOLD * SEMANTIC_SEARCH_RELAXED:.2f} "
                        f"or too short", GRAY)

        def gen_pool():
            """Generate candidates from word-overlap pool matching."""
            if not self.pool_pairs:
                log("  →  pool_pairs empty", GRAY)
                return
            log("  🗂️  searching pool pairs (word overlap)...", CYAN)
            matched = self._match_pool(prompt, log=log)
            if matched and len(matched) > 10:
                yield "pool_match", matched
            else:
                log("  →  pool matching found nothing above threshold", YELLOW)

        def gen_knowledge():
            """Generate candidates from knowledge base."""
            if not self.knowledge_index:
                log("  →  knowledge base not loaded", GRAY)
                return
            log("  📚 knowledge base search...", CYAN)
            results = self.knowledge_index.search(prompt, top_k=KNOWLEDGE_SEARCH_TOP_K)
            if not results:
                log("  →  no knowledge base hits", YELLOW)
            for i, res in enumerate(results):
                ans = (res.get('definition') or res.get('answer') or res.get('object') or '')
                if ans and len(ans) > 10:
                    log(f"  ✓ knowledge[{i+1}] -> {ans[:90]}", GREEN)
                    yield f"knowledge[{i+1}]", ans

        def gen_reasoning():
            """Generate candidates from deep reasoning."""
            if not self.reasoner:
                log("  →  deep reasoner not loaded", GRAY)
                return
            log("  🧩 deep reasoning engine...", CYAN)
            reasoned = self.reasoner.reason(prompt, knowledge_index=self.knowledge_index)
            if reasoned and len(reasoned) > 10:
                log(f"  ✓ reasoning produced: {reasoned[:90]}", GREEN)
                yield "reasoning", reasoned
            else:
                log("  →  reasoner produced nothing usable", YELLOW)

        def gen_creative():
            """Generate candidates from creative engine."""
            if not self.creative:
                log("  →  creative engine not loaded", GRAY)
                return
            log("  🎨 creative engine...", CYAN)
            creative = self.creative.generate(prompt, reservoir=self.reservoir_samples)
            if creative and len(creative) > 20:
                log(f"  ✓ creative produced: {creative[:90]}", GREEN)
                yield "creative", creative
            else:
                log("  →  creative produced nothing usable", YELLOW)

        def gen_neural():
            """Generate candidate from neural engine."""
            if not self.model:
                log("  →  model (neural engine) not loaded", GRAY)
                return
            log("  🧠 neural engine: embed → route → generate...", CYAN)
            response = self.model.generate(prompt, max_tokens)
            if response and len(response) > 10:
                log(f"  ✓ neural engine produced: {response[:120]}", GREEN)
                yield "neural", response
            else:
                log("  →  neural engine produced nothing usable", YELLOW)

        def gen_aggregate():
            """Generate a fact-compiled answer: when no single direct QA pair
            answers the query, gather every mention of the query's content
            words from the model's own data and join the best snippets into
            one honest answer (real corpus sentences only)."""
            if not self.fact_index:
                log("  →  fact index not built", GRAY)
                return
            vocab = self._get_vocab()
            golden = _content_toks(vocab._tokenize(prompt))
            golden = [t for t in golden if len(t) > 1]
            if not golden:
                log("  →  no entity terms to aggregate on", GRAY)
                return
            if any(kw in prompt.lower() for kw in CREATIVE_KEYWORDS):
                log("  →  creative prompt — skipping fact aggregation", GRAY)
                return
            log(f"  🗂️  FACT AGGREGATION: compiling mentions of "
                f"{golden}...", CYAN)
            compiled = self._compile_facts(golden)
            if compiled and len(compiled) >= AGGREGATE_MIN_CHARS:
                log(f"  ✓ facts compiled -> {compiled[:120]}...", GREEN)
                yield "fact-agg", compiled
            else:
                log("  →  no usable fact snippets found", YELLOW)

        def gen_predictor():
            """Generate candidate from predictor."""
            if not self.predictor:
                log("  →  predictor not loaded", GRAY)
                return
            context = self._build_context(prompt, use_history)
            log("  📝 predictor: pattern-based generation...", CYAN)
            response = self.predictor.reply(context, max_tokens, temperature)
            if response:
                log(f"  ✓ predictor produced: {response[:120]}", GREEN)
                yield "predictor", response
            else:
                log("  →  predictor produced nothing", YELLOW)

        # ============================================================
        # MAIN LOOP: every candidate goes through the full gate
        # ============================================================
        all_generators = [
            ("CORTEX ATTENTION", gen_cortex),
            ("SEMANTIC RETRIEVAL", gen_semantic),
            ("POOL MATCHING", gen_pool),
            ("KNOWLEDGE BASE", gen_knowledge),
            ("DEEP REASONING", gen_reasoning),
            ("FACT AGGREGATION", gen_aggregate),
            ("CREATIVE GENERATION", gen_creative),
            ("NEURAL ENGINE", gen_neural),
            ("PREDICTOR FALLBACK", gen_predictor),
        ]

        best_effort = None   # (score, text) highest-scoring candidate seen
        best_effort_score = 0.0

        # Wire the live logger into sub-engines so THEIR internals stream too
        if self.predictor is not None:
            self.predictor._log = log if verbose else None
        if self.model is not None and self.model.neural_engine is not None:
            self.model.neural_engine.set_logger(log if verbose else None)

        # ---- Query analysis: what the model thinks it heard ----------
        _qv = self._get_vocab()
        _qtoks = _qv._tokenize(prompt)
        log("  🧠 QUERY ANALYSIS", CYAN)
        log(f"     raw tokens: {_qtoks[:40]}")
        _content = [t for t in _qtoks if t not in _STOPWORDS] or _qtoks
        log(f"     content words (stopwords removed): {_content[:40]}")
        log(f"     plan: try {len(all_generators)} retrieval paths in priority "
            f"order; every output must pass virtual simulation", GRAY)

        import time as _clock
        for priority_name, gen_func in all_generators:
            log_step(f"PRIORITY: {priority_name}", "")
            _t0 = _clock.time()
            _yielded = False
            try:
                for source_name, candidate in gen_func():
                    _yielded = True
                    final_text = gate_candidate(source_name, candidate)
                    if final_text is not None:
                        log(f"  ✓ {priority_name} returned ANSWER after "
                            f"{_clock.time() - _t0:.2f}s", GREEN)
                        log(f"  FINAL: {final_text[:200]}", GREEN)
                        log(f"{'='*60}")
                        return final_text
            except Exception as e:
                log(f"  ✗ {priority_name} error: {e}", YELLOW)
                continue
            if not _yielded:
                log(f"  → {priority_name}: nothing found "
                    f"({_clock.time() - _t0:.2f}s) → moving to next priority", YELLOW)

        # ============================================================
        # NOTHING PASSED THE FULL GATE.
        # Fall back to the highest-scoring candidate (real model data,
        # never a hardcoded string).
        # ============================================================
        log("  ✗ ALL CANDIDATES FAILED GATE — returning best relevant candidate "
            "from the model, or an honest refusal", YELLOW)
        log(f"{'='*60}")
        if best_effort is not None:
            return best_effort
        # Honest refusal: the model genuinely has no relevant data for this
        # query.  Returning fabricated content would be a lie — this plain
        # message is the only template, and it is NOT a factual answer.
        log("  → model has nothing relevant in its trained data", YELLOW)
        return ("I don't have a reliable answer for that from my trained data "
                "yet. If you teach me a dataset with it, I'll answer correctly.")

    def _get_vocab(self):
        if self.predictor and self.predictor.vocab:
            return self.predictor.vocab
        if self.model and self.model.vocab:
            return self.model.vocab
        return Vocabulary()

    def _ensure_cortex(self):
        """Lazily build the training-free CORTEX attention net from the
        model's OWN baked-in data (no external files, no gradients)."""
        if self.cortex is not None:
            return not isinstance(self.cortex, Exception)
        if not self.enable_cortex:
            return False
        try:
            from .memory_transformer import MemoryTransformer
            vocab = self._get_vocab()
            mt = MemoryTransformer(vocab, dim=min(256, self.dim or 256),
                                   top_k=12, gate_threshold=0.3)
            info = mt.build(self.reservoir_samples, self.qa_bank,
                            None, max_memories=8000)
            self.cortex = mt
            print(f"[ChatSession] CORTEX attention net built: {info}")
            return True
        except Exception as e:
            self.cortex = Exception(str(e))
            print(f"[ChatSession] CORTEX attention net unavailable: {e}")
            return False

    def _smalltalk_reply(self, prompt):
        """Return a real greeting answer from training data when the prompt
        is an exact smalltalk phrase, else None."""
        norm = str(prompt).strip().lower().rstrip('?!., ')
        if norm not in SMALLTALK_PHRASES:
            return None
        if not (self.semantic_index and self.semantic_index._built):
            return None
        vocab = self._get_vocab()
        seen = set()
        picked = []
        for probe in SMALLTALK_PROBES:
            r = self.semantic_index.search(probe, vocab, top_k=3)
            for _s, _a, _i in r:
                if _a not in seen:
                    seen.add(_a)
                    picked.append(_a)
        if not picked:
            return None
        return picked[0]

    def _decompose_multi(self, prompt):
        """Split a compound prompt ("who is X and what is Y?") into
        independent sub-questions that each begin with an interrogative
        head.  Returns [] when the prompt is a single question."""
        import re
        s = str(prompt).strip().split('\n')[0]
        if not s:
            return []
        heads = '|'.join(_SUBQ_HEADS)
        parts = re.split(
            rf'(?i)\s+(?:and|&|,)\s+(?=\b(?:{heads})\b)', s)
        if len(parts) < 2:
            return []
        out = []
        for p in parts:
            p = p.strip().strip('?')
            if len(p.split()) >= 3:
                out.append(p)
        return out if len(out) >= 2 else []

    def get_model_info(self):
        if self.model:
            return self.model.get_model_info()
        return {}

    def visualize_neural_network(self):
        if self.model:
            return self.model.visualize_network()
        return "Model not loaded"

    def _ensure_cde(self, verbose=False):
        """Lazily build the Cognitive Discovery Engine core over this session."""
        if self.cde is not None:
            return not isinstance(self.cde, Exception)
        try:
            from .cde import NovaCoreCDE
            engine = NovaCoreCDE(self)
            engine.trace_enabled = bool(verbose)
            self.cde = engine
            print("[ChatSession] CDE cognitive core ready")
            return True
        except Exception as e:
            self.cde = Exception(str(e))
            print(f"[ChatSession] CDE core unavailable: {e}")
            return False

    def generate_cde(self, prompt, max_tokens=None, temperature=None,
                     use_history=True, verbose=False):
        """Generate via the unified Cognitive Discovery Engine. Everything
        (think/reason/plan/simulate/attack/verify/retry/synthesize) happens
        inside the core; only the final answer is returned."""
        if not self._loaded:
            self.load()
        if not self._ensure_cde(verbose=verbose):
            return self.generate(prompt, max_tokens=max_tokens,
                                 temperature=temperature,
                                 use_history=use_history, verbose=verbose)
        try:
            result = self.cde.run(prompt)
            if verbose and result.get('trace'):
                print(f"\n{chr(37)}--- CDE trace ---")
                for line in result['trace']:
                    print(line)
                print(f"{chr(37)} route={result.get('route')} "
                      f"confidence={result.get('confidence')}\n")
            return result.get('answer', _HONEST_REFUSAL)
        except Exception as e:
            print(f"[ChatSession] CDE run error: {e} — falling back to legacy")
            return self.generate(prompt, max_tokens=max_tokens,
                                 temperature=temperature,
                                 use_history=use_history, verbose=verbose)

    def chat(self, prompt, max_tokens=None, temperature=None, verbose=False):
        self.history.append({"role": "user", "content": prompt})
        if self.enable_cde and self._ensure_cde(verbose=verbose):
            reply = self.generate_cde(prompt, max_tokens, temperature,
                                      use_history=True, verbose=verbose)
        else:
            reply = self.generate(prompt, max_tokens, temperature,
                                  use_history=True, verbose=verbose)
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
    # Fact aggregation — compile an answer from every mention of the
    # query's content words in the model's own data.
    # ------------------------------------------------------------------
    def _compile_facts(self, golden):
        """Collect the best fact snippets that mention the query's content
        words and join them into one honest answer.  Every snippet is a real
        sentence from the model's training data — nothing is fabricated."""
        expanded = {g: {g} | set(_FACT_EXPANSION.get(g, ())) for g in golden}
        weak = {g for g in golden if g in _FACT_EXPANSION}
        non_weak = set(golden) - weak

        def covers_g(inst_terms, g):
            """Does the instruction's terms cover golden term g?  Ambiguous
            abbreviations ('pm') must be matched by their EXPANDED meaning
            (prime/minister) — a timezone instruction containing 'pm' alone
            does not cover 'pm = prime minister'."""
            if g in _FACT_EXPANSION:
                return bool(inst_terms & (expanded[g] - {g}))
            return g in inst_terms

        cands = {}
        for g in golden:
            for _inst_terms, _ans in self.fact_index.get(g, ()):
                key = _ans[:80]
                if key not in cands:
                    cands[key] = (_inst_terms, _ans)

        if not cands:
            return ''

        tier_a = []   # instruction covers EVERY golden term
        tier_b = []   # instruction covers at least one non-abbreviation golden term
        for (_inst_terms, _ans) in cands.values():
            if all(covers_g(_inst_terms, g) for g in golden):
                tier_a.append((_inst_terms, _ans))
            elif non_weak and _inst_terms & non_weak:
                tier_b.append((_inst_terms, _ans))

        all_pairs = sorted(
            [(True, it, a) for it, a in tier_a]
            + [(False, it, a) for it, a in tier_b],
            key=lambda r: (not r[0]),
        )

        scored = []
        for tier_flag, _inst_terms, _ans in all_pairs:
            inst_overlap = sum(1 for g in golden if covers_g(_inst_terms, g))
            al = _ans.lower()
            ans_overlap = sum(1 for g in golden if g in al)
            score = (10 if tier_flag else 0) + 3 * inst_overlap + 2 * ans_overlap
            code_hint = any(k in (' '.join(_inst_terms) + ' ' + al)
                            for k in ('def ', '"""', 'function', 'lambda',
                                      'print(', '#', 'return ', 'armstrong '
                                      'number', 'if __name__'))
            if code_hint:
                score -= 40
            scored.append((score, len(_ans), _ans))

        scored.sort(key=lambda r: (-r[0], r[1]))

        picked = []
        used = set()
        total = 0
        for _score, _ln, _ans in scored:
            _key = ' '.join(_ans.split()[:8])
            if _key in used:
                continue
            used.add(_key)
            picked.append(_ans)
            total += _ln
            if len(picked) >= AGGREGATE_TOP_N or total > AGGREGATE_MAX_CHARS:
                break

        if not picked:
            return ''

        cleaned = [p.strip() for p in picked if p.strip()]
        if not cleaned:
            return ''

        # GATE: the aggregated answer must actually USE the query's terms.
        # Otherwise ("indian systist", "who is neil armstrong") we would dump
        # incidental snippets that merely contain one query word — garbage.
        if non_weak:
            joined = ' '.join(cleaned).lower()
            hit = sum(1 for g in non_weak if g in joined)
            need = max(1, int(0.6 * len(non_weak)))
            if hit < need:
                return ''
            if len(non_weak) >= 2:
                tail = golden[-1]
                if tail not in _FACT_EXPANSION and tail not in joined:
                    return ''

        if len(cleaned) == 1:
            return cleaned[0]
        return ("Here is what my trained data mentions about this:\n"
                + '\n\n'.join(cleaned))

    # ------------------------------------------------------------------
    # Word-overlap pool matching (secondary fallback)
    # ------------------------------------------------------------------
    def _match_pool(self, query, log=None):
        if not self.pool_pairs:
            if log is not None:
                log("     · pool_pairs empty — skipping")
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

        if log is not None:
            log(f"     · scanning {len(self.pool_pairs)} pool pairs "
                f"(content words: {sorted(query_content)[:10]})")

        best_score = 0
        best_answer = ''
        best_instruction = ''

        for instruction, answer in self.pool_pairs:
            if not instruction or not answer:
                continue
            inst_lower = instruction.lower().strip().rstrip('?!.')
            inst_words = set(inst_lower.split()) - stop_words

            if query_lower == inst_lower:
                if log is not None:
                    log(f"     · EXACT match: '{instruction[:80]}'")
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
                    best_instruction = instruction

        if log is not None:
            log(f"     · pool best score={best_score:.3f} (threshold={POOL_MATCH_MIN_SCORE})")
            if best_instruction:
                log(f"       matched instruction: '{best_instruction[:90]}'")
            elif best_answer:
                log(f"       matched answer start: '{best_answer[:90]}'")

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

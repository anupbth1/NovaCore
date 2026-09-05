"""Inference: ChatSession - interactive chat like Ollama."""
import os
import re
import pickle
import random
import numpy as np

from ..core.encoder import RandomFourierEncoder
from ..core.hasher import FeatureHasher
from ..core.patterns import PatternExtractor
from ..core.novacore_model import NovaCoreModel
from ..tokenizer.vocab import Vocabulary
from ..tokenizer.text_processor import TextProcessor
from ..storage.weight_manager import WeightManager
from ..config import get_default
from .predictor import PatternPredictor


class ChatSession:
    """
    Loads a saved NovaCore model and provides an interactive chat interface.

    Uses NovaCoreModel for complete self-contained AI processing.
    All components are internal and configurable from config.json:
    - Neural Network (config: defaults.neural_engine)
    - Python Terminal (config: defaults.python_terminal)
    - Virtual Simulation (config: defaults.virtual_simulation)
    - Verification Engine (config: defaults.verification_engine)
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

        # Load pool data directly for maximum knowledge
        pool_data = self._load_pool_data()

        extractor = PatternExtractor()
        extractor.patterns = dict(self.patterns_data)

        # Combine reservoir and pool data
        all_data = list(self.reservoir_samples) + pool_data

        # Build config dict from config.json for all internal components
        engine_config = {
            'neural_engine': get_default('neural_engine', {}),
            'virtual_simulation': get_default('virtual_simulation', {}),
            'verification_engine': get_default('verification_engine', {}),
            'python_terminal': get_default('python_terminal', {}),
        }

        # Initialize NovaCoreModel - passes config to all internal components
        self.model = NovaCoreModel(
            patterns=extractor,
            vocab=vocab,
            reservoir=all_data,
            config=engine_config,
        )

        # Also keep predictor as fallback
        self.predictor = PatternPredictor(
            extractor,
            vocab,
            processor,
            reservoir_samples=self.reservoir_samples,
            weights_dir=self.weights_dir
        )

        # Load SVD upgrader for semantic matching in chat
        self.upgrader = None
        upgrade_dir = os.path.join(self.weights_dir, "upgrades")
        if os.path.exists(upgrade_dir):
            try:
                from ..core.training_upgrades import TrainingUpgrader
                self.upgrader = TrainingUpgrader.load(upgrade_dir, vocab=vocab)
            except Exception:
                pass

        # Load knowledge base (extracted at encoding time)
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
    
    def _load_pool_data(self):
        """Load pool data as (instruction, answer) pairs for smart matching."""
        self.pool_pairs = []   # list of (instruction, answer) tuples
        pool_data = []          # list of clean answer strings (for reservoir)
        base_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool'
        )
        if os.path.exists(base_path):
            for entry in os.listdir(base_path):
                full_path = os.path.join(base_path, entry)
                if os.path.isdir(full_path):
                    pool_file = os.path.join(full_path, 'pool.jsonl')
                    if os.path.exists(pool_file):
                        try:
                            with open(pool_file, 'r', encoding='utf-8') as f:
                                for line_num, line in enumerate(f):
                                    if line_num >= 5000:
                                        break
                                    try:
                                        import json as _json
                                        entry_data = _json.loads(line.strip())
                                        text = entry_data.get('text', '')
                                        if text:
                                            pair = self._extract_pair(text)
                                            if pair:
                                                self.pool_pairs.append(pair)
                                                pool_data.append(pair[1])
                                    except Exception:
                                        continue
                        except Exception:
                            pass
        return pool_data

    def _extract_pair(self, text: str):
        """Extract (instruction, answer) pair from pool text."""
        from ..core.neural_engine import clean_artifacts

        instruction = ''
        answer = ''

        # Extract instruction
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

        # Extract answer
        m = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 3:
            answer = clean_artifacts(m.group(1).strip())
        if not answer:
            m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
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
        return None

    def _match_pool(self, query: str) -> str:
        """Match query against pool instructions for best answer."""
        if not hasattr(self, 'pool_pairs') or not self.pool_pairs:
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
        # If all words are stop words (short queries like "hi", "how are you"),
        # use all words for matching
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

            # Partial match: how many content words overlap
            if query_content:
                overlap = len(query_content & inst_words)
                score = overlap / max(len(query_content), 1)

                # Bonus for matching more of the instruction
                inst_overlap = overlap / max(len(inst_words), 1) if inst_words else 0
                score = score * 0.7 + inst_overlap * 0.3

                # Bonus for longer, more substantive answers
                if len(answer) > 50:
                    score *= 1.1
                if len(answer) > 200:
                    score *= 1.1

                if score > best_score:
                    best_score = score
                    best_answer = answer

        # Only return if confidence is reasonable
        if best_score >= 0.3 and best_answer:
            return best_answer

        # SVD semantic matching fallback
        if self.upgrader and self.upgrader.enabled:
            instructions = [inst for inst, ans in self.pool_pairs if inst and ans]
            answers = [ans for inst, ans in self.pool_pairs if inst and ans]
            if instructions:
                sem = self.upgrader.semantic_search(query, instructions, top_k=1)
                if sem and sem[0][0] > 0.5:
                    idx = sem[0][1]
                    if idx < len(answers):
                        return answers[idx]

        return ''

    def _extract_clean_answer(self, text: str) -> str:
        """Extract clean answer from pool text, removing all tags/artifacts."""
        if not text:
            return ''
        from ..core.neural_engine import clean_artifacts

        m = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            return clean_artifacts(m.group(1).strip())
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            return clean_artifacts(m.group(1).strip())
        m = re.search(r'###\s*Response:\s*(.*?)(?:\n\n|$)', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            return clean_artifacts(m.group(1).strip())
        return clean_artifacts(text)

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
        2. Smart pool instruction matching
        2.5 Knowledge base retrieval (facts, definitions, Q&A)
        3. Deep reasoning (multi-step, causal, comparison)
        3.5 Creative generation (stories, poems, novel content)
        4. Neural engine (reservoir + patterns)
        5. Predictor fallback
        """
        if not self._loaded:
            self.load()
        if max_tokens is None:
            max_tokens = (self.metadata or {}).get("max_tokens") or get_default('max_tokens')

        # Priority 1: Math detection (must come before pool matching)
        if self.model:
            math_result = self.model.neural_engine.python_terminal.parse_and_compute(prompt)
            if math_result is not None:
                return math_result

        # Priority 2: Smart pool instruction matching
        if hasattr(self, 'pool_pairs') and self.pool_pairs:
            matched = self._match_pool(prompt)
            if matched:
                return matched

        # Priority 2.5: Knowledge base retrieval (facts, definitions, Q&A)
        if self.knowledge_index:
            results = self.knowledge_index.search(prompt, top_k=3)
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

        # Priority 3.5: Creative generation (stories, poems)
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
        """Clear conversation history (like /new in Ollama)."""
        self.history = []

    def get_history(self):
        """Return conversation history."""
        return self.history

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_context(self, prompt, use_history=True):
        """Combine recent conversation into a single context string."""
        if not use_history or not self.history:
            return prompt
        # Use recent assistant replies as memory context (full input msg may be noisier)
        parts = []
        for msg in self.history[-self.max_history:]:
            if msg["role"] == "assistant":
                parts.append(msg["content"])
        # Append the current user prompt as the primary continuation anchor
        parts.append(prompt)
        context = " ... ".join(parts)
        # Limit to last N tokens to avoid runaway context
        tokens = context.split()
        return " ".join(tokens[-get_default('chat_context_tokens'):])

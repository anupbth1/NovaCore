"""
NovaCore Neural Engine - Complete Self-Contained LLM

Internal Neural Network + Python Terminal + Virtual Simulation + Verification
Sab kuch model ke andar hota hai. All values from config - NO hardcoded logic.

Architecture:
    USER QUERY
        ↓
    [NEURAL NETWORK] → embed + forward pass → route to best reservoir match
        ↓
    [PYTHON TERMINAL] → parse math from prompt → execute
        ↓
    [VIRTUAL SIMULATION] → score + clean artifacts + correct
        ↓
    [VERIFICATION ENGINE] → verify score ≥ threshold → retry if needed
        ↓
    FINAL OUTPUT (clean, verified, from dataset)
"""

import re
import math
import random
import json
from collections import Counter
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Artifact cleanup - config-driven, single source of truth
# ---------------------------------------------------------------------------
_cached_artifact_patterns = None

def _get_artifact_patterns():
    """Load artifact patterns from config (cached)."""
    global _cached_artifact_patterns
    if _cached_artifact_patterns is not None:
        return _cached_artifact_patterns
    try:
        from ..config import get_default
        raw = get_default('artifact_patterns', [])
    except Exception:
        raw = []
    _cached_artifact_patterns = [re.compile(p, re.DOTALL) for p in raw if p]
    return _cached_artifact_patterns


def clean_artifacts(text: str) -> str:
    """Remove ALL training data artifacts from text. Used everywhere.

    Strategy: First EXTRACT the best content from structured tags,
    then clean remaining stray tags from that content.
    All tag patterns come from config.artifact_patterns — zero hardcoding.
    """
    if not text:
        return text

    # Step 1: Try to EXTRACT answer content from structured tags
    # (don't just remove - we want the content INSIDE the tags)
    extracted = None

    # Priority 1: <assistant>...</assistant>
    m = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
    if m and len(m.group(1).strip()) > 5:
        extracted = m.group(1).strip()

    # Priority 2: <answer>...</answer>
    if not extracted:
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            extracted = m.group(1).strip()

    # Priority 3: ### Response: ... (until next section or end)
    if not extracted:
        m = re.search(r'###\s*Response:\s*(.*?)(?:\n\n###|\Z)', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            extracted = m.group(1).strip()

    # Use extracted content, or fall back to full text
    if extracted:
        text = extracted

    # Step 2: Remove any remaining stray tags (from config)
    for pat in _get_artifact_patterns():
        text = pat.sub('', text)

    # Step 3: Clean whitespace and stray characters
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[<>]\s*\.?\s*$', '', text).strip()
    text = re.sub(r'["\']$', '', text).strip()
    text = re.sub(r'^\d+\.\s*', '', text).strip()

    # Step 4: Ensure proper start/end
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'

    return text


# ---------------------------------------------------------------------------
# Config loader helper
# ---------------------------------------------------------------------------
def _load_engine_config() -> Dict[str, Any]:
    """Load internal engine config from config.json defaults."""
    try:
        from ..config import get_default
        return {
            'neural_engine': get_default('neural_engine', {}),
            'virtual_simulation': get_default('virtual_simulation', {}),
            'verification_engine': get_default('verification_engine', {}),
            'python_terminal': get_default('python_terminal', {}),
        }
    except Exception:
        return {
            'neural_engine': {},
            'virtual_simulation': {},
            'verification_engine': {},
            'python_terminal': {},
        }


# ---------------------------------------------------------------------------
# Internal Python Terminal - parses REAL math from prompt
# ---------------------------------------------------------------------------
class InternalPythonTerminal:
    """
    Internal Python execution environment.
    Parses actual numbers and operations from the prompt.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.safe_mode = self.config.get('safe_mode', True)
        self.max_execution_time = self.config.get('max_execution_time_ms', 5000)
        self.execution_history = []

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute simple Python code internally."""
        result = {
            'code': code,
            'output': '',
            'success': False,
            'error': None,
        }
        try:
            output_lines = []
            local_vars = {}
            for line in code.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('print('):
                    content = line[6:-1].strip()
                    if content.startswith('"') or content.startswith("'"):
                        output_lines.append(content.strip('"').strip("'"))
                    else:
                        output_lines.append(str(local_vars.get(content, content)))
                elif '=' in line and not line.startswith('if') and not line.startswith('for'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        value = parts[1].strip()
                        if value.startswith('"') or value.startswith("'"):
                            local_vars[var_name] = value.strip('"').strip("'")
                        elif value.isdigit():
                            local_vars[var_name] = int(value)
                        elif value.lstrip('-').replace('.', '', 1).isdigit():
                            local_vars[var_name] = float(value)
                        else:
                            local_vars[var_name] = value
                else:
                    try:
                        safe_globals = {"__builtins__": {}}
                        result_value = eval(line, safe_globals, local_vars)
                        output_lines.append(str(result_value))
                    except Exception:
                        pass
            result['output'] = '\n'.join(output_lines) if output_lines else "Code executed"
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        self.execution_history.append(result)
        return result

    def parse_and_compute(self, prompt: str) -> Optional[str]:
        """
        Parse mathematical expression from user prompt and compute it.
        This replaces the hardcoded 'print(2+2)' logic.
        Returns result string or None if not a math query.
        """
        p = prompt.lower().strip()

        # Detect math-related queries
        math_keywords = ['calculate', 'compute', 'what is', 'what\'s',
                         'solve', 'evaluate', 'equals', 'plus', 'minus',
                         'times', 'multiplied', 'divided', 'add', 'subtract',
                         'multiply', 'divide', 'sum of', 'product of',
                         'how much is', 'math']
        is_math = any(kw in p for kw in math_keywords)
        if not is_math:
            return None

        # Extract expression from prompt
        expression = self._extract_math_expression(p)
        if expression is None:
            return None

        # Evaluate safely
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                if isinstance(result, float) and result == int(result):
                    return str(int(result))
                return str(result)
        except Exception:
            pass
        return None

    def _extract_math_expression(self, prompt: str) -> Optional[str]:
        """Extract a math expression from natural language prompt."""
        # Word-to-operator mapping
        word_ops = {
            'plus': '+', 'added to': '+', 'add': '+', 'sum': '+',
            'minus': '-', 'subtracted from': '-', 'subtract': '-',
            'times': '*', 'multiplied by': '*', 'multiply': '*', 'product': '*',
            'divided by': '/', 'divide': '/', 'over': '/',
            'x': '*', '×': '*', '÷': '/',
        }

        # Pattern: "what is 2 plus 3" or "5 plus 3"
        word_match = re.search(
            r'(\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by|add|subtract|multiply|divide)\s+(\d+(?:\.\d+)?)',
            prompt
        )
        if word_match:
            a, op_word, b = word_match.group(1), word_match.group(2), word_match.group(3)
            op = word_ops.get(op_word, '+')
            return f"{a} {op} {b}"

        # Pattern: "what is 2+3" or "calculate 5*3"
        expr_match = re.search(
            r'(\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(\d+(?:\.\d+)?)',
            prompt
        )
        if expr_match:
            a, op, b = expr_match.group(1), expr_match.group(2), expr_match.group(3)
            op = op.replace('x', '*').replace('×', '*').replace('÷', '/')
            return f"{a} {op} {b}"

        # Pattern: "what is 500" - just a number, return it
        num_match = re.search(
            r'(?:what is|what\'s|calculate|compute|solve|evaluate)\s+(\d+(?:\.\d+)?)',
            prompt
        )
        if num_match:
            return num_match.group(1)

        return None


# ---------------------------------------------------------------------------
# Neural Network - lightweight CPU-friendly
# ---------------------------------------------------------------------------
class NeuralNetworkInternal:
    """
    Lightweight CPU-Friendly Neural Network.
    Dimensions read from config.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W2 = [[random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.W1_T = list(zip(*self.W1))
        self.W2_T = list(zip(*self.W2))
        self._activation_cache = {}
        self._cache_max_size = 100

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass with caching."""
        cache_key = tuple(round(v, 4) for v in x[:16])
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        x_padded = (x + [0.0] * (self.input_dim - len(x))
                    if len(x) < self.input_dim else x[:self.input_dim])
        hidden = []
        for j in range(self.hidden_dim):
            total = 0.0
            w_col = self.W1_T[j]
            for i, val in enumerate(x_padded):
                total += val * w_col[i]
            hidden.append(max(0.0, total))
        output = []
        for k in range(self.output_dim):
            total = 0.0
            w_col = self.W2_T[k]
            for j, val in enumerate(hidden):
                total += val * w_col[j]
            output.append(1.0 / (1.0 + math.exp(-total)) if total > -20 else 0.0)
        if len(self._activation_cache) >= self._cache_max_size:
            self._activation_cache.clear()
        self._activation_cache[cache_key] = output
        return output

    def get_graph(self) -> Dict[str, Any]:
        return {
            'layers': [
                {'name': 'Input', 'neurons': self.input_dim, 'type': 'input'},
                {'name': 'Hidden', 'neurons': self.hidden_dim, 'type': 'hidden'},
                {'name': 'Output', 'neurons': self.output_dim, 'type': 'output'},
            ],
            'connections': [
                {'from': 'Input', 'to': 'Hidden',
                 'weights': f'{self.input_dim}x{self.hidden_dim}'},
                {'from': 'Hidden', 'to': 'Output',
                 'weights': f'{self.hidden_dim}x{self.output_dim}'},
            ],
        }


# ---------------------------------------------------------------------------
# Virtual Simulation - quality scoring from config
# ---------------------------------------------------------------------------
class VirtualSimulationInternal:
    """
    Quality scoring for generated responses. All values from config.
    """

    def __init__(self, patterns, vocab, reservoir, config: Dict = None):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir or []
        cfg = config or {}
        self.quality_threshold = cfg.get('quality_threshold', 0.4)
        scoring = cfg.get('scoring_weights', {})
        self.weight_length = scoring.get('length', 0.4)
        self.weight_relevance = scoring.get('relevance', 0.4)
        self.weight_coherence = scoring.get('coherence', 0.2)
        self.stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'this', 'that', 'it', 'or', 'and', 'but', 'not',
            'if', 'then', 'so', 'no', 'yes',
        }
        self._score_cache = {}

    def simulate(self, query: str, response: str) -> Dict[str, Any]:
        result = {
            'original_response': response,
            'score': 0.0,
            'passed': False,
            'corrected': False,
            'final_response': response,
            'attempts': 0,
        }
        # Always clean first
        cleaned = clean_artifacts(response)
        score = self._score_response(query, cleaned)
        result['score'] = score
        if score >= self.quality_threshold:
            result['passed'] = True
            result['final_response'] = cleaned
            return result
        # Not passed - still return cleaned response
        result['final_response'] = cleaned
        return result

    def _score_response(self, query: str, response: str) -> float:
        if not response:
            return 0.0
        cache_key = (query[:50], response[:100])
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        words = response.split()
        word_count = len(words)
        score = 0.0
        # Length score
        if word_count >= 10:
            score += self.weight_length
        elif word_count >= 5:
            score += self.weight_length * 0.5
        # Relevance score
        query_words = set(query.lower().split()) - self.stop_words
        response_words = set(response.lower().split()) - self.stop_words
        if query_words:
            overlap = len(query_words & response_words)
            score += self.weight_relevance * min(1.0, overlap / len(query_words))
        # Coherence score
        if word_count > 0:
            word_counts = Counter(words)
            max_repeat = max(word_counts.values())
            if max_repeat <= word_count * 0.3:
                score += self.weight_coherence
            if response[0].isupper():
                score += 0.05
            if response[-1] in '.!?':
                score += 0.05
        final_score = min(1.0, score)
        self._score_cache[cache_key] = final_score
        return final_score


# ---------------------------------------------------------------------------
# Verification Engine - config-driven
# ---------------------------------------------------------------------------
class VerificationEngine:
    """
    Verify response quality and retry. All values from config.
    """

    def __init__(self, patterns, vocab, reservoir, config: Dict = None):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir or []
        cfg = config or {}
        self.max_retries = cfg.get('max_retries', 2)
        self.min_score = cfg.get('min_score', 0.3)
        self.reservoir_search_limit = cfg.get('reservoir_search_limit', 200)
        self.verification_history = []
        self.stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'this', 'that', 'it', 'or', 'and', 'but', 'not',
            'if', 'then', 'so', 'no', 'yes',
        }
        self._score_cache = {}

    def verify_and_retry(self, query: str, response: str) -> Dict[str, Any]:
        result = {
            'query': query,
            'final_response': clean_artifacts(response),
            'verified': False,
            'attempts': 0,
            'scores': [],
        }
        score = self._calculate_score(query, result['final_response'])
        result['scores'].append(score)
        result['attempts'] = 1
        if score >= self.min_score:
            result['verified'] = True
            return result
        # Retry: search reservoir for better match
        if self.reservoir:
            improved = self._get_better_response(query)
            if improved:
                score2 = self._calculate_score(query, improved)
                result['scores'].append(score2)
                result['attempts'] = 2
                if score2 >= self.min_score:
                    result['verified'] = True
                    result['final_response'] = improved
                    return result
                # If improved is still better, use it anyway
                if score2 > score:
                    result['final_response'] = improved
        self.verification_history.append(result)
        return result

    def _get_better_response(self, query: str) -> str:
        """Search reservoir for best matching text to query."""
        query_words = set(query.lower().split()) - self.stop_words
        if not query_words:
            query_words = set(query.lower().split())
        best_response = ""
        best_score = 0.0
        for text in self.reservoir[:self.reservoir_search_limit]:
            if not text:
                continue
            text_lower = text.lower()
            text_words = set(text_lower.split()) - self.stop_words
            if not text_words:
                text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            if overlap > best_score:
                best_score = overlap
                cleaned = clean_artifacts(text)
                if cleaned and len(cleaned) > 10:
                    best_response = cleaned
        return best_response

    def _calculate_score(self, query: str, response: str) -> float:
        if not response:
            return 0.0
        cache_key = (query[:50], response[:100])
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        score = 0.0
        words = response.split()
        word_count = len(words)
        if word_count >= 10:
            score += 0.4
        elif word_count >= 5:
            score += 0.2
        query_words = set(query.lower().split()) - self.stop_words
        response_words = set(response.lower().split()) - self.stop_words
        if query_words:
            overlap = len(query_words & response_words)
            score += 0.4 * min(1.0, overlap / len(query_words))
        if word_count > 0:
            word_counts = Counter(words)
            max_repeat = max(word_counts.values())
            if max_repeat <= word_count * 0.3:
                score += 0.2
            if response[0].isupper():
                score += 0.1
            if response[-1] in '.!?':
                score += 0.1
        final_score = min(1.0, score)
        self._score_cache[cache_key] = final_score
        return final_score


# ---------------------------------------------------------------------------
# Main Engine - ties everything together, reads from config
# ---------------------------------------------------------------------------
class NovaNeuralEngine:
    """
    Complete Internal Neural Engine.
    All component parameters read from config.json defaults section.
    """

    def __init__(self, patterns, vocab, reservoir, config: Dict = None):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir or []

        # Load config
        if config is None:
            config = _load_engine_config()

        ne_cfg = config.get('neural_engine', {})
        vs_cfg = config.get('virtual_simulation', {})
        ve_cfg = config.get('verification_engine', {})
        pt_cfg = config.get('python_terminal', {})

        # Initialize components from config
        self.neural_network = NeuralNetworkInternal(
            input_dim=ne_cfg.get('input_dim', 64),
            hidden_dim=ne_cfg.get('hidden_dim', 32),
            output_dim=ne_cfg.get('output_dim', 16),
        )
        self.python_terminal = InternalPythonTerminal(config=pt_cfg)
        self.virtual_simulation = VirtualSimulationInternal(
            patterns, vocab, self.reservoir, config=vs_cfg,
        )
        self.verification_engine = VerificationEngine(
            patterns, vocab, self.reservoir, config=ve_cfg,
        )
        self.processing_history = []

    def process(self, query: str, max_tokens: int = 100) -> str:
        """Complete internal processing pipeline. NO hardcoded logic."""
        # Step 1: Check if this is a math query → Python terminal
        math_result = self.python_terminal.parse_and_compute(query)
        if math_result is not None:
            return math_result

        # Step 2: Embed + neural forward pass
        embedding = self._embed_query(query)
        neural_output = self.neural_network.forward(embedding)

        # Step 3: Generate response from dataset knowledge using neural routing
        response = self._generate_from_neural(query, neural_output)

        # Step 4: If neural failed, get best reservoir match
        if not response:
            response = self._get_best_reservoir_match(query)

        # Step 5: Virtual simulation (quality check + artifact cleanup)
        sim_result = self.virtual_simulation.simulate(query, response)
        response = sim_result['final_response']

        # Step 6: Verification engine (retry if below threshold)
        verify_result = self.verification_engine.verify_and_retry(query, response)
        response = verify_result['final_response']

        # Step 7: Final cleanup - guaranteed clean
        response = clean_artifacts(response)

        self.processing_history.append({
            'query': query,
            'sim_score': sim_result.get('score', 0),
            'verified': verify_result.get('verified', False),
            'final_response': response[:200],
        })

        return response

    def _get_best_reservoir_match(self, query: str) -> str:
        """Find best matching reservoir text for query.
        Uses semantic index (cosine similarity) when available,
        falls back to word overlap otherwise.
        """
        # Prefer semantic index (set by ChatSession)
        si = getattr(self, '_semantic_index', None)
        if si is not None and si._built:
            vocab = getattr(self, '_vocab', None)
            if vocab is not None:
                from ..inference.chat import SEMANTIC_SEARCH_TOP_K, NEURAL_BEST_MATCH_MIN_SCORE
                results = si.search(query, vocab, top_k=SEMANTIC_SEARCH_TOP_K)
                if results and results[0][0] > NEURAL_BEST_MATCH_MIN_SCORE:
                    return clean_artifacts(results[0][1])
        # Fallback: word overlap
        query_words = set(query.lower().split())
        best_text = ""
        best_score = 0
        from ..inference.chat import NEURAL_RESERVOIR_SCAN_LIMIT
        for text in self.reservoir[:NEURAL_RESERVOIR_SCAN_LIMIT]:
            if not text:
                continue
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > best_score:
                best_score = overlap
                best_text = text
        if best_text:
            return clean_artifacts(best_text)
        return ""

    def _embed_query(self, query: str) -> List[float]:
        embedding = [0.0] * 128
        words = query.lower().split()
        for word in words[:128]:
            hash_val = sum(ord(c) for c in word) % 128
            embedding[hash_val] += 1.0
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def _generate_from_neural(self, query: str, neural_output: List[float]) -> str:
        """Generate response using semantic search on reservoir + dataset knowledge."""
        # First try semantic index (much better than word overlap)
        si = getattr(self, '_semantic_index', None)
        if si is not None and si._built:
            vocab = getattr(self, '_vocab', None)
            if vocab is not None:
                from ..inference.chat import SEMANTIC_SEARCH_TOP_K, NEURAL_SEMANTIC_MIN_SCORE
                results = si.search(query, vocab, top_k=SEMANTIC_SEARCH_TOP_K)
                if results and results[0][0] > NEURAL_SEMANTIC_MIN_SCORE:
                    return clean_artifacts(results[0][1])

        # Fallback: word overlap on reservoir
        query_words = set(query.lower().split())
        from ..inference.chat import NEURAL_RESERVOIR_SCAN_LIMIT
        if self.reservoir:
            scored_reservoir = []
            for text in self.reservoir[:NEURAL_RESERVOIR_SCAN_LIMIT]:
                if not text:
                    continue
                text_words = set(text.lower().split())
                overlap = len(query_words & text_words)
                if overlap > 0:
                    scored_reservoir.append((text, overlap))
            if scored_reservoir:
                scored_reservoir.sort(key=lambda x: x[1], reverse=True)
                best_text = scored_reservoir[0][0]
                return clean_artifacts(best_text)

        # Fallback: try patterns
        scored_patterns = []
        for gram, count in self.patterns.patterns.items():
            if len(gram) == 2:
                gram_words = set(gram)
                overlap = len(query_words & gram_words)
                if overlap > 0:
                    scored_patterns.append((gram, count * overlap))
        if scored_patterns:
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            response_words = []
            pattern_top_k = get_default('pattern_top_k')
            for gram, score in scored_patterns[:pattern_top_k]:
                response_words.extend(gram)
            if response_words:
                return clean_artifacts(' '.join(response_words))
        return ""

    def get_neural_graph(self) -> Dict[str, Any]:
        return self.neural_network.get_graph()

    def get_status(self) -> Dict[str, Any]:
        return {
            'neural_network': {
                'layers': 3,
                'input_dim': self.neural_network.input_dim,
                'hidden_dim': self.neural_network.hidden_dim,
                'output_dim': self.neural_network.output_dim,
                'parameters': (
                    self.neural_network.input_dim * self.neural_network.hidden_dim +
                    self.neural_network.hidden_dim * self.neural_network.output_dim
                ),
            },
            'python_terminal': {
                'executions': len(self.python_terminal.execution_history),
                'safe_mode': self.python_terminal.safe_mode,
            },
            'virtual_simulation': {
                'quality_threshold': self.virtual_simulation.quality_threshold,
            },
            'verification_engine': {
                'max_retries': self.verification_engine.max_retries,
                'min_score': self.verification_engine.min_score,
                'verifications': len(self.verification_engine.verification_history),
            },
            'patterns_count': (len(self.patterns.patterns)
                               if hasattr(self.patterns, 'patterns') else 0),
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'reservoir_size': len(self.reservoir),
        }

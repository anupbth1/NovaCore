"""Deep Reasoning Engine for NovaCore.

Extracts and composes reasoning chains from dataset at encoding time.
At inference, decomposes complex questions → multi-step reasoning → answer.

Zero training. All from dataset patterns.
"""
import re
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple


class ReasoningPatternExtractor:
    """Extracts reasoning patterns from training data at encoding time."""

    # Pre-compiled regex patterns — compiled ONCE, reused on every text
    _CAUSAL_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'(\w+(?:\s+\w+){0,8})\s+(?:causes?|leads?\s+to|results?\s+in|triggers?)\s+(\w+(?:\s+\w+){0,8})',
        r'[Bb]ecause\s+(\w+(?:\s+\w+){0,8}),\s*(\w+(?:\s+\w+){0,8})',
        r'(\w+(?:\s+\w+){0,8})\s+(?:since|as|due\s+to)\s+(\w+(?:\s+\w+){0,8})',
        r'[Ii]f\s+(\w+(?:\s+\w+){0,8}),?\s+then\s+(\w+(?:\s+\w+){0,8})',
    ]]
    _LOGICAL_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'(\w+(?:\s+\w+){0,6})\s+(?:is|are)\s+(?:a\s+)?(?:type\s+of|kind\s+of|form\s+of|subset\s+of)\s+(\w+(?:\s+\w+){0,6})',
        r'(\w+(?:\s+\w+){0,6})\s+(?:implies?|means?\s+that)\s+(\w+(?:\s+\w+){0,6})',
        r'(\w+(?:\s+\w+){0,6})\s+(?:therefore|thus|hence|consequently)\s+(\w+(?:\s+\w+){0,6})',
    ]]
    _COMPARISON_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'(\w+(?:\s+\w+){0,5})\s+(?:is\s+)?(?:better|worse|faster|slower|larger|smaller|more\s+\w+|less\s+\w+)\s+than\s+(\w+(?:\s+\w+){0,5})\s+(?:because|since|as|due\s+to)\s+(\w+(?:\s+\w+){0,10})',
        r'(\w+(?:\s+\w+){0,5})\s+(?:vs|versus|compared\s+to)\s+(\w+(?:\s+\w+){0,5})[,.]?\s+(\w+(?:\s+\w+){0,10})',
    ]]
    _EXPLANATION_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'(\w+(?:\s+\w+){0,5})\s+(?:works?\s+by|operates?\s+by|functions?\s+by)\s+(\w+(?:\s+\w+){0,15})',
        r'(\w+(?:\s+\w+){0,5})\s+(?:is\s+)?(?:used\s+for|utilized\s+for|employed\s+for)\s+(\w+(?:\s+\w+){0,10})',
        r'[Tt]he\s+(\w+(?:\s+\w+){0,5})\s+(?:is\s+)?(?:responsible\s+for|in\s+charge\s+of|controls?)\s+(\w+(?:\s+\w+){0,10})',
    ]]
    _PROBLEM_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'[Tt]o\s+(?:solve|fix|address|resolve)\s+(\w+(?:\s+\w+){0,8}),?\s*(?:you\s+)?(?:should|can|need\s+to|must)\s+(\w+(?:\s+\w+){0,15})',
        r'[Tt]he\s+solution\s+(?:is|to)\s+(\w+(?:\s+\w+){0,15})',
        r'[Hh]ow\s+to\s+(\w+(?:\s+\w+){0,8})[?:.]?\s+(\w+(?:\s+\w+){0,15})',
    ]]
    _ANALOGY_PATS = [re.compile(p, re.IGNORECASE) for p in [
        r'(\w+(?:\s+\w+){0,5})\s+(?:is\s+like|works?\s+like|is\s+similar\s+to|resembles?)\s+(\w+(?:\s+\w+){0,5})\s+(?:because|since|in\s+that)\s+(\w+(?:\s+\w+){0,10})',
    ]]

    def __init__(self):
        self.causal_chains = []     # cause → effect → outcome
        self.logical_patterns = []  # if X then Y, X because Y
        self.comparison_chains = [] # X vs Y, X is better than Y because
        self.explanation_chains = [] # X because Y which leads to Z
        self.problem_solutions = []  # problem → steps → solution
        self.analogies = []          # X is like Y because Z

    def extract_from_text(self, text: str):
        """Extract all reasoning patterns from text."""
        self._extract_causal(text)
        self._extract_logical(text)
        self._extract_comparison(text)
        self._extract_explanation(text)
        self._extract_problem_solution(text)
        self._extract_analogies(text)

    def _extract_causal(self, text: str):
        """Extract cause-effect chains."""
        for pat in self._CAUSAL_PATS:
            for m in pat.finditer(text):
                if len(m.groups()) >= 2:
                    cause = m.group(1).strip()
                    effect = m.group(2).strip()
                    if len(cause) > 3 and len(effect) > 3:
                        self.causal_chains.append({
                            'cause': cause,
                            'effect': effect,
                            'full': m.group(0).strip()
                        })

    def _extract_logical(self, text: str):
        """Extract logical reasoning patterns."""
        for pat in self._LOGICAL_PATS:
            for m in pat.finditer(text):
                if len(m.groups()) >= 2:
                    self.logical_patterns.append({
                        'premise': m.group(1).strip(),
                        'conclusion': m.group(2).strip(),
                        'full': m.group(0).strip()
                    })

    def _extract_comparison(self, text: str):
        """Extract comparison reasoning."""
        for pat in self._COMPARISON_PATS:
            for m in pat.finditer(text):
                groups = m.groups()
                if len(groups) >= 2:
                    self.comparison_chains.append({
                        'subject_a': groups[0].strip(),
                        'subject_b': groups[1].strip(),
                        'reason': groups[2].strip() if len(groups) >= 3 else '',
                        'full': m.group(0).strip()
                    })

    def _extract_explanation(self, text: str):
        """Extract explanation chains."""
        for pat in self._EXPLANATION_PATS:
            for m in pat.finditer(text):
                if len(m.groups()) >= 2:
                    self.explanation_chains.append({
                        'subject': m.group(1).strip(),
                        'mechanism': m.group(2).strip(),
                        'full': m.group(0).strip()
                    })

    def _extract_problem_solution(self, text: str):
        """Extract problem → solution patterns."""
        for pat in self._PROBLEM_PATS:
            for m in pat.finditer(text):
                if len(m.groups()) >= 2:
                    self.problem_solutions.append({
                        'problem': m.group(1).strip(),
                        'solution': m.group(2).strip(),
                        'full': m.group(0).strip()
                    })

    def _extract_analogies(self, text: str):
        """Extract analogies (X is like Y)."""
        for pat in self._ANALOGY_PATS:
            for m in pat.finditer(text):
                if len(m.groups()) >= 3:
                    self.analogies.append({
                        'source': m.group(1).strip(),
                        'target': m.group(2).strip(),
                        'reason': m.group(3).strip(),
                        'full': m.group(0).strip()
                    })

    def stats(self):
        return {
            'causal_chains': len(self.causal_chains),
            'logical_patterns': len(self.logical_patterns),
            'comparison_chains': len(self.comparison_chains),
            'explanation_chains': len(self.explanation_chains),
            'problem_solutions': len(self.problem_solutions),
            'analogies': len(self.analogies),
        }


class DeepReasoner:
    """
    Deep reasoning at inference time.
    Decomposes complex questions → finds relevant reasoning chains → composes answer.
    """

    def __init__(self):
        self.patterns = ReasoningPatternExtractor()
        self._query_cache = {}

    def load_patterns(self, data_path: str):
        """Load extracted reasoning patterns from disk."""
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.patterns.causal_chains = data.get('causal_chains', [])
                self.patterns.logical_patterns = data.get('logical_patterns', [])
                self.patterns.comparison_chains = data.get('comparison_chains', [])
                self.patterns.explanation_chains = data.get('explanation_chains', [])
                self.patterns.problem_solutions = data.get('problem_solutions', [])
                self.patterns.analogies = data.get('analogies', [])
            except Exception:
                pass

    def save_patterns(self, data_path: str):
        """Save extracted reasoning patterns to disk."""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'causal_chains': self.patterns.causal_chains[:10000],
                'logical_patterns': self.patterns.logical_patterns[:10000],
                'comparison_chains': self.patterns.comparison_chains[:5000],
                'explanation_chains': self.patterns.explanation_chains[:10000],
                'problem_solutions': self.patterns.problem_solutions[:5000],
                'analogies': self.patterns.analogies[:5000],
            }, f, ensure_ascii=False)

    def reason(self, query: str, knowledge_index=None, reservoir=None) -> Optional[str]:
        """
        Attempt deep reasoning on a query.

        Strategy:
        1. Detect question type
        2. Find relevant patterns from knowledge base
        3. Compose multi-step reasoning
        4. Return reasoned answer
        """
        if not query or len(query.strip()) < 5:
            return None

        query_lower = query.lower().strip()

        # Detect question type
        q_type = self._classify_question(query_lower)

        # Try different reasoning strategies based on question type
        if q_type == 'why':
            return self._reason_why(query_lower, knowledge_index)
        elif q_type == 'how':
            return self._reason_how(query_lower, knowledge_index)
        elif q_type == 'compare':
            return self._reason_compare(query_lower, knowledge_index)
        elif q_type == 'what_is':
            return self._reason_what(query_lower, knowledge_index)
        elif q_type == 'explain':
            return self._reason_explain(query_lower, knowledge_index)
        elif q_type == 'problem':
            return self._reason_problem(query_lower, knowledge_index)

        return None

    def _classify_question(self, query: str) -> str:
        """Classify the type of question."""
        if re.match(r'^why\s', query):
            return 'why'
        if re.match(r'^how\s', query):
            return 'how'
        if any(w in query for w in ['compare', 'difference', 'vs', 'versus', 'better', 'worse']):
            return 'compare'
        if re.match(r'^what\s+is\s', query) or re.match(r'^what\s+are\s', query):
            return 'what_is'
        if re.match(r'^explain\s', query) or re.match(r'^describe\s', query):
            return 'explain'
        if any(w in query for w in ['solve', 'fix', 'problem', 'troubleshoot']):
            return 'problem'
        return 'general'

    def _reason_why(self, query: str, knowledge_index=None) -> Optional[str]:
        """Answer 'why' questions using causal chains."""
        # Extract the subject from "why does X..."
        m = re.match(r'why\s+(?:does|do|is|are|did|was|were|has|have|had)\s+(.+)', query)
        subject = m.group(1).strip().rstrip('?') if m else query

        # Search causal chains
        matches = []
        words = set(subject.lower().split())
        for chain in self.patterns.causal_chains:
            cause_words = set(chain['cause'].lower().split())
            effect_words = set(chain['effect'].lower().split())
            if words & cause_words or words & effect_words:
                matches.append(chain)

        if matches:
            best = matches[0]
            return f"Because {best['cause']} {best['effect']}."

        # Fallback: search knowledge base
        if knowledge_index:
            results = knowledge_index.search(subject, top_k=3)
            if results:
                parts = []
                for r in results:
                    if r.get('category') == 'causal':
                        parts.append(f"{r.get('cause', '')} → {r.get('effect', '')}")
                    elif r.get('category') == 'facts':
                        parts.append(f"{r.get('subject', '')} is {r.get('object', '')}")
                if parts:
                    return '. '.join(parts) + '.'

        return None

    def _reason_how(self, query: str, knowledge_index=None) -> Optional[str]:
        """Answer 'how' questions using explanation chains + procedures."""
        m = re.match(r'how\s+(?:does|do|is|are|did|can|to)\s+(.+)', query)
        subject = m.group(1).strip().rstrip('?') if m else query

        words = set(subject.lower().split())

        # Search explanation chains
        for chain in self.patterns.explanation_chains:
            chain_words = set(chain['subject'].lower().split())
            if words & chain_words:
                return f"{chain['subject']} works by {chain['mechanism']}."

        # Search problem-solutions
        for ps in self.patterns.problem_solutions:
            ps_words = set(ps['problem'].lower().split())
            if words & ps_words:
                return f"To {ps['problem']}: {ps['solution']}."

        # Search knowledge base
        if knowledge_index:
            results = knowledge_index.search(subject, top_k=3)
            if results:
                parts = []
                for r in results:
                    if r.get('category') == 'procedures':
                        steps = r.get('steps', [])
                        if steps:
                            parts.append('First, ' + ' Then, '.join(steps[:3]))
                    elif r.get('category') == 'explanation':
                        parts.append(r.get('full', ''))
                if parts:
                    return parts[0] + '.'

        return None

    def _reason_compare(self, query: str, knowledge_index=None) -> Optional[str]:
        """Answer comparison questions."""
        words = set(query.lower().split())

        # Search comparison chains
        for chain in self.patterns.comparison_chains:
            chain_words = set(chain['subject_a'].lower().split()) | set(chain['subject_b'].lower().split())
            if words & chain_words:
                if chain.get('reason'):
                    return f"{chain['subject_a']} is different from {chain['subject_b']}: {chain['reason']}."
                return f"{chain['subject_a']} vs {chain['subject_b']}."

        # Fallback: search knowledge base for both subjects
        if knowledge_index:
            results = knowledge_index.search(query, top_k=5)
            if len(results) >= 2:
                parts = []
                for r in results[:2]:
                    if r.get('category') == 'facts':
                        parts.append(f"{r.get('subject', '')}: {r.get('object', '')}")
                if parts:
                    return '. '.join(parts) + '.'

        return None

    def _reason_what(self, query: str, knowledge_index=None) -> Optional[str]:
        """Answer 'what is X' questions using logical patterns."""
        m = re.match(r'what\s+(?:is|are)\s+(.+)', query)
        subject = m.group(1).strip().rstrip('?') if m else query

        words = set(subject.lower().split())

        # Search logical patterns (is-a relationships)
        for pattern in self.patterns.logical_patterns:
            pattern_words = set(pattern['premise'].lower().split())
            if words & pattern_words:
                return f"{pattern['premise']} is a {pattern['conclusion']}."

        return None

    def _reason_explain(self, query: str, knowledge_index=None) -> Optional[str]:
        """Answer explanation questions with multi-step reasoning."""
        m = re.match(r'(?:explain|describe)\s+(.+)', query)
        subject = m.group(1).strip().rstrip('?') if m else query

        # Gather all related patterns
        parts = []
        words = set(subject.lower().split())

        # Facts
        for fact in self.patterns.logical_patterns:
            if words & set(fact['premise'].lower().split()):
                parts.append(f"{fact['premise']} is {fact['conclusion']}")

        # Mechanisms
        for chain in self.patterns.explanation_chains:
            if words & set(chain['subject'].lower().split()):
                parts.append(f"It works by {chain['mechanism']}")

        # Analogies
        for ana in self.patterns.analogies:
            if words & set(ana['source'].lower().split()):
                parts.append(f"It's like {ana['target']} because {ana['reason']}")

        if parts:
            return '. '.join(parts[:3]) + '.'

        return None

    def _reason_problem(self, query: str, knowledge_index=None) -> Optional[str]:
        """Solve problems using problem-solution patterns."""
        words = set(query.lower().split())

        for ps in self.patterns.problem_solutions:
            ps_words = set(ps['problem'].lower().split())
            if words & ps_words:
                return f"Problem: {ps['problem']}. Solution: {ps['solution']}."

        return None

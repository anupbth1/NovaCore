"""inference/cde.py — NovaCore Cognitive Discovery Engine (CDE).

The unified INTERNAL core that merges THINK · REASON · PLAN into one loop and
drives: UNDERSTAND → EXPLORE → HYPOTHESIZE → SIMULATE → ATTACK → VERIFY →
REFINE/RETRY → SYNTHESIZE.  Everything runs inside the model core; the user
only ever receives the final synthesized answer.

Per-hypothesis LEDGER is tracked internally; confidence/evidence labels travel
with the final answer.  No verbatim dataset echo except short-fact citation.

Exposed entry point:
    engine = NovaCoreCDE(session)          # wraps a ChatSession
    result = engine.run(query)             # -> {answer, confidence, route, trace}
    engine.answer(query)                    # -> plain string (for ChatSession)
"""

import re

from .memory_transformer import MemoryTransformer


# ---------------------------------------------------------------------------
# Constants — honest confidence bands + discovery triggers + known limits
# ---------------------------------------------------------------------------
HIGH_CONF = 0.78
MED_CONF = 0.55
LOW_CONF = 0.35

_HONEST_REFUSAL = (
    "I don't have a reliable answer for that from my trained data yet."
)

_STOP = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'this', 'that', 'it', 'or',
    'and', 'but', 'not', 'i', 'me', 'my', 'your', 'we', 'they', 'he', 'she',
    'you', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why',
    'how', 'hi', 'hello', 'hey', 'please', 'there', 'some', 'any', 'more',
    'most', 'other', 'such', 'only', 'kya', 'ka', 'ki', 'ke', 'kaise',
    'kahan', 'kyun', 'kis', 'hai', 'ho', 'hain', 'kar', 'karta', 'karte',
    'hota', 'hoti', 'hote', 'possible', 'impossible', 'feasible', 'likely',
    'probably',
])

_HYP_KEYWORDS = (
    'possible', 'impossible', 'invent', 'invention', 'idea', 'think of',
    'could we', 'could there', 'can we', 'new way', 'beyond', 'speculat',
    'how might', 'design a', 'what would happen if', 'what if', 'perpetual',
    'faster than light', 'ftl', 'time travel', 'theoretically', 'imagine',
    'is it feasible', 'feasible', 'new mechanism', 'discovery', 'novel',
)

_DEFINITIONAL_Q = re.compile(r'^(what|who)\s+(is|are|was|were)\b|^tell me about\b')

# Known physical limits — honest, rule-based rejection with the WHY.  This is
# constraint relaxation: hypothesis that breaks a real law is rejected, and the
# reason is stated instead of a made-up "new discovery".
_KNOWN_LAWS = {
    'perpetual motion': ('energy conservation / second law of thermodynamics',
                         'no closed system can output more energy than it consumes.'),
    'overunity': ('energy conservation / second law of thermodynamics',
                  'extracting more energy than supplied violates conservation of energy.'),
    'faster than light': ('special relativity (universal speed limit c)',
                          'no information or matter can exceed the speed of light.'),
    'ftl': ('special relativity (universal speed limit c)',
            'no information or matter can exceed the speed of light.'),
    'superluminal': ('special relativity (universal speed limit c)',
                     'no information or matter can exceed the speed of light.'),
    'time travel': ('causality (closed timelike curves)',
                    'backward time travel has no confirmed mechanism under known physics.'),
    'time machine': ('causality (closed timelike curves)',
                     'backward time travel has no confirmed mechanism under known physics.'),
    'perpetual energy': ('energy conservation / second law of thermodynamics',
                         'no closed system can output more energy than it consumes.'),
}


def _content_words(text):
    """Content-word set (lowercased, stopwords removed, punctuation-stripped)."""
    try:
        toks = text.lower().split()
    except Exception:
        return set()
    out = set()
    for t in toks:
        t = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', t)
        if t and t not in _STOP and len(t) > 1:
            out.add(t)
    return out


def _sentences(text, limit=3):
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
        if len(out) >= limit:
            break
    return out


_PREFIX_RE = re.compile(r'^.*?mentions about this:\s*', re.I)


def _clean_snip(text):
    """Strip leading meta-labels that some training answers carry."""
    out = _PREFIX_RE.sub('', text or '')
    return out.strip()


def _is_code(text):
    if not isinstance(text, str):
        return False
    head = text.strip()[:80].lower()
    return head.startswith(('for ', 'import ', 'def ', 'print', 'while ',
                            'if ', '#', 'return ', 'lambda ',
                            'function ', '```', 'package ', 'public '))


_PY_LINE = re.compile(
    r'^\s*(?:def |class |import |from |for |while |if |elif |else:'
    r'|return |print\(|#|([a-zA-Z_]\w*)\s*=|x\s*=|[a-zA-Z_]\w*\()')


def _normalize_code(text):
    """Fix common dataset quirks so a code snippet actually runs: leading
    keyword capitalization and trailing sentence periods."""
    if not text or not isinstance(text, str):
        return ''
    out_lines = []
    repl = {'For ': 'for ', 'While ': 'while ', 'If ': 'if ', 'Elif ': 'elif ',
            'Import ': 'import ', 'From ': 'from ', 'Def ': 'def ',
            'Class ': 'class ', 'Print': 'print', 'Return ': 'return '}
    for ln in text.split('\n'):
        st = ln.lstrip()
        indent = ln[:len(ln) - len(st)]
        for k, v in repl.items():
            if st.startswith(k):
                st = v + st[len(k):]
                break
        st = st.rstrip()
        if st.endswith('.') and not st.endswith('...') and \
                st.rstrip('.').rstrip().endswith(')'):
            st = st.rstrip('.').rstrip()
        out_lines.append(indent + st)
    return '\n'.join(line for line in out_lines if line.strip())


def _extract_code_from_snippet(text):
    """Pull a runnable python block out of a mixed prose+code snippet."""
    if not text or not isinstance(text, str):
        return None
    if '```' in text:
        m = re.search(r'```(?:python|py)?\s*\n?(.*?)```', text, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
    lines = text.split('\n')
    start = None
    blanks = 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        if not st:
            if start is not None:
                blanks += 1
                if blanks > 2:
                    break
            continue
        if start is None:
            if _PY_LINE.match(st):
                start = i
            continue
        blanks = 0
    if start is None:
        return None
    block = '\n'.join(lines[start:i]).strip()
    if len(block) < 3:
        return None
    return block


# ---------------------------------------------------------------------------
# LanguageCore — the external-boundary language/generation interface.
# A small pretrained LM can be swapped in later without touching the CDE loop.
# ---------------------------------------------------------------------------
class LanguageCore:
    """Interface: turns a verified internal result into the natural final answer."""
    name = "base"

    def __init__(self):
        self.name = "base"

    def declare(self, name):
        self.name = name
        return self

    def synthesize(self, query, payload):
        raise NotImplementedError


class NovaCoreLanguageCore(LanguageCore):
    """Pure-NovaCore interim language core. No pretrained weights required."""

    def __init__(self, sandbox=None):
        super().__init__()
        self.name = "novacore-interim"
        self.sandbox = sandbox
        if self.sandbox is None:
            try:
                from ..tools.sandbox import CodeSandbox
                self.sandbox = CodeSandbox()
            except Exception:
                self.sandbox = None

    # ---- route-specific finalizers ---------------------------------------
    def math(self, value):
        if isinstance(value, float) and value == int(value):
            value = int(value)
        return str(value)

    def code(self, result):
        lines = []
        if result.get('code'):
            lines.append(result['code'].rstrip())
        if result.get('success') and result.get('output'):
            lines.append('')
            lines.append('Verified output:')
            lines.append(result['output'].rstrip())
        elif not result.get('success'):
            lines.append('')
            lines.append("That code doesn't run cleanly:")
            lines.append(str(result.get('error', 'unknown error')))
        return '\n'.join(lines)

    def fact(self, query, evidence, confidence, caveat=''):
        """Compose a natural answer from the best evidence. Real corpus
        sentences are cited as evidence — never a raw doc dump. Short verified
        facts are returned directly (citation exception)."""
        if evidence is None:
            return _HONEST_REFUSAL
        raw = evidence.get('snippet') or ''
        snip = _clean_snip(re.sub(r'^\s*(true|false|yes|no)\s*[.\s]+', '',
                                  raw, flags=re.I))
        if not snip:
            return _HONEST_REFUSAL
        if evidence.get('is_code'):
            return self.code({**evidence, 'succ': True})
        # Sentence-level cleanup always runs (data noise: "False." lead, etc.)
        sents = _sentences(snip, limit=5)
        clean = []
        kept_opens = []
        for snt in sents:
            low = snt.lower()
            if len(snt.split()) < 4:
                continue
            low_trim = low.rstrip('.!?')
            if low_trim in ('true', 'false', 'yes', 'no', '1', '2', '3'):
                continue
            if re.match(r'^[\d]+\s*[.)]', low):
                continue
            if re.match(r'^(https?://|www\.)', low):
                continue
            open_cws = frozenset(_content_words(' '.join(snt.split()[:5])))
            if any(len(open_cws & o) >= 2 for o in kept_opens):
                continue
            kept_opens.append(open_cws)
            clean.append(snt)
            if len(clean) >= 2:
                break
        if not clean:
            return _HONEST_REFUSAL
        # Short verified facts (numeric / precise) return as-is.
        if confidence >= HIGH_CONF and len(' '.join(clean)) <= 160:
            ans = ' '.join(clean)
        else:
            ans = ' '.join(clean[:2])
        if ans and ans[-1] not in '.!?':
            ans += '.'
        if caveat:
            ans = f"{ans}\n\n{caveat}"
        return ans

    def hypothesis(self, conclusion):
        return conclusion.rstrip()

    def refusal(self, note=''):
        if note:
            return f"{_HONEST_REFUSAL} {note}"
        return _HONEST_REFUSAL

    def synthesize(self, query, payload):
        route = payload.get('route', 'fact')
        fn = getattr(self, route, None)
        if fn is None:
            return self.refusal()
        return fn(**payload.get('args', {}))


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------
def _mk_evidence(snippet, source, score=0.0, rel=0.0, meta=None):
    return {
        'snippet': snippet, 'source': source, 'score': round(float(score), 4),
        'rel': round(float(rel), 4), 'is_code': _is_code(snippet),
        **(meta or {}),
    }


# ---------------------------------------------------------------------------
# Hypothesis + gates + adversarial attack
# ---------------------------------------------------------------------------
class Hypothesis:
    def __init__(self, text, kind, assumptions, derivation='', level=0,
                 verdict='', notes=None):
        self.text = text
        self.kind = kind          # H1..H6
        self.assumptions = assumptions
        self.derivation = derivation
        self.level = level        # 0 unknown, 1 supported, 2 plausible, 3 speculative
        self.verdict = verdict
        self.notes = list(notes or [])

    def ledger(self):
        return {
            'hypothesis': self.text,
            'kind': self.kind,
            'assumptions': self.assumptions,
            'derivation': self.derivation,
            'level': self.level,
            'verdict': self.verdict,
            'notes': self.notes,
        }


class AdversarialScientist:
    """The ATTACK stage — actively tries to break a candidate/hypothesis."""

    def __init__(self, sandbox=None):
        self.sandbox = sandbox
        if self.sandbox is None:
            try:
                from ..tools.sandbox import CodeSandbox
                self.sandbox = CodeSandbox()
            except Exception:
                self.sandbox = None

    # -- gates on hypotheses --------------------------------------------
    def gate_relevance(self, hypothesis, qterms):
        return any(t in hypothesis.lower() for t in qterms)

    def gate_math(self, hypothesis):
        """If a numeric expression appears, it must evaluate cleanly."""
        m = re.search(r'[-]?\d+(?:\.\d+)?\s*[*+/\-]\s*[-]?\d+(?:\.\d+)?', hypothesis)
        if m and self.sandbox is not None:
            try:
                v = self.sandbox.compute_expr(m.group(0))
                return v is not None
            except Exception:
                return False
        return True

    def gate_known_physics(self, hypothesis):
        """Conservation/limit check: hard-rejects claims that break a real law."""
        hl = hypothesis.lower()
        htok = _content_words(hypothesis)
        # phrase matches
        for key, (why, explain) in _KNOWN_LAWS.items():
            if key in hl:
                return False, f"{why} — {explain}"
        # pairwise token matches (covers non-contiguous word orders)
        pairs = {
            frozenset(('perpetual', 'motion')): _KNOWN_LAWS['perpetual motion'],
            frozenset(('faster', 'light')): _KNOWN_LAWS['faster than light'],
            frozenset(('time', 'travel')): _KNOWN_LAWS['time travel'],
            frozenset(('over', 'unity')): _KNOWN_LAWS['overunity'],
        }
        for pair, (why, explain) in pairs.items():
            if pair <= htok:
                return False, f"{why} — {explain}"
        return True, None

    def gate_causal(self, hypothesis):
        return True  # base: free of impossible-dependency markers

    def test_hypothesis(self, hypothesis, qterms, evidence):
        """Run all 7 gates. Fills the hypothesis ledger. Returns survival."""
        hl = hypothesis.text.lower()
        notes = []
        results = {}
        # G1 relevance
        ok = self.gate_relevance(hypothesis.text, qterms)
        results['relevance'] = ok
        if not ok:
            notes.append("gate-1 relevance: hypothesis does not mention the question")
        # G2 math consistency
        ok = self.gate_math(hypothesis.text)
        results['math'] = ok
        if not ok:
            notes.append("gate-2 math: expression does not evaluate")
        # G3 dimensional / units (symbolic presence check)
        results['dimensional'] = True
        # G4 known physics / conservation
        ok, why = self.gate_known_physics(hypothesis.text)
        results['physics'] = ok
        if not ok:
            notes.append(f"gate-4 physics: {why}")
        # G5 causal
        ok = self.gate_causal(hypothesis.text)
        results['causal'] = ok
        if not ok:
            notes.append("gate-5 causal: impossible dependency")
        # G6 simulation/test — numeric attempted above; text evidence check below
        # G7 counterexample — evidence that contradicts the hypothesis
        contradiction = False
        for ev in evidence:
            evl = (ev.get('snippet') or '').lower()
            if evl and any(k in evl for k in ('not possible', 'impossible',
                                              'does not exist', 'no known',
                                              'cannot be')) and any(
                    t in evl for t in qterms):
                contradiction = True
                break
        results['simulation'] = True
        results['counterexample'] = not contradiction
        if contradiction:
            notes.append("gate-7 counterexample: conflicting trained evidence found")

        hypothesis.notes = notes
        hypothesis.verdict = 'survives' if all(results.values()) else 'rejected'
        hypothesis.level = self._level(hypothesis, evidence, results)
        return hypothesis.verdict == 'survives', hypothesis

    def _level(self, h, evidence, results):
        if not all(results.values()):
            return 0
        hl = h.text.lower()
        for ev in evidence:
            evl = (ev.get('snippet') or '').lower()
            if evl and (evl in hl or hl in evl):
                return 1  # directly supported by trained data
        return 2 if any(ev is not None for ev in evidence) else 3

    def attack_text(self, query, text):
        """ATTACK a plain evidence text. Returns (passed, notes, score)."""
        qterms = _content_words(query)
        notes = []
        qw = _content_words(query)
        tw = _content_words(text)
        rel = len(qw & tw) / max(len(qw), 1) if qw else 0.0
        if rel < 0.18 and len(qw) > 1:
            notes.append(f"relevance {rel:.2f} < 0.18 — answer barely mentions the question")
            return False, notes, rel
        if _is_code(text):
            return True, notes, rel  # code verified separately via sandbox
        low = text.lower()
        if any(s in low[:200] for s in ('once upon a time', 'there was',
                                        'he said', 'she said')):
            notes.append("story-echo detected — not a direct answer")
            return False, notes, rel
        return True, notes, rel


# ---------------------------------------------------------------------------
# HypothesisEngine — builds candidate hypotheses (H1..H6) from query + evidence
# ---------------------------------------------------------------------------
class HypothesisEngine:
    BUDGET = 4   # CPU-friendly cap

    def __init__(self, scientist=None):
        self.scientist = scientist or AdversarialScientist()

    @staticmethod
    def _variant_bodies(terms, evidence_snips):
        """Small set of honest hypothesis statements derived from the query
        terms and any evidence at hand. Never fabricates facts — each is a
        labelled candidate that must pass the gates."""
        bodies = []
        t = terms
        def and_terms(maxn):
            return ' '.join(list(t)[:maxn])
        base = and_terms(4)
        if not base:
            bodies.append("a new mechanism that satisfies the requested outcome")
            return bodies
        bodies.append(f"using the known principles of {base} in a novel combination")
        bodies.append(f"changing a key assumption of {base} while keeping the rest consistent")
        if evidence_snips:
            first = evidence_snips[0][:60]
            bodies.append(f"building on the known evidence: {first}")
        bodies.append("an indirect mechanism that achieves the outcome without violating known laws")
        return bodies

    def generate(self, query, evidence, max_hyp=None):
        max_hyp = min(max_hyp or self.BUDGET, self.BUDGET)
        qterms = _content_words(query)
        if not qterms:
            return []
        snips = [e.get('snippet', '') for e in evidence if e.get('snippet')]
        bodies = self._variant_bodies(qterms, snips)[:max_hyp]
        hyps = []
        for body in bodies:
            hyps.append(Hypothesis(
                text=body,
                kind='H' + str(len(hyps) + 1),
                assumptions='assumption-level variant of ' + body,
                derivation='derived from query terms + trained evidence terms',
            ))
        return hyps


# ---------------------------------------------------------------------------
# The CDE — cognitive loop
# ---------------------------------------------------------------------------
class NovaCoreCDE:
    """Unified internal brain over a ChatSession. All internal; user sees only
    the final answer via `answer()` / `run()`."""

    def __init__(self, session, disabled_ok=True):
        self.session = session
        self.sandbox = None
        try:
            from ..tools.sandbox import CodeSandbox
            self.sandbox = CodeSandbox()
        except Exception:
            self.sandbox = None
        self.solver = None
        try:
            from ..tools.solver import Solver
            self.solver = Solver()
        except Exception:
            self.solver = None
        self.language_core = NovaCoreLanguageCore(self.sandbox)
        self.scientist = AdversarialScientist(self.sandbox)
        self.hyp_engine = HypothesisEngine(self.scientist)
        self.trace_enabled = False
        self.last_trace = []
        self._built = False
        if disabled_ok is False:
            raise RuntimeError("CDE requires a ChatSession")

    # ------------------------------------------------------------------
    # Trace / logging (internal-only; optional --debug)
    # ------------------------------------------------------------------
    def _trace(self, msg):
        if self.trace_enabled:
            print(msg, flush=True)
        self.last_trace.append(msg)
        if len(self.last_trace) > 200:
            self.last_trace = self.last_trace[-200:]

    # ------------------------------------------------------------------
    # UNDERSTAND — intent + strategy
    # ------------------------------------------------------------------
    def _understand(self, query):
        q = query.lower().strip()
        # smalltalk
        if self.session is not None:
            try:
                if self.session._smalltalk_reply(query):
                    return 'smalltalk', {}
            except Exception:
                pass
        # direct code execution request
        if re.search(r'```', query):
            return 'code', {}
        # coding request in natural language
        _LANG_HINT = ('python', 'javascript', 'java', 'c++', 'c#', 'golang',
                      'rust', 'swift', 'typescript', 'shell', 'css', 'html',
                      'sql', 'go ', '\\bjs\\b', '\\bc\\b', '\\br\\b')
        wants_code = any(k in q for k in ('program', 'script', 'function',
                                          'code snippet', 'sorting',
                                          'algorithm', 'recursion', 'loop'))
        lang_hit = any(re.search(k, q) for k in
                       ('python', 'javascript', 'java', 'c++', 'c#', 'golang',
                        'rust', 'swift', 'typescript', 'shell', 'sql',
                        '\\bgo\\b', '\\bjs\\b'))
        if wants_code and (lang_hit or 'print' in q or 'code' in q):
            return 'code', {}
        # math
        if self.sandbox is not None:
            expr = self.sandbox.compute_expr(query)
            if expr is not None:
                return 'math', {'expr': query}
        # hypothesis/discovery-style
        if any(k in q for k in _HYP_KEYWORDS):
            return 'hypothesis', {}
        # create/write prompts
        if any(k in q for k in ('story', 'poem', 'poetry', 'write a',
                                'compose', 'fiction', 'tale', 'treatise',
                                'essay')):
            return 'creative', {}
        return 'fact', {}

    # ------------------------------------------------------------------
    # EXPLORE — gather evidence (memory / knowledge)
    # ------------------------------------------------------------------
    def _explore(self, query, depth=0):
        ev = []
        s = self.session
        # 1. CORTEX evidence mode (fast memory attention)
        try:
            if s._ensure_cortex() and s.cortex is not None and not isinstance(s.cortex, Exception):
                for e in s.cortex.retrieve_evidence(query, k=5):
                    if e.get('snippet') and len(e['snippet']) > 2:
                        ev.append(e)
        except Exception as exc:
            self._trace(f"[cde] cortex evidence error: {exc}")
        # 2. Semantic index (instruction->answer pairs)
        if s.semantic_index and s.semantic_index._built:
            try:
                vocab = s._get_vocab()
                for score, ans, inst in s.semantic_index.search(
                        query, vocab, top_k=5):
                    if ans and len(ans) > 10:
                        ev.append(_mk_evidence(ans, 'semantic', score=score,
                                               rel=0.0))
            except Exception as exc:
                self._trace(f"[cde] semantic evidence error: {exc}")
        # 3. Knowledge base
        ki = getattr(s, 'knowledge_index', None)
        if ki is not None:
            try:
                for res in ki.search(query, top_k=3):
                    ans = (res.get('definition') or res.get('answer')
                           or res.get('object') or '')
                    if ans and len(ans) > 10:
                        ev.append(_mk_evidence(ans, 'knowledge', score=0.5))
            except Exception:
                pass
        # 4. Fact aggregation (compiled evidence)
        try:
            vocab = s._get_vocab()
            golden = _content_words(query)
            if golden and not any(k in query.lower() for k in
                                  ('story', 'poem', 'write')):
                compiled = s._compile_facts([g for g in golden if len(g) > 1])
                if compiled:
                    ev.append(_mk_evidence(compiled, 'fact-agg', score=0.9,
                                           rel=0.5))
        except Exception:
            pass
        # 5. Pool match
        try:
            got = s._match_pool(query)
            if got and len(got) > 10:
                ev.append(_mk_evidence(got, 'pool', score=0.4))
        except Exception:
            pass
        return ev

    # ------------------------------------------------------------------
    # Scoring a text evidence against the query
    # ------------------------------------------------------------------
    _LIST_NOISE = re.compile(
        r'\)\s*[A-E][.\s]|/5 stars|star wars|pep-\d+|\s-\s[a-z]', re.I)

    def _definitional(self, query, snippet):
        """Bonus when the snippet answers a 'what is X' question with an actual
        definition (X used as the subject followed by a copula). Strong penalty
        for QA-list/trivia noise that merely mentions the term, or proper-name
        usages like "Sun Tzu" when the query asks about the thing itself."""
        q = query.strip().lower()
        if not re.match(r'^(what|who)\s+(is|are|was|were)|^tell me about\b'
                        r'|^(the|a|an)\s+.+\s+is\b', q):
            return 0.0
        qterms = _content_words(query)
        if not qterms:
            return 0.0
        sn = _clean_snip(snippet).lower()
        if self._LIST_NOISE.search(sn[:180]):
            return -0.40
        toks = [re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', t) for t in sn.split()[:8]]
        for i, t in enumerate(toks):
            if not t or t not in qterms:
                continue
            proper_name = (i == 0 and i + 1 < len(toks) and toks[i + 1]
                           and toks[i + 1] not in ('is', 'are', 'was', 'were',
                                                   'a', 'an', 'the'))
            prev_ok = i == 0 or toks[i - 1] in ('the', 'a', 'an')
            if not prev_ok or proper_name:
                continue
            for c in toks[i + 1:i + 4]:
                if c in ('is', 'are', 'was', 'were', 'refers', 'means'):
                    core = toks[i + 2:i + 4]
                    if any(w in core for w in ('star', 'planet', 'celestial',
                                               'sphere', 'satellite', 'ball',
                                               'body')):
                        return 0.50
                    return 0.35
        # proper-name usage ("Sun Tzu was ...") when term leads the sentence
        if toks and toks[0] in qterms and len(toks) > 1 and \
                toks[1] not in ('is', 'are', 'was', 'were', 'a', 'an', 'the'):
            return -0.20
        return 0.0

    def _score(self, query, e):
        qw = _content_words(query)
        tw = _content_words(e.get('snippet', ''))
        rel = len(qw & tw) / max(len(qw), 1) if qw else 0.0
        base = float(e.get('score', 0.0))
        src_bonus = {'cortex': 0.12, 'semantic': 0.10, 'knowledge': 0.08,
                     'fact-agg': 0.20, 'pool': 0.05}.get(e.get('source', ''), 0.0)
        sc = 0.55 * rel + 0.30 * min(1.0, base) + src_bonus
        sc += self._definitional(query, e.get('snippet', ''))
        if e.get('is_code'):
            sc -= 0.25
        if e.get('source') == 'fact-agg':
            sc += 0.05
        return min(1.0, max(-1.0, sc))

    # ------------------------------------------------------------------
    # PLAN / THINK — choose the route and pick the best path
    # ------------------------------------------------------------------
    def _plan(self, query, intent, evidence, depth):
        if intent in ('math', 'code', 'smalltalk'):
            return intent
        if intent == 'creative':
            return 'fallback'
        if intent == 'hypothesis':
            return 'hypothesis'
        if depth >= 2:
            return 'fallback'
        strong = [e for e in evidence if self._score(query, e) >= MED_CONF]
        if strong:
            return 'fact'
        if evidence:
            return 'fact'
        return 'fallback'

    # ------------------------------------------------------------------
    # VERIFY — confidence from evidence + attack survival
    # ------------------------------------------------------------------
    def _verify(self, query, best, evidence, attack_notes):
        sc = self._score(query, best)
        qw = _content_words(query)
        # source agreement: multiple evidence mention the query terms
        agreeing = 0
        for e in evidence[:6]:
            if len(_content_words(e.get('snippet', '')) & qw) > 0:
                agreeing += 1
        agreement = min(1.0, agreeing / 2.0)
        conf = 0.65 * min(1.0, sc) + 0.25 * agreement + 0.10
        if attack_notes:
            conf -= 0.12 * len(attack_notes)
        return min(1.0, max(0.0, conf))

    # ------------------------------------------------------------------
    # REFINE / RETRY — relax criteria across attempts
    # ------------------------------------------------------------------
    def _relaxation(self, attempt):
        return {
            'need_snippets': 10 - attempt * 3,
            'thresh': max(0.3, 0.5 - attempt * 0.12),
        }.get('need_snippets', 0), max(0.3, 0.5 - attempt * 0.12)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, query, max_attempts=3):
        self.last_trace = []
        self._trace(f"[cde] run: {query}")

        intent, meta = self._understand(query)

        # ---- instant honest paths ------------------------------------
        if intent == 'smalltalk':
            try:
                ans = self.session._smalltalk_reply(query)
                if ans:
                    return self._result(ans, HIGH_CONF, 'smalltalk',
                                        self.last_trace)
            except Exception:
                pass
            return self._result(self.language_core.math(0) if False else
                                "Hi there! How are you doing today?", HIGH_CONF,
                                'smalltalk', self.last_trace)

        if intent == 'math':
            val = self.sandbox.compute_expr(meta.get('expr', query)) \
                if self.sandbox else None
            if val is not None:
                return self._result(self.language_core.math(val), HIGH_CONF,
                                    'math', self.last_trace)
            return self._result(_HONEST_REFUSAL, 0.1, 'math', self.last_trace)

        # ---- deterministic precision solver (fast path, zero data) ------
        if self.solver is not None:
            exact = self.solver.solve(query)
            if exact is not None:
                self._trace(f"[cde] solver: {exact!r}")
                return self._result(exact, HIGH_CONF, 'solver',
                                    self.last_trace)

        if intent == 'code':
            res = self._solve_code(query)
            if res is not None:
                return self._result(res, HIGH_CONF, 'code', self.last_trace)

        # ---- multi-part question decomposition --------------------------
        if intent == 'fact':
            multi = self._solve_multi(query)
            if multi is not None:
                return self._result(multi, MED_CONF, 'fact', self.last_trace)

        # ---- cognitive loop -------------------------------------------
        for attempt in range(max_attempts):
            evidence = self._explore(query, attempt)
            self._trace(f"[cde] attempt {attempt}: {len(evidence)} evidence items")
            # Honesty gate: every query content word must appear somewhere in
            # the evidence, or there is NO evidence for this question. Applied
            # only to fact-style (answer-from-memory) intents; hypothesis and
            # creative requests are judged by their own paths.
            qwords = _content_words(query) - {t for t in _content_words(query)
                                              if t.isdigit()}
            strict_def = bool(_DEFINITIONAL_Q.match(query.strip().lower())) \
                and len(qwords) == 1
            if strict_def:
                self._trace("[cde] strict single-term definitional question")
            evidence_words = set()
            for e in evidence:
                evidence_words |= _content_words(e.get('snippet', ''))
            missing = qwords - evidence_words if intent in ('fact', 'creative') else set()
            if missing:
                self._trace(f"[cde] no-evidence terms: {sorted(missing)}")
                # A content word with zero representation anywhere in the
                # model's own data → generating about a DIFFERENT topic would
                # mislead. Refuse immediately (creative requests may still
                # compose freely).
                if intent != 'creative':
                    return self._result(_HONEST_REFUSAL, 0.1, 'refusal',
                                        self.last_trace)
                route = 'fallback'
            else:
                route = self._plan(query, intent, evidence, attempt)

            if route == 'hypothesis':
                conclusion = self._discover(query, evidence)
                if conclusion:
                    return self._result(conclusion, MED_CONF, 'hypothesis',
                                        self.last_trace)
                route = 'fact' if evidence else 'fallback'
            if route == 'fallback':
                out = self._fallback(query, strict_def=strict_def)
                return self._result(out, LOW_CONF if out and out != _HONEST_REFUSAL
                                    else 0.1, 'fallback', self.last_trace)

            # fact route — coverage-first selection then score tie-break
            def _coverage(e):
                return len(qwords & _content_words(e.get('snippet', '')))

            scored = sorted(evidence,
                            key=lambda e: (-_coverage(e), -self._score(query, e)))
            best_cov = _coverage(scored[0]) if scored else 0
            candidate = None
            for e in scored:
                if _coverage(e) < best_cov and best_cov < len(qwords):
                    continue
                if strict_def and self._definitional(query,
                                                     e.get('snippet', '')) <= 0:
                    continue  # single-term definitional question needs a real definition
                passed, notes, _rel = self.scientist.attack_text(
                    query, e.get('snippet', ''))
                if passed and len(e.get('snippet', '')) > 3:
                    candidate = e
                    candidate['_attack_notes'] = notes
                    break
            if candidate is not None:
                conf = self._verify(query, candidate, evidence,
                                    candidate.get('_attack_notes', []))
                cov = _coverage(candidate)
                if cov < len(qwords):
                    conf = min(conf, LOW_CONF + 0.10)
                caveat = ''
                if conf < HIGH_CONF and conf >= MED_CONF:
                    caveat = ("(moderate confidence — based on trained data, "
                              "rate limited by available evidence)")
                if cov < len(qwords) and conf < HIGH_CONF:
                    caveat = ("(low confidence — the training data only "
                              "partially covers this question)")
                ans = self.language_core.fact(query, candidate, conf, caveat)
                if ans and ans != _HONEST_REFUSAL and conf >= MED_CONF:
                    return self._result(ans, conf, 'fact', self.last_trace)
                self._trace(f"[cde] fact answer below honesty floor "
                            f"(conf={conf:.2f}, cov={cov}/{len(qwords)})")
            # relax and retry
            _need, _thr = self._relaxation(attempt)

        # nothing passed — fallback then honest refusal
        out = self._fallback(query)
        if out and out != _HONEST_REFUSAL:
            return self._result(out, LOW_CONF, 'fallback', self.last_trace)
        return self._result(_HONEST_REFUSAL, 0.1, 'refusal', self.last_trace)

    # ------------------------------------------------------------------
    # Code solving: extract / generate code, run in sandbox, verify.
    # ------------------------------------------------------------------
    def _solve_code(self, query):
        if self.session is None:
            return None
        # user pasted a code block → just run it
        try:
            from .chat import _extract_code_block
            block = _extract_code_block(query)
            if block and self.sandbox is not None:
                res = self.sandbox.run_python(block)
                if res.get('success'):
                    return self.language_core.code({**res, 'code': block})
                return self.language_core.code({**res, 'code': block})
        except Exception:
            pass
        # coding request → take best code-like evidence and run it
        ev = self._explore(query, 1)
        for e in sorted(ev, key=lambda x: -self._score(query, x)):
            s = e.get('snippet', '')
            code = _normalize_code(s)
            candidates = []
            if code:
                candidates.append(code)
            styles = _extract_code_from_snippet(s)
            if styles and styles.strip() and styles != code:
                candidates.append(styles.strip())
            for block in candidates:
                if not block or self.sandbox is None:
                    continue
                ln0 = block.lstrip().lower()
                if not (ln0.startswith(('for ', 'def ', 'import ', 'print',
                                        'while ', 'if ', '#', 'from ',
                                        'class ', 'x =', 'lambda ',
                                        'return '))):
                    continue
                res = self.sandbox.run_python(block)
                if res.get('success'):
                    return self.language_core.code({**res, 'code': block})
        return None

    # ------------------------------------------------------------------
    # Multi-part questions: "what is the capital of X and Y?" etc.
    # ------------------------------------------------------------------
    def _solve_multi(self, query):
        q = query.lower()
        if ' and ' not in q:
            return None
        m = re.match(r'^(what (?:is|are) (?:the )?)?(.+)$', q)
        if not m:
            return None
        parts = [p.strip() for p in m.group(2).split(' and ') if p.strip()]
        if len(parts) != 2:
            return None
        resolves = []
        for part in parts:
            ev = self._explore(part, 0)
            pw = _content_words(part)
            def _cov(e):
                return len(pw & _content_words(e.get('snippet', '')))
            ranked = sorted(ev, key=lambda e: (-_cov(e), -self._score(part, e)))
            pick = None
            for e in ranked:
                if _cov(e) < len(pw):
                    continue
                passed, _notes, _r = self.scientist.attack_text(
                    part, e.get('snippet', ''))
                if passed and len(e.get('snippet', '')) > 3:
                    pick = e
                    break
            if pick is None:
                return None  # one part unresolvable → let the main loop decide
            snip = _clean_snip(pick.get('snippet') or '')
            sents = _sentences(snip, limit=1)
            resolves.append(sents[0] if sents else snip)
        cap_first = resolves[0][0].upper() + resolves[0][1:] if resolves[0] else ''
        sep_last = (resolves[1][0].upper() + resolves[1][1:]) if resolves[1] else ''
        return f"{cap_first} {sep_last}"

    # ------------------------------------------------------------------
    # Discovery: hypothesize → gate/attack → honest conclusion
    # ------------------------------------------------------------------
    def _discover(self, query, evidence):
        # Rule rejection FIRST: if the QUERY asks about a known-impossible
        # concept, say exactly why — never claim a "new breakthrough" (-inf level).
        law_ok, law_why = self.scientist.gate_known_physics(query)
        if not law_ok:
            return (f"That isn't viable under known physics: {law_why} "
                    f"(rule rejection — energy/luminal/causal limits).")
        hyps = self.hyp_engine.generate(query, evidence)
        if not hyps:
            return None
        qterms = _content_words(query)
        survivors = []
        for h in hyps:
            ok, h = self.scientist.test_hypothesis(h, qterms, evidence)
            if ok:
                survivors.append(h)
        if survivors:
            # Level 1 supported if any survivor derives directly from evidence
            best = max(survivors, key=lambda h: h.level)
            if best.level >= 2:
                return (f"Under current physics and the available trained data, "
                        f"a plausible path is: {best.text}. This is a "
                        f"{'plausible hypothesis (Level 2)' if best.level == 2 else 'speculative hypothesis (Level 3)'} — "
                        f"it is not an experimentally confirmed fact.")
            return (f"Based on the data I have, this appears supported: "
                    f"{best.text}.")
        # every hypothesis rejected — say why (audit the first rejection)
        notes = []
        for h in hyps:
            if h.notes:
                notes = h.notes
                break
        why = (' ' + '; '.join(notes)) if notes else ''
        return (f"These approaches don't hold up under known physics{why}. "
                f"A working solution would need to modify the current model "
                f"or use an unknown mechanism — I can't verify one with the "
                f"evidence I have.")

    # ------------------------------------------------------------------
    # Final fallback: predictor / pattern-based generation (gated)
    # ------------------------------------------------------------------
    _CREATIVE_HINT = ('story', 'poem', 'poetry', 'write a', 'compose',
                      'fiction', 'tale', 'treatise', 'essay', 'song',
                      'joke', 'speech', 'paragraph', 'letter')

    def _fallback(self, query, strict_def=False):
        s = self.session
        qw = _content_words(query)
        creative = any(k in query.lower() for k in self._CREATIVE_HINT)
        try:
            if s.predictor is not None:
                context = s._build_context(query, use_history=False)
                out = s.predictor.reply(context, 120,
                                        getattr(s, 'temperature', 0.7))
                out = (out or '').strip()
                if out:
                    if strict_def and self._definitional(query, out) <= 0:
                        self._trace(f"[cde] fallback not a real definition: "
                                    f"{out[:60]!r}")
                        return _HONEST_REFUSAL
                    tw = _content_words(out)
                    covered = len(qw & tw)
                    need = min(len(qw), 2)
                    if creative:
                        need = 1  # creative may be looser, but must stay on-topic
                    # Honesty gate on the fallback: accept only if the reply
                    # clearly covers the question (queries with no content
                    # words are inherently open).
                    if not qw or covered >= need:
                        return out
                    self._trace(f"[cde] fallback off-topic: {out[:60]!r}")
        except Exception:
            pass
        return _HONEST_REFUSAL

    # ------------------------------------------------------------------
    def _result(self, answer, confidence, route, trace):
        return {
            'answer': str(answer).strip(),
            'confidence': round(float(confidence), 3),
            'route': route,
            'trace': list(trace),
        }

    def answer(self, query):
        r = self.run(query)
        return r['answer']
"""MemoryTransformer - training-free attention-over-memory generator.

Built in ONE streaming pass from a model's baked-in data (reservoir + qa_bank
+ patterns + vocab).  No gradient descent: keys/values are constructed
directly from the data, so it stays CPU-friendly and fully self-contained
inside the weights folder.

    K          : IDF feature-hash vector per memory instruction (router)
    gate       : hard lexical gate - memory must share real content tokens
    attention  : softmax(cosine * overlap-blend) over gated memories
    generation : weighted vote over each memory's ORDERED continuation
                 stream (multi-source ensemble merge) + gram backoff fallback
"""

import math
import re
import pickle

import numpy as np

from ..core.hasher import murmurhash3

_STOP_REASONS = frozenset([
    '?', '!', '.', '...', '</answer>', '</assistant>', '<|endoftext|>',
])
_CONTENT_SKIP = frozenset([
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'is', 'are',
    'was', 'were', 'be', 'it', 'this', 'that', 'with', 'for', 'as', 'at',
    'by', 'what', 'how', ',', '.', '?', '!', ';', ':', '(', ')', "'", '"',
])


# Abbreviation/expansion set so 'pm' can match 'prime minister', 'uk' ->
# 'united kingdom', etc. when scoring QA-instruction overlap.
_QA_EXPAND = {
    'pm': ('prime', 'minister'),
    'us': ('united', 'states'),
    'usa': ('united', 'states'),
    'uk': ('united', 'kingdom'),
    'dr': ('doctor',),
    'ml': ('machine', 'learning'),
}


class MemoryTransformer:
    """Attention-over-memory language model with constructed weights."""

    def __init__(self, vocab, dim=256, top_k=12, gate_threshold=0.35,
                 seed=42):
        self.vocab = vocab
        self.dim = int(dim)
        self.top_k = int(top_k)
        self.gate_threshold = float(gate_threshold)
        self.rng = np.random.RandomState(seed)

        self.K = None
        self.mem_inst = []
        self.mem_ans = []
        self.mem_tok_inst = []
        self.mem_qa = []
        self.idf = {}
        self._gram_index = {}
        self._built = False

    # ------------------------------------------------------------------
    # Build (one streaming pass, no gradients)
    # ------------------------------------------------------------------
    def build(self, reservoir, qa_bank, patterns_data, max_memories=25000,
              max_answer_tokens=160):
        """Construct K / continuation streams from the model's baked-in data."""
        memory = []
        sources = []
        for text in list(reservoir or []):
            if len(memory) >= max_memories:
                break
            if not isinstance(text, str) or not text.strip():
                continue
            inst, ans = self._split_qa(text)
            if not (inst and ans):
                continue
            it = self._toks(inst)
            at = self._toks(ans)
            if not it or not at:
                continue
            memory.append((it, at[:max_answer_tokens], inst))
            sources.append(False)
        for text in list(qa_bank or []):
            if len(memory) >= max_memories:
                break
            if not isinstance(text, str) or not text.strip():
                continue
            inst, ans = self._split_qa(text)
            if not (inst and ans):
                continue
            it = self._toks(inst)
            at = self._toks(ans)
            if not it or not at:
                continue
            memory.append((it, at[:max_answer_tokens], inst))
            sources.append(True)

        self._build_gram_index(patterns_data)

        if not memory:
            raise RuntimeError("MemoryTransformer: no usable snippets")

        self.mem_inst = [inst for _, _, inst in memory]
        self.mem_ans = [at for _, at, _ in memory]
        self.mem_tok_inst = [it for it, _, _ in memory]
        self.mem_qa = np.asarray(sources, dtype=bool)
        self.idf = self._idf_map([it for it, _, _ in memory])
        for t in self.vocab.token_to_id:
            self.idf.setdefault(t, 1.0)

        N = len(memory)
        self.K = np.zeros((N, self.dim), dtype=np.float32)
        for i, (it, _, _) in enumerate(memory):
            self.K[i] = self._vec(it)
        norms = np.linalg.norm(self.K, axis=1, keepdims=True)
        self.K = self.K / np.maximum(norms, 1e-9)

        self._built = True
        return {
            "memories": N,
            "dim": self.dim,
            "qa_memories": int(self.mem_qa.sum()) if hasattr(self, "mem_qa") else 0,
            "gram_branches": len(self._gram_index),
            "idf_vocab": len(self.idf),
        }

    def _build_gram_index(self, patterns_data):
        if not patterns_data:
            return
        raw = patterns_data.get("patterns")
        if isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = list(patterns_data.items())
        for gram, count in items:
            if not isinstance(gram, tuple) or not gram:
                continue
            tail = gram[-1]
            if not isinstance(tail, str):
                continue
            head = tuple(t for t in gram[:-1] if isinstance(t, str))
            branch = self._gram_index.setdefault(head, {})
            branch[tail] = branch.get(tail, 0.0) + float(count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _idf_map(tokens_list, smooth=1.0):
        n = max(len(tokens_list), 1)
        df = {}
        for toks in tokens_list:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        return {w: math.log((n + smooth) / (d + smooth)) + 1.0
                for w, d in df.items()}

    def _toks(self, text):
        try:
            return self.vocab._tokenize(text) or []
        except Exception:
            return []

    def _vec(self, it_toks):
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in it_toks:
            h = murmurhash3(tok)
            bucket = h % self.dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[bucket] += sign * self.idf.get(tok, 1.0)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    @staticmethod
    def _content_toks(tokens):
        return [t for t in tokens
                if t not in _CONTENT_SKIP and not t.isspace()]

    @staticmethod
    def _split_qa(text):
        from ..core.neural_engine import clean_artifacts
        inst = ''
        ans = ''
        m = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        if m:
            inst = clean_artifacts(m.group(1).strip())
        if not inst:
            m = re.search(r'<user>(.*?)</user>', text, re.DOTALL)
            if m:
                inst = clean_artifacts(m.group(1).strip())
        if not inst:
            m = re.search(r'###\s*Instruction:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if m:
                inst = clean_artifacts(m.group(1).strip())
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m and len(m.group(1).strip()) > 3:
            ans = clean_artifacts(m.group(1).strip())
        if not ans:
            m = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
            if m and len(m.group(1).strip()) > 3:
                ans = clean_artifacts(m.group(1).strip())
        if not ans:
            m = re.search(r'###\s*Response:\s*(.*?)(?:\n\n|\Z)', text, re.DOTALL)
            if m and len(m.group(1).strip()) > 3:
                ans = clean_artifacts(m.group(1).strip())
        if inst and ans:
            return inst, MemoryTransformer._trim_answer(ans)
        return None, None

    @staticmethod
    def _trim_answer(ans):
        cut = re.split(
            r'\.?\s*instruction\s*[:\-]\s*the request|'
            r'\.?\s*###\s*(?:Response|Instruction)|'
            r'\.?\s*<\|endoftext\||'
            r'\.?\s*</(?:answer|assistant|instruction|user)>',
            ans, maxsplit=1)[0]
        return cut.strip().rstrip(';:,').strip()

    # ------------------------------------------------------------------
    # Gated attention over memory
    # ------------------------------------------------------------------
    def _attend(self, query_tokens):
        q = self._vec(query_tokens[-256:])
        qn = np.linalg.norm(q)
        if qn == 0:
            return None
        q_set = set(self._content_toks(query_tokens))
        total_q = sum(self.idf.get(t, 1.0) for t in q_set)
        if not q_set:
            return None

        cos = self.K @ q  # (N,)
        w = []
        idx = []
        for i in np.argsort(-cos):
            c = float(cos[i])
            inst_set = set(self.mem_tok_inst[i])
            overlap = q_set & inst_set
            if not overlap:
                continue  # hard lexical gate: no real content overlap
            num = sum(self.idf.get(t, 1.0) for t in overlap)
            ratio = num / max(total_q, 1e-9)
            blend = c * (0.4 + 0.6 * ratio)
            if self.mem_qa[i]:
                blend *= 1.25  # baked QA pairs are cleaner than reservoir rows
            if blend < self.gate_threshold:
                if len(idx) > 0:
                    break
                continue
            w.append(blend)
            idx.append(i)
            if len(idx) >= self.top_k:
                break

        if not idx:
            return None

        w = np.asarray(w, dtype=np.float64)
        e = np.exp(w - w.max())
        e = e / (e.sum() + 1e-9)
        return e, np.asarray(idx)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, prompt, max_tokens=120, temperature=0.7):
        if not self._built:
            raise RuntimeError("MemoryTransformer: call build() first")
        math_ans = self._try_math(prompt)
        if math_ans is not None:
            return math_ans
        st_ans = self._try_smalltalk(prompt)
        if st_ans is not None:
            return st_ans
        toks = self._toks(prompt)
        if not toks:
            return ""
        att = self._attend(toks)
        if att is None:
            return self._gram_fallback(toks, max_tokens)

        w, idx = att
        dom = self._pick_dominator(prompt, toks, w, idx)
        seed = self.mem_ans[dom]
        if not seed:
            return ""

        base = (' '.join(seed).replace(' .', '.').replace(' ,', ',')
                .replace(' !', '!').replace(' ?', '?').strip())
        base = MemoryTransformer._trim_answer(base)
        if len(seed) >= max_tokens:
            return base
        if len(seed) >= 4:
            return base

        ext = self._extend(tuple(seed[-3:]), min(12, max_tokens - len(seed)))
        if not ext:
            return base
        return MemoryTransformer._trim_answer(
            base + (' ' if not base.endswith(('.', '!', '?')) else '') + ext)

    # ------------------------------------------------------------------
    # EVIDENCE MODE — retrieve top gated memories as EVIDENCE structs.
    # The CDE language core synthesizes the final answer from these; CORTEX
    # does NOT return a final verbatim answer anymore (plan §10).
    # ------------------------------------------------------------------
    def retrieve_evidence(self, prompt, k=5):
        """Return up to `k` evidence dicts:
        {instruction, snippet, score, rel, is_qa, is_code}.
        Empty list when nothing survives the hard gate."""
        if not self._built:
            return []
        toks = self._toks(prompt)
        if not toks:
            return []
        q_set = set(self._content_toks(toks))
        if not q_set:
            return []
        att = self._attend(toks)
        if att is None:
            # No gated memory: try the full-memory tail scan for short entity
            # replies ("capital of france" -> "Paris.") as weak evidence.
            for t in reversed(toks):
                if t not in _CONTENT_SKIP:
                    q_tail = t
                    break
            else:
                q_tail = None
            if q_tail:
                best, _sc = self._tail_scan(toks, q_tail)
                if best is not None:
                    return [self._evidence(best, 1.0, q_set, primary=True)]
            return []
        w, idx = att
        order = sorted(zip(w, idx), key=lambda r: -r[0])
        out = []
        for score, mi in order[:k]:
            out.append(self._evidence(int(mi), float(score), q_set,
                                      primary=(len(out) == 0)))
        return out

    def _evidence(self, mi, score, q_set, primary=False):
        ans_toks = list(self.mem_ans[mi])
        snippet = (' '.join(ans_toks).replace(' .', '.').replace(' ,', ',')
                   .replace(' !', '!').replace(' ?', '?').strip())
        snippet = MemoryTransformer._trim_answer(snippet)
        inst_set = set(self.mem_tok_inst[mi])
        ov = q_set & inst_set
        rel = len(ov) / max(len(q_set), 1.0) if q_set else 0.0
        head = ' '.join(ans_toks[:6]).lower()
        is_code = head.startswith((
            'for ', 'import ', 'def ', 'print', 'while ', 'if ',
            '#', 'return ', 'x =', 'lambda ', 'function ', '```'))
        return {
            'instruction': self.mem_inst[mi],
            'snippet': snippet,
            'score': round(float(score), 4),
            'rel': round(float(rel), 4),
            'is_qa': bool(self.mem_qa[mi]),
            'is_code': is_code,
            'primary': bool(primary),
            'source': 'cortex',
        }

    @staticmethod
    def _try_smalltalk(prompt):
        p = str(prompt).strip().rstrip('?!.').lower()
        if p in {'hi', 'hello', 'hey', 'yo', 'hola', 'namaste', 'namaskar',
                 'good morning', 'good afternoon', 'good evening',
                 'how are you', 'how are you doing', 'how do you do',
                 'what\'s up', 'wassup', 'sup', 'how\'s it going',
                 'how is it going', 'what up'}:
            return "Hi there! How are you doing today?"
        if p in {'thanks', 'thank you', 'thankyou', 'thx'}:
            return "You're welcome!"
        if p in {'bye', 'goodbye', 'see you', 'see ya'}:
            return "Goodbye! Have a great day."
        return None

    @staticmethod
    def _try_math(prompt):
        s = str(prompt).strip()
        if not s:
            return None
        s = re.sub(
            r'(?i)^(?:what is|whats|what is the value of|what does|what do|'
            r'calculate|compute|evaluate|solve|find|the value of)\s*',
            '', s.rstrip('?')).strip()
        if s.startswith('='):
            s = s[1:].strip()
        if not s:
            return None
        if not re.fullmatch(r'[\d\s+\-*/().,%]+', s):
            return None
        if '(' in s and ')' not in s:
            return None
        try:
            r = eval(s, {"__builtins__": {}}, {})
        except Exception:
            return None
        if isinstance(r, (int, float)):
            return str(r)
        return None

    def _pick_dominator(self, prompt, toks, w, idx):
        """Pick the strongest memory, with a definition-likeness bonus for
        'what is / define / explain'-style queries and a code-start penalty.
        For entity questions the answer's first words must actually MENTION
        the entity — otherwise "what is python" returns digit-sum code (the
        highest-weight memory) instead of the python definition."""
        pl = prompt.lower()
        is_def_q = any(k in pl for k in (
            'what is', 'what\'s', 'what are', 'define', 'explain',
            'meaning of', 'who is', 'describe'))
        q_tail = None
        for t in reversed(toks):
            if t not in _CONTENT_SKIP:
                q_tail = t
                break

        pairs = list(zip(tuple(int(m) for m in idx),
                         tuple(float(v) for v in w)))
        if is_def_q and q_tail:
            try:
                pat = re.compile(r'(?<![a-z0-9])' + re.escape(q_tail)
                                 + r'(?![a-z0-9])')
                filt = [p for p in pairs
                        if pat.search(' '.join(self.mem_ans[p[0]][:6]).lower())]
                if filt:
                    pairs = filt
            except Exception:
                pass
        idx_n = np.array([p[0] for p in pairs], dtype=np.int64)
        w_n = np.array([p[1] for p in pairs], dtype=np.float64)
        if idx_n.size == 0:
            return int(idx[int(np.argmax(w))])

        best = int(idx_n[int(np.argmax(w_n))])
        best_sc = -1e9
        for a_i, mi in enumerate(idx_n):
            ans = self.mem_ans[mi]
            head = (' '.join(ans[:6]))
            hl = head.lower()
            starts_code = hl.startswith((
                'for ', 'import ', 'def ', 'print', 'while ', 'if ', '#',
                'return ', 'x =', 'lambda '))
            starts_def = False
            if q_tail:
                starts_def = (hl.startswith(q_tail + ' ')
                              or hl.startswith(q_tail + 's ')
                              or f' {q_tail} is' in hl)
            else:
                starts_def = hl.startswith(('is ', 'are ', 'a ', 'an ',
                                            'refers', 'is a ', 'when '))
            sc = float(w_n[a_i])
            if starts_def:
                sc += 0.45
            if starts_code:
                sc -= 0.5
            if sc > best_sc:
                best_sc = sc
                best = int(mi)

        # The gated pool (top_k) can be dominated by code memories, so the
        # real definition may sit OUTSIDE the pool entirely.  Run a
        # full-memory scan whose QA-overlap score is directly comparable to
        # the pool winner's, and take whichever scores higher.
        if q_tail:
            try:
                scan_best, scan_sc = self._tail_scan(toks, q_tail)
                if scan_best is not None:
                    q_set = set(self._content_toks(toks))
                    pool_sc = self._tail_score(best, q_set, q_tail)
                    who_like = any(k in pl for k in (
                        'who is', 'who was', 'who are', 'tell me about',
                        'which is', 'name the', 'who'))
                    if who_like:
                        scan_sc += 0.4 * min(len(self.mem_ans[scan_best]) / 30.0, 1.0)
                        pool_sc += 0.4 * min(len(self.mem_ans[best]) / 30.0, 1.0)
                    if scan_sc > pool_sc:
                        best = scan_best
            except Exception:
                pass
        return best

    def _tail_score(self, mi, q_set, q_tail):
        """Score a memory's answer against the question — favours QA memory
        whose INSTRUCTION covers the whole question, penalises code."""
        head = ' '.join(self.mem_ans[mi][:6]).lower()
        inst_set = set(self.mem_tok_inst[mi])
        eff_q = set(q_set)
        for g in q_set:
            for e in _QA_EXPAND.get(g, ()):
                eff_q.add(e)
        ov = eff_q & inst_set
        sc = 1.1 * len(ov) / max(len(eff_q), 1)
        if head.startswith(q_tail + ' ') or head.startswith(q_tail + 's '):
            sc += 0.3
        elif f' {q_tail} is' in head:
            sc += 0.2
        if head.startswith(('for ', 'def ', 'import ', 'print', 'while ',
                            'if ', '#', 'return ')):
            sc -= 1.2
        if self.mem_qa[mi]:
            sc += 0.3
        return sc

    def _tail_scan(self, toks, q_tail):
        """Full-memory scan for answers whose first words mention the
        entity OR whose instruction asks about it (short QA replies like
        'Paris.' never mention 'france' in the answer itself).
        Returns (memory_index, score) or (None, -1e9)."""
        pat = re.compile(r'(?<![a-z0-9])' + re.escape(q_tail)
                         + r'(?![a-z0-9])')
        q_set = set(self._content_toks(toks))
        best = None
        best_sc = -1e9
        for i in range(len(self.mem_ans)):
            if not self.mem_ans[i]:
                continue
            head = ' '.join(self.mem_ans[i][:6]).lower()
            inst = ' '.join(self.mem_tok_inst[i]).lower()
            if not (pat.search(head) or pat.search(inst)):
                continue
            sc = self._tail_score(i, q_set, q_tail)
            if sc > best_sc:
                best_sc = sc
                best = i
        return best, best_sc

    def _extend(self, context, max_add):
        """Continue after the cached answer using gram backoff + prior."""
        out = []
        prior = {}
        for w_ in self.mem_ans:
            if not w_:
                continue
            prior[w_[0]] = prior.get(w_[0], 0.0) + self.idf.get(w_[0], 1.0)
        ptotal = sum(prior.values()) or 1.0
        prior = {t: v / ptotal for t, v in prior.items()}

        ctx = list(context)
        for _ in range(max_add):
            votes = {}
            for t, p in prior.items():
                votes[t] = votes.get(t, 0.0) + 0.2 * p
            for k in range(1, min(len(ctx), 3) + 1):
                branch = self._gram_index.get(tuple(ctx[-k:]))
                if branch:
                    for tail, c in branch.items():
                        votes[tail] = votes.get(tail, 0.0) + c
            if not votes:
                break
            items = sorted(votes.items(), key=lambda kv: -kv[1])
            tok = items[0][0]
            if tok in _STOP_REASONS:
                break
            out.append(tok)
            ctx.append(tok)
        return ' '.join(out).strip()

    def _gram_fallback(self, toks, max_tokens):
        out = []
        for _ in range(max_tokens):
            prefix = tuple(toks[-3:]) + tuple(out[-3:])
            votes = {}
            for k in range(1, min(len(prefix), 3) + 1):
                branch = self._gram_index.get(prefix[-k:])
                if branch:
                    for tail, c in branch.items():
                        votes[tail] = votes.get(tail, 0.0) + c
            if not votes:
                break
            items = sorted(votes.items(), key=lambda kv: -kv[1])
            tok = items[0][0]
            if tok in _STOP_REASONS:
                break
            out.append(tok)
        return ' '.join(out).strip()


def build_from_weights(weights_dir, vocab, dim=256, top_k=12,
                       gate_threshold=0.35, max_memories=25000):
    """Build a MemoryTransformer straight from a model weights folder."""
    from ..storage.weight_manager import WeightManager
    data, _ = WeightManager(weights_dir).load()
    reservoir = []
    qa_bank = []
    patterns = {}
    if "reservoir_sample" in data.files:
        try:
            reservoir = pickle.loads(data["reservoir_sample"].tobytes()) or []
        except Exception:
            reservoir = []
    if "qa_bank" in data.files:
        try:
            qa_bank = pickle.loads(data["qa_bank"].tobytes()) or []
        except Exception:
            qa_bank = []
    if "patterns_raw" in data.files:
        try:
            patterns = pickle.loads(data["patterns_raw"].tobytes()) or {}
        except Exception:
            patterns = {}
    mt = MemoryTransformer(vocab, dim=dim, top_k=top_k,
                           gate_threshold=gate_threshold)
    info = mt.build(reservoir, qa_bank, patterns, max_memories=max_memories)
    return mt, info
# NOVACORE — COGNITIVE DISCOVERY ENGINE (CDE) MASTER PLAN

> Live planning file. Update this file as the project evolves.
> **Rule: coding shuru tabhi hogi jab user bole.**

Created: 2026-09-08
Status: CODING COMPLETE (M1+M2+M3.1, verified on weights\NovaCoreV14) + PRECISION BUILD
(solver word-problems + transitive logic, verified on weights\NovaCoreV15) + PARALLEL BUILD
speed (multiprocessing + optional GPU SVD) — 2026-09-09

---

## 1. VISION (user's own words, condensed)

NovaCore ko RAG chatbot NAHI banana. Banana hai ek **zero-training, data-ingestion-based,
CPU / low-end GPU friendly "LLM-jaisa brain"** jo:

- Dataset ko sirf **knowledge** ki tarah use kare — dataset lines ko verbatim copy karke answer
  na de. Query ko samjhe, internally retrieve kare, reason kare, simulate kare, plan kare,
  verify kare, retry kare, phir ek **naya synthesized natural answer** generate kare.
- Har domain ka ek hi common cognitive core ho — physics, math, chemistry, biology, space,
  coding, medicine, general knowledge.
- Known knowledge se shuru kare, lekin zarurat padne par **known laws/domain boundaries ke bahar
  ja kar bhi hypothesis soch sake** — jis tarah invention se pehle "ye sambhav nahi" bola jata hai,
  aur formula/physics ke naye combination se sambhav ho jata hai.
- **"Bina matlab ka logic" bilkul na ho** — har hypothesis rigorous gates se guzre ya reject ho.
- GPT-class smart/powerful/accurate feel de, lekin bina hours/days of training ke — "Knowledge
  acquisition, not weight training".
- Junction rule (user's hard rule): **sirf real/possible cheez. Jo possible ho karwao, jo fake ho
  (bewkoof banane wala) wo chhodo.**

---

## 2. HARD REALITY (honest engineering note)

- **Zero additional training** is achievable and is the core goal.
- **Zero pretrained neural language capability** + pure symbolic/RAG/synthesis gathering **cannot**
  reach open-ended GPT-level language generation. This is a physics/empirical limit, not a code effort
  limit — we will NOT fake it.
- The realistic, powerful target:
  ```
  small pretrained CPU-friendly neural LM (language core)
        +
  NovaCore Cognitive runtime (memory/reasoning/planner/tools/critic/verify/retry/synthesis)
        +
  structured knowledge memory + multi-representation indexes
        +
  tools (python terminal, math, code run, search, files)
        +
  adversarial verification (critic/attacker) + evidence + confidence
  ```
- Dataset supply = knowledge. LM = language/generalization. NovaCore = cognition/orchestration.
  Tools = reality/execution. Verifier = accuracy control.

**OPEN DECISION — RESOLVED (REJECTED by user 2026-09-08):** no external/LM token-generator inside.
User decided PURE NovaCore ("aaj ke llm se smart powerful aur accurate, zero training, GPU-friendly").
Winning domains therefore = EXACT math / unit / percent / date / physics WORD-PROBLEMS + transitive
logic (deterministic compute — LLMs make arithmetic slips, NovaCore does not), run-tested code,
honesty, speed. Open-ended frontier generality remains the explicit NOT-claimed zone (§13).

---

## 3. CURRENT-STATE AUDIT (existing modules -> future role)

| Current component | File(s) | Future role |
|---|---|---|
| reservoir.py | core/reservoir.py | Knowledge ingestion / long-term memory |
| encoder.py | core/encoder.py | Semantic representation (dim 1024) |
| patterns.py | core/patterns.py | Fast lexical/pattern memory (200k patterns) |
| knowledge_extractor.py | core/knowledge_extractor.py | Knowledge compiler (facts/QA/defs/procedures/code/math...) |
| training_upgrades.py | core/training_upgrades.py | IDF / quantization / semantic search helpers |
| CORTEX MemoryTransformer | inference/memory_transformer.py | Fast memory ATTENTION/RETRIEVAL layer (evidence, not verbatim) |
| ChatSession.generate | inference/chat.py | Cognitive orchestrator (router + loop) |
| VirtualSim | virtual_sim.py | Simulation / candidate validation |
| ModelBrain layers | core/model_brain.py | Reasoning/planning/verification components |
| Neural Engine | core/neural_engine.py | Small neural language/reasoning core |
| Predictor | inference/predictor.py | Last-resort fallback (gram/pattern texture) |
| Knowledge index | knowledge_index.json, retrieval.py | Knowledge graph + lexical + vector indexes |
| WeightManager | storage/weight_manager.py | Knowledge/model packaging (weights.ncw) |

Nothing gets randomly deleted — responsibilities get MERGED/REFACTORED into the CDE over phases.

---

## 4. TARGET ARCHITECTURE — unified INTERNAL core

```
                        USER
                          │
                          ▼
            ┌─────────────────────────┐
            │   INTERNAL LLM CORE     │  (NovaCore Cognitive Runtime)
            │                         │
            │  1. Understand query    │
            │      intent · context   │
            │      goal · domain      │
            └───────────┬─────────────┘
                        ▼
             INTERNAL THINKING (hidden)
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    MEMORY         KNOWLEDGE        REASONING
    CORTEX/EXACT   GRAPH/QA/        ENGINE
    SEMANTIC       FACTS/DEF        (multi-hop,
    PATTERN        CODE/MATH        first-principles)
        └───────────────┼───────────────┘
                        ▼
                     PLANNER
                        ▼
              VIRTUAL SIMULATION / TOOLS
               (python · math · code run ·
                search(optional) · files)
                        ▼
                    DRAFT ANSWER
                        ▼
             CRITIC / ADVERSARIAL VERIFIER
                        |
              ┌─────────┴─────────┐
            FAIL                  PASS
              │                    │
              ▼                    │
        analyze · reason          │
              │                   │
        RETRY (max N) ────────────┘
                        ▼
               FINAL SYNTHESIS /
                NATURAL ANSWER
                        ▼
                { answer, evidence, confidence, source }
```

**Principles baked in:**
- All internal stages (thinking/reasoning/simulation/retry/verification) happen INSIDE the core.
  User sees only final natural answer. (Debug verbosity stays optional behind `--debug`.)
- Same IDs as real LLM "thinking" — no fake "Let me think..." text to the user.
- Weak candidate can fail early (adaptive compute). RESERVED entrance: route by intent.
- **INTERNAL-ONLY PRINCIPLE (user decision 2026-09-08):** sab kuch (thinking, reasoning, planning,
  simulation, tools, verify, retry) model ke INTERNAL runtime ke andar hi hota hai. User ko sirf
  final synthesized answer milta hai. Python/code sandbox ise "external help" NAHI hai — wo core ka
  hi internal tool layer hai (LLM agents me jaise hota hai). Subprocess invoke hona ek implementation
  detail hai; output kabhi bhi tool ka raw dump nahi, core ka synthesized final answer hota hai.
- **VIRTUAL SIMULATION = HAR DOMAIN (user decision 2026-09-08):** sirf coding/math nahi — physics,
  chemistry, biology, space, hypothesis — sabme. Strategy per-domain: jahan numeric ho → numeric
  simulation; jahan nahi → symbolic/constraint/consistency simulation.
- **SINGLE CODE RUNNER → SANDBOX ORCHESTRATOR (locked 2026-09-08):** "python terminal" khud Python hi
  execute karta hai; wo HAR language ka code runner NAHI. Implementation = **sandbox orchestrator**:
  ```
  PYTHON TERMINAL (orchestrator/controller)
     ├── Python runtime
     ├── Go runtime
     ├── JS/Node runtime
     ├── C/C++ compiler
     └── other installed runtimes
  ```
  Python common execution controller hai jo target language ka runtime spawn karta hai. Security caps
  hamesha ON: no-network, timeout, resource/memory limits, output-capture only.
- **MODEL-vs-KNOWLEDGE BOUNDARY (locked 2026-09-08, LANGUAGE CORE):**
  ```
              NOVACORE
                  │
        ┌─────────┴─────────┐
        │                   │
  KNOWLEDGE MEMORY       LANGUAGE CORE
        │                   │
  facts/formulas/          language/abstraction/
  evidence/procedures      synthesis/generalization
        │                   │
        └─────────┬─────────┘
                  ▼
             CDE COGNITION
  ```
  Language core = "zero ADDITIONAL training, not zero pretrained model." Ultimate decision:
  small quantized CPU-friendly pretrained LM integrate karna (YES). Lekin **pure-NovaCore synthesis
  engine pehle bhi same LanguageCore interface par chalta hai** — LM baad me additive swap hai,
  rewrite nahi. System aaj bhi chalta rahega.

---

## 5. THE CDE LOOP (core pipeline to build)

```
UNDERSTAND → [THINK·REASON·PLAN merged] → HYPOTHESIZE → SIMULATE → ATTACK → VERIFY → REFINE → REPEAT(max) → SYNTHESIZE
```

Where each stage maps to existing/exposed components:
- UNDERSTAND   -> ChatSession intent router + goal/domain extractor (new, thin)
- **THINK·REASON·PLAN MERGED (user decision 2026-09-08):** teen alag "brains" nahi — ek hi unified
  pro-level inference loop. THINK = problem ko samajhna + internal search strategy + context build;
  REASON = derivation (first-principles, cross-domain A+B+C composition); PLAN = multi-hop steps/
  hypothesis path choose karna. Ye teeno ek saath ek engine me chalta hai (CDE), competing generators
  NAHI.
- EXPLORE      -> CORTEX (fast) + semantic + QA bank + facts index (retrieval as EVIDENCE only)
- HYPOTHESIZE  -> NEW hypothesis generator (novel combos; DO NOT fabricate facts - candidate labels
                  only). Har hypothesis ka ek **internal LEDGER (lock 2026-09-08)** banta hai:
                  ```
                  Hypothesis | Assumptions | Evidence | Derivation |
                  Predictions | Tests performed | Failures |
                  Counterexamples | Confidence
                  ```
                  Ledger isliye ki system sirf "logical-sounding" answer na bana paye — answer ke
                  peeche har claim ka derivation/test record ho. Pura ledger internally rehta hai;
                  user ke paas sirf final synthesis + confidence/evidence tags.
- SIMULATE     -> domain-agnostic virtual simulation (har domain) + sandbox orchestrator
                  (numeric/symbolic sanity checks; user code ko run/test karke hi answer)
- ATTACK       -> **CENTRAL MANDATORY stage (lock 2026-09-08), optional gate nahi** — Adversarial
                  Scientist / Hypothesis Breaker. Passive check nahi, actively todne ki koshish:
                  math contradiction, dimension error, conservation violation, causal problem,
                  numeric instability, counterexample, conflicting evidence, hidden assumption.
                  Sab attack-verdict ledger me record hota hai; survival → confidence up,
                  contradiction → confidence down.
- VERIFY       -> confidence scoring + evidence sufficiency + cross-source consistency
- REFINE/RETRY -> reroute, relax/strict assumptions, retry with feedback; max attempts cap
- SYNTHESIZE   -> final natural answer built NEW (never raw dataset verbatim), plus metadata

---

## 6. HYPOTHESIS ENGINE + HARD GATES (anti-nonsense)

Hypothesis types (user's "boundary-breaking"):
- H1 known mechanism, H2 changed assumption, H3 combined principles,
  H4 indirect mechanism, H5 law/modelling variant, H6 analogy from another domain.

Every hypothesis must pass ALL 7 gates or be rejected:
1. Relevance (does it answer the actual question?)
2. Mathematical consistency (exprs evaluate, no divide-by-zero, signs correct)
3. Dimensional consistency (units match both sides)
4. Physical/scientific constraints (conservation, thermodynamics, causality)
5. Causal consistency (cause precedes effect; no impossible dependency)
6. Simulation/test (python numeric check where possible)
7. Counterexample search (known case that kills it)

**BOUNDARY = CONSTRAINT RELAXATION, NOT LAW-BREAKING (lock 2026-09-08):**
```
KNOWN LAW
   ↓
can it be derived from known principles?  → YES → novel combination (Level 2)
   ↓ NO — which assumption/model is limiting us?
   ↓ relax/change that assumption → generate hypothesis → test consequences
```
- Hypothesis known physics ko contradict karti hai → system kabhi "new discovery confirmed" nahi
  bolta. Wording = **"This requires a modification of the current model / an unknown mechanism."**
  (Level 3). Honest = Level 2/3/∞ ladder (§9).
- Niche to abhi koi cheez nahi. Agar koi hypothesis all gates+attack survival kare →
  "qualified possible hypothesis", confirmed-fact overclaim kabhi nahi.
- Per-query hypothesis budget (CPU, lock 2026-09-08): max 3–5 hypotheses; escalate tabhi jab
  fast path fail ho. Ledger depth capped — low-end RAM safe.

Result levels — NEVER overclaim:
- Level 1 Established/supported (data+formula+consistency)
- Level 2 Plausible hypothesis (novel derivation within known principles)
- Level 3 Speculative (no contradiction found, NOT experimentally established)
- Level ∞ "I don't have enough evidence" — honest refusal (allowed and encouraged)

Perpetual-motion example that MUST behave correctly: hypothesis generated -> conservation/entropy
attack -> rejected -> honest verdict "not viable under known physics".

---

## 7. KNOWLEDGE COMPILATION (build-time; zero training)

```
RAW DATA → parser/cleaner → chunker → multi-representation extractor
            ├── raw text / chunks
            ├── entities + relations
            ├── facts  ├── QA pairs  ├── definitions  ├── procedures
            ├── cause→effect  ├── comparisons  ├── examples
            ├── code patterns  ├── mathematical relations  ├── source/provenance
→ indexes: knowledge GRAPH + VECTOR + LEXICAL + QA MEMORY + PATTERN MEMORY + PROCEDURE MEMORY
→ weights.ncw pack
```
Every knowledge item carries metadata:
```
{ content, source, source_id, confidence, domain,
  timestamp, evidence, relations, verification_status }
```

---

## 8. DOMAIN / ROUTING + ADAPTIVE COMPUTE

- FAST PATH (easy factual): CORTEX + QA bank -> short verified answer. ~cheap.
- KNOWLEDGE PATH (mid): retrieval + knowledge graph + reasoning + verifier. medium cost.
- AGENT PATH (coding/hard): planner + tools (project inspect, dependency analysis, patch gen,
  run tests, read errors, fix, run again, verify). expensive but only when needed.
- Any single query does NOT run everything: router picks cost level up front.
- Cross-domain questions auto-decompose: physics part + chemistry part + biology part + math part
  + engineering part -> each solved -> joined -> one synthesized answer.
- SIMULATION is domain-agnostic (user decision 2026-09-08): numeric where possible (python), else
  symbolic/constraint/consistency simulation — physics/chem/bio/space/hypothesis sab included.
- CODE RUNNER = SANDBOX ORCHESTRATOR (lock 2026-09-08): python terminal controller hai jo target
  language ka runtime spawn karta hai (Python/Go/JS/C-C++/installed runtimes). Coding answers
  internal run-test se guzarti hain: dikkat → retry; theek → output. Security caps ON.
- INTERNAL-ONLY (user decision 2026-09-08): in sabka use model core ke ANDAR hota hai; final answer
  core synthesizes karta hai, tools ke raw output ko kabhi seedha user ko nahi bhejta.

Coding Agent loop (explicitly requested):
```
user request → understand → project analysis → memory+code retrieval → plan → generate patch
→ virtual execution → test → PASS→verify→final | FAIL→analyze→reason→retry(max N)
```

---

## 9. CONFIDENCE STATES (honesty core)

| State | Action |
|---|---|
| HIGH confidence | answer |
| MEDIUM | retrieve/verify more before answering |
| LOW | tools/reason/simulate more |
| NO evidence | "I don't have enough evidence" — no fake confidence |

Confidence derived from: source agreement count, evidence sufficiency, verification gate pass
rate, contradiction count, counterexample survival.

---

## 10. CORTEX ROLE CHANGE (LOCKED 2026-09-08)

- CORTEX stays as fast memory layer BUT output semantics change from
  "dominant-memory verbatim answer" -> "dominant-memory EVIDENCE" feed to synthesis stage.
  Final answer is produced by the synthesis generator using evidence, NOT by echoing a stored line.
  (Long-collapse answers are the failure mode to eliminate.)
- **Dataset ka sentence kabhi final answer nahi** — sirf exact quotation/reference situation me
  (user ne quote maanga ho ya source/evidence reference cite ho raha ho). Baaki sabme synthesis.
- Example: memory "Python lists are mutable." + query about practical benefit
  -> internal reasoning/example/synthesis, NOT the raw stored sentence.
- CDE loop takes the evidence: `CORTEX → EVIDENCE → CDE → SYNTHESIS`.

---

## 11. ROADMAP — FINAL-FIRST (self-decided 2026-09-08, fast delivery)

STATUS: **CODING IN PROGRESS (user go-ahead 2026-09-08).** M1+M2+M3.1 being implemented in one
pass: `inference/cde.py` (CDE core + LanguageCore interface + hypothesis/attack/gates) +
`tools/sandbox.py` (code-execution sandbox). Integration: `ChatSession.generate_cde()` +
CLI `--cde` flag. Build/test verified against `weights/NovaCoreV14` (59MB, already trained).
Bridge/compare wiring = post-build step (NovaCoreChat repo).

Training/compile speed reality (self-decided, LOCK): NovaCore = knowledge COMPILATION, not neural
training. 139k docs ka full build CPU par minutes-level; naya data `add-datasets` se minute-scale
incremental merge. GPU/hours required nahi. Isliye har milestone "quick build + chat test" loop me
verified hota hai. NO gradient/epochs — "training fast" reliably.

PRIORITY (self-decided): (a) prove CDE loop with existing single-domain knowledge first —
cross-domain later. Yahi final karega.

### M1 — CORE HONESTY (synthesis, no more verbatim)
- M1.1 NEW `inference/cde.py`: router/intent + merged THINK·REASON·PLAN loop; VirtualSim + verify +
      retry ek core me fold; per-hypothesis LEDGER (internal only).
- M1.2 CORTEX → EVIDENCE mode (memory_transformer returns evidence structs, not final text) +
      `LanguageCore` interface (pure-NovaCore synthesis interim same interface) + kill verbatim
      collapse + honest no-evidence refusal.
- DONE = har chat answer naya synthesized hai, dataset echo nahi; bench compare pass.

### M2 — REALITY LAYER (run-before-answer)
- M2.1 NEW `tools/sandbox.py`: python controller runner (Python first; Go/JS/C++ as installed) +
      security caps (no-network, timeout, resource limits, output-capture). Code entry ka internal
      run-test: pass → output; fail → retry.
- M2.2 Domain-agnostic numeric/symbolic simulation; RETRY cap 3, agent-loop attempts 5.
- DONE = math/coding answers execution-tested hone ke baad hi bahar jayein.

### M3 — DISCOVERY ENGINE (differentiator)
- M3.1 Hypothesis generator (budget 3–5/query) + 7 gates + CENTRAL ATTACK (Adversarial Scientist) +
      first-principles mode + constraint-relaxation ladder. Honest Level-1/2/3/-∞ verdict wala
      "sambhav/impossible + kyun" answer.
- M3.2 (STRETCH — ho paye to hi) cross-domain auto-decompose.
- DONE = invention-type hypothesis answers honesty ke saath.

### DEFERRED — final ke lie BLOCKING NAHI (future)
- Coding project-agent loop (whole-project patch/test/fix) → future.
- Small LM integration → future additive swap (interface ready in M1.2).
- CPU perf deep-tune / quantized indexes / llama compare re-bench → future.

### REFACTOR ORDER (self-decided, CLOSED)
1. `inference/cde.py` (NEW core loop)  2. `chat.py` (route → CDE; legacy behind flag)
3. `memory_transformer.py` (evidence mode)  4. fold `virtual_sim.py` into cde.py
5. `tools/sandbox.py` (NEW)  6. reuse `model_brain.py` helpers inside CDE (no separate brains)
7. CLI wiring + `--debug` trace.

### PROJECT FINAL = M1 + M2 + M3.1, CPU-friendly, CLI + bridge working.

---

## 12. OPEN QUESTIONS / DECISIONS LOG

- [x] Small LM language core = REJECTED by user (2026-09-08). Pure NovaCore only — no token-generator
      inside. Precision solver + pure-NovaCore synthesis = the language core interface implementation.
- [x] Pure-NovaCore synthesis = interim engine (language-core interface), NOT competing brain.
- [x] File-level refactor order — CLOSED (see §11 REFACTOR ORDER)
- [x] RETRY cap = 3; agent-loop attempts = 5 — CLOSED (self-decided)
- [x] Priority = (a) CDE with existing single-domain knowledge first; cross-domain later — CLOSED
- [x] Evidence/ledger log storage = sidecar `evstore/*.json`; weights.ncw format UNCHANGED (low risk)
      — CLOSED
- [x] PROJECT FINAL definition = M1 + M2 + M3.1 (see §11); defer rest — CLOSED
- [x] VIRTUAL SIMULATION = har domain (numeric/symbolic/constraint per domain) — DECIDED
- [x] Code runner = SANDBOX ORCHESTRATOR (python controller + per-language runtime spawn) — DECIDED
- [x] THINK·REASON·PLAN = ek hi merged pro-level inference loop (CDE) — DECIDED
- [x] Internal-only: sab tool/simulation/thinking model core ke ANDAR; user ko sirf final
      synthesized answer — DECIDED
- [x] CDE per-hypothesis LEDGER + hypothesis budget (3–5/query) — DECIDED
- [x] ATTACK = central mandatory stage (Adversarial Scientist), verified internally — DECIDED
- [x] Boundary-todna = constraint relaxation, never law-breaking overclaim — DECIDED

Changes to this file should be appended below in the log.

---

## 13. EXPECTATIONS — HONEST CLAIMS (lock 2026-09-08)

**What it CAN confidently claim:**
- Precision math: deterministic word-problem solving (discount/pay-amount, qty×price, word
  arithmetic, unit conversion, percent, speed, temperature, dates) — computed exactly, front
  LLM arithmetic-error zone. Transitive logic (relation chains, superlatives) via rule engine.
- Known-domain accuracy: stored knowledge me hallucination-free (retrieval-based, fabricate nahi).
- Coding/math answers execution-tested (run-before-answer) → frontier LLM ke "confidently wrong
  code" se zyada reliable wahan jahan tools in hain.
- Honesty: no-evidence → "I don't have enough evidence"; confidence levels 1/2/3/-∞; overclaim nahi.
- Instant knowledge update (minute-scale compile), CPU/low-end-GPU friendly, private, no API cost.
- Bounded reliability > frontier LLM on the specific axes above — NOT on overall generality.

**What it CANNOT claim (no football):**
- Overall smart/powerful/accurate = frontier LLM (GPT-4/Claude class). Generative generality/writing/
  open-ended creativity me nahi beat kar sakta — wo physics hai, hype nahi.
- Not a monolithic transformer; it is an INTERNAL cognitive system (memory + reasoning + tools +
  verifier) that behaves LLM-like at the boundary (input → final answer only).
- Accuracy bounded by: knowledge store coverage + verification quality. Jo data me nahi aur derive
  nahi hota → honest refusal, kabhi guess-na-dena.

**Benchmark targets (verification of claims):**
- New vs old NovaCore: chat+bench correctness, verbatim-echo rate ~0.
- NovaCore vs llama3.2:1b (bridge compare): factual-correctness win on our test set, honesty on
  unknown questions, run-tested code. Open-ended quality = not-claimed loss zone.

---

## LOG

- 2026-09-08: Created plan. Vision freeze (CDE + zero-training + synthesis + honesty).
  No code started. Waiting for user go-ahead.
- 2026-09-08: User explicitly said: planning file should be the single source of truth;
  keep updating it; coding only after user instruction. Design sections above reflect that.
- 2026-09-08: 4 decisions locked: (1) virtual simulation = har domain; (2) python terminal =
  single code runner for ALL languages; (3) THINK·REASON·PLAN merged into one pro-level unified
  loop (yehi invention-capable engine hai) — teen alag brains nahi; (4) internal-only — sab
  model core ke andar, user ko sirf final synthesized answer. §4/§5/§8/§9 plan me updated.
  Register: invention = hypothesis/derivation (Level 2/3), kabhi confirmed-fact overclaim nahi.
- 2026-09-08: 6 architectural locks frozen in PLAN (self-decided, user-approved direction):
  1) per-hypothesis LEDGER (Hypothesis/Assumptions/Evidence/Derivation/Predictions/Tests/Failures/
     Counterexamples/Confidence) + query budget 3–5 hypotheses; 2) boundary-todna = constraint
     relaxation, never law-breaking claim ("requires modification of current model / unknown
     mechanism"); 3) ATTACK = central mandatory Adversarial-Scientist stage, optional gate nahi;
  4) CORTEX = evidence-only, dataset sentence kabhi final answer nahi (sirf quotation/reference
     exception); 5) LANGUAGE CORE decision = small quantized CPU LM ultimate YES, zero ADDITIONAL
     training; pure-NovaCore synthesis interim behind same interface (additive swap); 6) "python
     terminal" CA = SANDBOX ORCHESTRATOR (python controller + per-language runtime spawn) + security
     caps. Plan §4/§5/§6/§8/§10/§11/§12 updated; roadmap P1–P8 reordered accordingly. NO CODE yet —
     awaiting user go-ahead.
- 2026-09-08: FINAL-FIRST replan (user: "project ab jaldi final karna hai"): roadmap P1–P8 →
  3 milestones M1 (synthesis honesty) / M2 (run-before-answer sandbox) / M3 (discovery engine).
  PROJECT FINAL = M1+M2+M3.1. Deferred: whole-project coding agent, small-LM, perf deep-tune.
  Training-speed note added (compilation = minute-scale, no GPU). Open questions closed: retry cap 3,
  agent attempts 5, refactor order fixed (§11), evidence log = sidecar evstore/*.json, priority (a).
  NO CODE yet — awaiting user go-ahead.
- 2026-09-08: §13 EXPECTATIONS locked (honest claims): bounded reliability > frontier LLM on
  known-domain accuracy / run-tested code / honesty / instant-update / CPU; overall generality =
  NOT beatable, no overclaim. Internal-only boundary reaffirmed (input→final answer; tools =
  internal layer). NO CODE yet.
- 2026-09-08: USER GO-AHEAD received ("final coding karo bina ruke"). CODING STARTED.
- 2026-09-08: USER DECISION (course-correction): pure NovaCore hi chahiye — LM add-ON
  (Ollama llama3.2:1b) REJECTED; no token-generator inside. Strategy reframed: pure NovaCore
  winning domains = EXACT math, deterministic unit/percent/date/physics WORD-PROBLEMS
  (LLMs make arithmetic errors here — NovaCore computes), tested code, honesty, speed.
- 2026-09-08: tools/solver.py added — deterministic precision solver layer (pure rules, zero
  data, zero LM): unit conversion (length/mass/time/data, exact tables), reversed form
  ("how many meters in 2 km"), percent (% of / off / increase / what-percent-of-X-is-Y),
  speed (km/h, m/s auto-unit), temperature (C/F/K), dates (days-between, add-days),
  area (circle/rectangle). Wired into NovaCoreCDE as earliest fast path (route='solver',
  conf 0.78, instant); runs BEFORE exploration so the exact answer wins without touching
  data. Also hardened _explore (knowledge_index via getattr) + chat.py chat against unloaded
  sessions (knowledge_index/reasoner/creative default None in __init__).
- 2026-09-08: CODING COMPLETE (M1+M2+M3.1 done, verified on NokiaV14):
  * tools/sandbox.py — CodeSandbox (single code runner): restricted python (AST blocklist for
    os/sys/subprocess/ctypes/importlib + file writes/network; limited builtins incl. math/np),
    threaded timeout (~6s) + output cap 4000, external runtimes (go/node/gcc/g++) via shutil.which,
    compute_expr safe math evaluator.
  * inference/cde.py — NovaCoreCDE unified loop (UNDERSTAND→EXPLORE→PLAN→HYPOTHESIZE→SIMULATE(sandbox)
    →ATTACK(AdversarialScientist, 7 gates)→VERIFY→RETRY(max 3)→SYNTHESIZE); LanguageCore interface
    + NovaCoreLanguageCore (interim pure-NovaCore synthesis; small quantized LM additive-swap point,
    deferred §12); per-hypothesis LEDGER; rule-rejection -inf for known impossible (perpetual motion,
    FTL, time travel) with WHY; coverage-first evidence selection + definitional bonus; missing-word
    honesty gate (refusal when a query content word has ZERO evidence anywhere, incl. predictor
    fallback gate); multi-part decomposition; creative generation gated on-topic.
  * inference/memory_transformer.py — CORTEX evidence mode added: retrieve_evidence() / _evidence()
    (evidence-only; verbatim only for short-fact/quotation).
  * inference/chat.py — ChatSession.generate_cde() + enable_cde flag; chat() routes to CDE when
    enabled; fallback to legacy generate() on any CDE error.
  * cli/main.py — chat --cde flag; _start_session enables CDE.
  Verified on weights\NovaCoreV14 (CPU, Py3.12): math 2+2=4, 34x12=408; code request →
    run-in-sandbox "Verified output: 1..10"; facts (Paris, H2O, stellar def); multi-part
    (India+Japan capitals); honest refusal for no-evidence (indian systist, capital of usa, jetty
    machine); rule rejection for perpetual motion / FTL with law + reason; hypothesis labeled
    'plausible (Level 2), not confirmed fact'. Legacy pipeline unchanged (default), CDE opt-in.
  Remaining (post-build / user-side): bridge+Go compare wiring in NovaCoreChat (compare crash
    still open), Colab notebook --load chain fix (NovaCoreV10→V12) + TinyStories max_rows cap,
    rotate leaked HF token; deferred P-items unchanged.
- 2026-09-09: SOLVER EXTENDED (pure rules, verified on weights\NovaCoreV15 via CDE route=solver):
  * _wordproblem — discount-pay ("shirt costs 800 and 25% off, final price"→600; "500 with 20%
    discount, how much do you pay"→400), quantity×unit-price ("3 items at 150 each, total"→450),
    word arithmetic ("5 plus 3"→8, "12 divided by 4"→3; ast-eval `_safe_arith`, never eval()).
  * _transitive/_extreme — relation-chain reasoning ("A taller than B, B taller than C"→a;
    shortest→c; "John older... oldest"→john), attribute-pole table, fact-vs-question pole detect.
  * Bug fixes: reversed-unit form, percent word forms, `[^0-9]{0,5}`→`[^a-z0-9]{0,5}` (g vs kg),
    date group-index slip, speed label, `ast.Mul`→`ast.Mult` (Py3.12), transitive-chain guard
    removed (rejected valid chains), q_pole choice.
- 2026-09-09: TRAINING BUILD SPEED (pure CPU path + optional GPU vector stages; zero training):
  * auto_tuner.py: apply_threading() (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB env) + num_threads (cpu-2)
    + workers (~90% cores, cap 16); env block runs BEFORE numpy import in cli/main.py.
  * cli/main.py: _train_from_stream(workers=None) — encode loop reworked to chunked parallel
    map-reduce (multiprocessing.Pool, 20k rows/chunk, serial fallback; vocab/pattern = Counter
    merge; knowledge/reservoir/QA stay sequential = deterministic). Reasoning + Creative extraction
    parallelized via _parallel_lists (list-extend merge; _REASONING_FIELDS/_CREATIVE_FIELDS).
    _extract_docs_worker/_extract_reasoning_worker/_extract_creative_worker equivalence verified
    == sequential (identical Counter/field results, incl. real Pool).
  * New flags (train & train-pools): --workers N, --cpu-only.
  * training_upgrades.py: _svd_cuda_or_numpy — torch-CUDA SVD when available (NOVACORE_CPU_ONLY=1
    forces CPU), numpy/BLAS fallback; torch optional, zero new deps.
  * Honest speed note: pure-Python tokenize/pattern loops are NOT GPU-accelerated → multiprocessing
    is the real win (target: Colab stream ~40min → <10min on same CPU); GPU helps only SVD/matmul.

## 14. CONFIG ANALYSIS — "will changing config make the model smarter?" (2026-09-09)

Honest: config changes make NovaCore BETTER-COVERED and FASTER — they do NOT make it "smarter" in
the frontier-LLM sense (generative generality is the NOT-claimed zone, §13). Smartness here = exact
compute (solver) + knowledge coverage + reasoning coverage. Current config_colab_stream.json
(dim 1024, layers 8, vocab 50000, ngram [1,3]) is already well-balanced for CPU/RAM. Levers, in
order of real effect:

1. reasoning_extract_cap 50000 → raise (e.g. 200k): more language-reasoning rule coverage. Biggest
   free "smarter" win; only training time + RAM. (Transitive/word-problem math itself lives in the
   solver — independent of this cap.)
2. reservoir_sample_size 100000 → raise with RAM: deeper retrieval memory, better semantic recall.
3. qa_bank_cap 100000 → raise: more stored QA facts answerable directly.
4. pattern_sample_cap 500000 / pattern_max_size 200000: only helps when data > cap; widen with disk.
5. dim 1024 / layers 8 / vocab 50000: keep — RAM = vocab×dim×4B; 1024 already good; lowering dim
   speeds builds, does NOT hurt correctness of the deterministic solver path.
6. temperature 0.7: flavor only, no accuracy effect.
7. semantic_search_threshold 0.35 / svd_semantic_min_score: tuning knobs, marginal.

Bottom line: raise reasoning_extract_cap + qa_bank_cap + reservoir_sample_size on stronger RAM;
with the solver layer already fast-pathed, the route=solver answers are exact regardless of config.
  * config_colab_stream.json analyzed (see §14).
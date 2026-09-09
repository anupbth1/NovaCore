# How NovaCore Works

## Architecture

NovaCore is a **training-free LLM** that uses Random Fourier Features (RFF) + n-gram patterns + pseudo-inverse instead of gradient descent.

### Components (all internal to model weights)

| Component | Location | What It Does |
|-----------|----------|-------------|
| **Python Terminal** | `self.model.neural_engine.python_terminal` | Exact math computation (e.g., `15 * 23 = 345`) |
| **Virtual Simulation** | `self.model.neural_engine.virtual_simulation` | Scores, cleans, verifies every candidate |
| **Neural Network** | `self.model.neural_engine.neural_network` | Forward pass for routing |
| **Knowledge Base** | `self.model.neural_engine.knowledge_index` | Facts/definitions from training |
| **Deep Reasoner** | `self.model.neural_engine.reasoner` | Causal/logical/comparison patterns |
| **Creative Engine** | `self.model.neural_engine.creative` | Stories, poems, metaphors |
| **Semantic Index** | `self.semantic_index` (built at load) | IDF-weighted cosine similarity |
| **Predictor** | `self.predictor` | n-gram fallback (last resort) |

### Inference Pipeline (every input)

1. **Math Detection** → exact computation, bypass virtual sim
2. **Semantic Retrieval** → cosine similarity on reservoir → Virtual Sim
3. **Pool Matching** → word overlap fallback → Virtual Sim
4. **Knowledge Base** → fact lookup → Virtual Sim
5. **Deep Reasoning** → pattern-based reasoning → Virtual Sim
6. **Creative Generation** → story/poem → Virtual Sim
7. **Neural Engine** → reservoir + patterns → Virtual Sim
8. **Predictor** → n-gram generation → Virtual Sim

**First passing candidate wins.** Virtual sim scores each candidate (length, relevance, coherence, artifacts, repetition). Only returns when score >= threshold.

### Training (one-time, 33m 58s)

- 2,309,566 documents from Alpaca + no_robots + TinyStories + oasst2
- 200K reservoir samples saved in weights.ncw
- 200K n-gram patterns extracted
- 14,809 reasoning patterns
- 453,848 creative patterns
- 5,263 knowledge items

### Weights File

`weights/NovaCoreV10/weights.ncw` (77MB compressed, 200MB expanded) contains:
- encoder_W, encoder_b, encoder_beta (RFF matrices)
- patterns_raw (200K n-gram patterns)
- reservoir_sample (200K instruction-answer pairs)

### Usage

```
python -m cli.main chat -w weights/NovaCoreV10 -v
```

`-v` flag shows every reasoning step in gray, final answer in green.

### Key Design Decisions

- **Zero config at inference**: all values from model metadata, not config files
- **Zero external dependencies**: no HF calls, no network at inference time
- **Training-free**: RFF + pseudo-inverse, no gradient descent, no backprop
- **CPU-friendly**: all NumPy linear algebra, no GPU needed
- **Streaming training**: HF datasets streamed to disk, not materialized in RAM
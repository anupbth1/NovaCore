# NovaCore - STRUCTURE.md
> Last Updated: 2026-09-01 (data/ + weights/ centralization)

```
C:\project6\NovaCore\
├── novacore/                    # Main package
│   ├── __init__.py              # __version__ = "0.1.0"
│   ├── core/
│   │   ├── __init__.py
│   │   ├── encoder.py           # RandomFourierEncoder (analytic fit)  [DONE]
│   │   ├── hasher.py            # FeatureHasher (Count-Min-Sketch)     [DONE]
│   │   ├── reservoir.py         # ReservoirSampler                     [DONE]
│   │   ├── patterns.py          # PatternExtractor (n-grams)           [DONE]
│   │   ├── neural_engine.py     # NovaNeuralEngine (NEW - internal neural network + python terminal + virtual simulation + verification) [NEW]
│   │   └── novacore_model.py    # NovaCoreModel (NEW - complete self-contained model) [NEW]
│   ├── logger.py                # Professional console logger ([OK]/[SKIP]/[WARN]/[DONE] + timing) [DONE]
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── vocab.py             # Vocabulary                           [DONE]
│   │   └── text_processor.py    # TextProcessor                        [DONE]
│   ├── storage/
│   │   ├── __init__.py
│   │   └── weight_manager.py    # WeightManager                        [DONE]
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py         # Predictor + PatternPredictor         [DONE]
│   │   ├── chat.py              # ChatSession (Ollama-style)           [DONE]
│   │   ├── pool_retrieval.py    # PoolRetriever (NEW - pool data loading) [NEW]
│   │   └── retrieval.py         # RetrievalGenerator (NEW - retrieval-based generation) [NEW]
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py            # DatasetLoader (csv/json/txt/tsv)     [DONE]
│   │   ├── quality.py           # QualityFilter (CompactHashSet)       [DONE]
│   │   └── hf_loader.py         # HFLoader (download/stream) + cache auto-reuse/resume/random + sahi-split real disk download + texts-cache + accumulated pool (uid dedupe, random/add_datasets config) [DONE]
│   └── virtual_sim.py           # VirtualVerifier + VirtualSelfCorrector + SimulationEngine [DONE]
├── config/
│   ├── config.json              # Model config (defaults/paths/cli + internal components) [DONE]
│   ├── config-test.json         # Test config with internal components [DONE]
│   ├── data_config.json         # Dataset config (hf + native schema)    [DONE]
│   ├── data_small.json          # Small test dataset config             [DONE]
│   └── README.md                # Config documentation                  [NEW]
├── cli/
│   ├── __init__.py
│   └── main.py                  # All CLI commands + train + chat + hf [DONE]
├── tests/
│   ├── test_encoder.py          # [PASSED]
│   ├── test_tokenizer.py        # [PASSED]
│   ├── test_inference.py        # [PASSED]
│   ├── test_cli.py              # [PASSED - end to end]
│   ├── test_chat.py             # [PASSED - chat session]
│   └── test_neural_engine.py    # [NEW - neural engine tests]
├── data/                        # LOCAL datasets yahi hain
│   ├── sample.txt               # 16 sample documents
│   ├── conversation.txt         # 40 conversation documents
├── data/hf_cache/               # HF cached downloads + login token (config: hf_cache)
├── data/hf_data/                # HF saved/downloaded text (config: hf_data)
├── weights/                     # ALL models yahi bante hain (config: default_weights)
│   ├── TestV1/                  # dim=256, layers=2 (small test model)
│   ├── TestV2/                  # dim=512, layers=6 (larger test model)
│   ├── VedNex_V1/               # 50GB dataset model
│   ├── VedNex_V2/               # 50GB dataset model v2
│   └── ...                      # More models
├── PLAN.md
├── STRUCTURE.md
├── COMMANDS.md
├── README.md
├── RULES.md                     # Development rules (no hardcoded)
├── VedNex_NOTES.md              # Build log for 50GB models
├── MASTER_PLAN.md               # Master task list
├── setup.py
├── requirements.txt
├── test_improved.py             # Test script for improved generation
└── config_section.txt           # Config documentation (temp)
```

## New Internal Components (Phase 3)

### 1. Neural Engine (`novacore/core/neural_engine.py`)
- **NovaNeuralEngine**: Complete internal processing engine
- **NeuralNetworkInternal**: Lightweight CPU-friendly NN (64x32x16)
- **InternalPythonTerminal**: Internal code execution
- **VirtualSimulationInternal**: Quality scoring with caching
- **VerificationEngine**: Verification with retry logic

### 2. Model Brain (`novacore/core/novacore_model.py`)
- **NovaCoreModel**: Complete self-contained LLM
- Integrates all internal components
- Zero external dependencies

### 3. Pool Retrieval (`novacore/inference/pool_retrieval.py`)
- **PoolRetriever**: Direct pool.jsonl loading for retrieval

### 4. Retrieval Generator (`novacore/inference/retrieval.py`)
- **RetrievalGenerator**: Retrieval-based generation from dataset

### 5. Config Updates
- **config.json**: Added `neural_engine`, `virtual_simulation`, `verification_engine`, `python_terminal`, `chat` sections
- **config-test.json**: Test config with all internal components
- **config/README.md**: Config documentation

### 6. Documentation
- **README.md**: Updated with architecture, config guide
- **COMMANDS.md**: Added internal components configuration
- **STRUCTURE.md**: This file - updated with new components
# NovaCore - Training-Free LLM

> **Zero training. CPU-friendly. Learning at data ingestion time.**

NovaCore is a training-free Large Language Model that uses:
- **Random Fourier Features (RFF)** + n-gram patterns + pseudo-inverse for encoding
- **Knowledge Extraction** at encoding time — facts, procedures, causal relations, code, math
- **Virtual Simulation** — verifies every output before returning
- **Neural Network** — configurable internal processing pipeline
- **Math Detection** — Python terminal for calculations
- **Smart Pool Matching** — conversational patterns, Q&A retrieval

No GPU required. No backpropagation. No gradient descent. Everything happens on CPU.

---

## Features

| Feature | Description |
|---------|-------------|
| Zero Training | Learning happens at data ingestion, not training |
| CPU Friendly | Runs on any CPU with 8GB+ RAM |
| Knowledge Extraction | Extracts facts, procedures, code, math from data |
| Virtual Simulation | Verifies outputs before returning |
| Math Calculator | Built-in Python terminal for math |
| Chat Interface | Interactive conversational AI |
| Config-Driven | All parameters from config.json |
| HuggingFace Integration | Train from any HF dataset |

---

## Quick Start (Google Colab - Recommended)

### Step 1: Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)

### Step 2: Clone & Install
```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/anupbth1/NovaCore.git
%cd NovaCore
!pip install numpy tqdm datasets huggingface_hub
```

### Step 3: Login to HuggingFace
```python
# Cell 2: Login (needed for some datasets)
from huggingface_hub import login
login(token="YOUR_HF_TOKEN_HERE")
# Get token from: https://huggingface.co/settings/tokens
```

### Step 4: Download Datasets
```python
# Cell 3: Download all datasets for training
!python cli/main.py hf download --dataset config/data_colab.json
```

### Step 5: Train Model
```python
# Cell 4: Train from downloaded datasets (~15 min on Colab)
!python cli/main.py train-pools --dataset config/data_colab.json --output NovaCoreV1 --config config/config_colab.json
```

### Step 6: Test Chat
```python
# Cell 5: Test the model with various queries
from novacore.config import configure
configure('config/config_colab.json')
from novacore.inference.chat import ChatSession

chat = ChatSession('weights/NovaCoreV1')
chat.load()

tests = [
    'hello',
    'who are you',
    'how are you',
    'what is machine learning?',
    'calculate 2 plus 2',
    'tell me a joke',
    'namaste kaise ho',
    'capital of india',
    'what is python',
    'write a poem about nature',
]

for q in tests:
    r = chat.generate(q)
    print(f'You: {q}')
    print(f'NovaCore: {r[:200]}')
    print()
```

### Step 7: Interactive Chat
```python
# Cell 6: Chat loop (type 'quit' to exit)
chat.reset()
while True:
    user_input = input('You> ')
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    response = chat.generate(user_input)
    print(f'NovaCore> {response}')
    print()
```

---

## Local Installation

### Requirements
- Python 3.10+
- 8GB+ RAM recommended
- No GPU required

### Setup
```bash
git clone https://github.com/anupbth1/NovaCore.git
cd NovaCore
pip install numpy tqdm datasets huggingface_hub
```

### Train from Datasets
```bash
# Download datasets
python cli/main.py hf download --dataset config/data_small.json

# Train model
python cli/main.py train-pools --dataset config/data_small.json --output MyModel --config config/config-test.json
```

### Chat
```bash
python cli/main.py chat --weights weights/MyModel
```

---

## Project Structure

```
NovaCore/
├── novacore/
│   ├── core/
│   │   ├── encoder.py              # Random Fourier Features encoder
│   │   ├── patterns.py             # N-gram pattern extraction
│   │   ├── reservoir.py            # Reservoir sampling
│   │   ├── knowledge_extractor.py  # Learning at encoding time
│   │   ├── neural_engine.py        # Internal processing pipeline
│   │   └── novacore_model.py       # Model wrapper
│   ├── inference/
│   │   ├── chat.py                 # Chat interface
│   │   ├── predictor.py            # Pattern-based prediction
│   │   └── pool_retrieval.py       # Pool retrieval
│   ├── tokenizer/
│   │   ├── vocab.py                # Vocabulary management
│   │   └── text_processor.py       # Text processing
│   ├── storage/
│   │   └── weight_manager.py       # Model weight storage
│   ├── dataset/
│   │   ├── hf_loader.py            # HuggingFace dataset loader
│   │   └── loader.py               # Dataset loading
│   ├── config.py                   # Configuration management
│   └── auto_tuner.py               # Auto-tuning
├── config/
│   ├── config.json                 # Default config
│   ├── config-test.json            # Test config (larger model)
│   ├── config_colab.json           # Colab config (optimized)
│   ├── data_small.json             # Small dataset config (4 datasets)
│   ├── data_colab.json             # Colab dataset config (6 datasets)
│   └── README.md                   # Config documentation
├── cli/
│   └── main.py                     # CLI interface
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Data Ingestion (Learning Without Training)
When data enters NovaCore, the Knowledge Extractor automatically learns:
- **Facts**: "X is Y" patterns → stored as (subject, object) pairs
- **Q&A Pairs**: Question → Answer mappings
- **Procedures**: Step-by-step instructions
- **Causal Relations**: "X causes Y" patterns
- **Code Snippets**: Programming patterns
- **Math Formulas**: Mathematical expressions
- **Definitions**: "X means Y" patterns

### 2. Encoding
```
Text → Tokenize → Random Fourier Features → Feature Hashing
                                         ↓
                    Patterns extracted via n-gram analysis
                                         ↓
                    Reservoir sampling for representative subset
                                         ↓
                    Analytic weights computed via pseudo-inverse
```

### 3. Inference (Chat)
Query processing priority:
1. **Math Detection** → Python terminal calculation
2. **Pool Matching** → Conversational patterns + Q&A retrieval
3. **Neural Engine** → Pattern-based generation
4. **Predictor** → Fallback generation

### 4. Knowledge Base
Extracted knowledge is indexed by keywords for fast retrieval at inference time.

---

## Configuration

### config-test.json / config_colab.json
```json
{
  "defaults": {
    "dim": 1024,
    "layers": 8,
    "vocab_size": 50000,
    "max_tokens": 4096,
    "temperature": 0.7
  }
}
```

### Dataset Config
Working datasets:
1. **tatsu-lab/alpaca** (50K) - Instruction-following
2. **HuggingFaceH4/no_robots** (20K) - High-quality responses
3. **roneneldan/TinyStories** (20K) - Story generation
4. **OpenAssistant/oasst2** (30K) - Conversational AI
5. **dmlconvai/ConvAI2** (15K) - Dialogue
6. **squad_v2** (20K) - Q&A

---

## CLI Commands

```bash
# Download datasets
python cli/main.py hf download --dataset config/data_small.json

# Train model
python cli/main.py train-pools --dataset config/data_small.json --output ModelName --config config/config-test.json

# Chat
python cli/main.py chat --weights weights/ModelName

# Model info
python cli/main.py info --weights weights/ModelName

# Generate text
python cli/main.py generate --weights weights/ModelName --prompt "Your prompt here"
```

---

## Limitations

NovaCore is a training-free LLM using pattern matching + knowledge retrieval:
- ✅ Answer factual questions correctly
- ✅ Handle greetings and conversations
- ✅ Do math calculations
- ✅ Generate poems and stories
- ✅ Explain concepts
- ✅ Hindi-English bilingual
- ❌ Not deep reasoning like GPT-4
- ❌ Not novel creative content generation

---

## License

MIT License

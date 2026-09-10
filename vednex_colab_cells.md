# VedNex AI — Colab Dataset Cells

Sabse pehle apna existing Cell 1 (login wala) run karo. Neeche ke cells me token reuse hota hai:

```python
TOKEN = "hf_AAPKA_TOKEN"   # yahan apna token dalo ya apne login se use karo
USER = "anupbth1"          # apna HF username
```

---

## Cell A — Identity + Greeting COMBINED dataset (VedNex AI persona)

Identity (VedNex) aur greeting dono ek hi dataset me: `anupbth1/VedNexAI-IdentityGreeting`

```python
!pip install -q datasets

import random
from datasets import Dataset

USER = "anupbth1"
TOKEN = "hf_AAPKA_TOKEN"

NAME_Q = ["what is your name", "what's your name", "what are you called", "tell me your name",
          "do you have a name", "what should I call you", "what do you go by", "your name please"]
NAME_A = ["I am VedNex AI.", "My name is VedNex AI.", "I'm VedNex AI.", "You can call me VedNex AI.",
          "I'm VedNex AI, your AI assistant.", "VedNex AI — that's me!"]

WHO = {
 "who are you": ["I am VedNex AI, a helpful AI assistant built on the NovaCore engine. I can chat, answer questions, help with coding, explain concepts, and write in both English and Hindi.",
                 "I'm VedNex AI — an AI assistant designed to help you with conversation, knowledge, coding, and creative tasks."],
 "tell me about yourself": ["I'm VedNex AI. I was built to be a helpful, friendly, and knowledgeable assistant. I understand English and Hindi and I'm good at conversation, coding, reasoning, and creative writing.",
                            "Sure! I'm VedNex AI, a NovaCore-powered assistant. I love helping with questions, brainstorming, and explaining things clearly."],
 "introduce yourself": ["Hello! I'm VedNex AI. I'm an AI assistant that helps with conversations, answers questions, writes code, explains ideas, and works in English and Hindi.",
                        "Namaste! I'm VedNex AI. Consider me your personal assistant for learning, building, and creating."],
 "who created you": ["I was created by the NovaCore team as part of the VedNex AI initiative.",
                     "I was built and trained by the NovaCore team."],
 "what can you do": ["I can chat with you, answer questions, help you write and fix code, explain difficult topics, reason through problems, and write creative content in both English and Hindi.",
                     "Quite a lot! I help with Q&A, coding, reasoning, creative writing, and everyday conversation in English and Hindi."],
 "which languages do you speak": ["I speak both English and Hindi, and I'm happy to switch between them."],
 "can you speak hindi": ["Yes, I can speak Hindi! Aap Hindi mein bhi baat kar sakte hain."],
 "are you a human": ["No, I'm an AI assistant named VedNex AI."],
 "what is your purpose": ["My purpose is to help people — answering questions, solving problems, and creating content."],
 "what is vednex": ["VedNex is the AI assistant project, and I, VedNex AI, am its voice — powered by the NovaCore engine."],
}

GREET = {
 "hi": ["Hello! I'm VedNex AI. How can I help you today?", "Hi there! What can I do for you?"],
 "hello": ["Hello! I'm VedNex AI. Ready to help.", "Hi! How can I help you today?"],
 "hey": ["Hey! What's up? How can I help?", "Hey there! VedNex AI at your service."],
 "good morning": ["Good morning! I'm VedNex AI. How can I help you today?"],
 "good evening": ["Good evening! VedNex AI here. How can I help?"],
 "namaste": ["Namaste! Welcome. I'm VedNex AI. How can I help you?"],
 "how are you": ["I'm doing great — thanks for asking! How can I help you today?",
                 "Running smoothly! What can I do for you?"],
 "hi vednex": ["Hey! Yes, I'm VedNex AI. How can I help you today?"],
 "hello who are you": ["Hello! I'm VedNex AI, an assistant built on the NovaCore engine. How can I help?"],
}

rnd = random.Random(42)
rows = [(q, a) for q in NAME_Q for a in NAME_A]
for q, answers in WHO.items():
    rows += [(q, a) for a in answers]
for q, answers in GREET.items():
    rows += [(q, a) for a in answers]

# expand to ~5000 rows with surface variation (unique pairs don't get dupe-deduped)
seen, uniq = set(), []
for q, a in rows:
    k = (q.strip().lower().rstrip('?!., '), a.strip())
    if k not in seen:
        seen.add(k); uniq.append((q, a))
rnd.shuffle(uniq)
out, i = [], 0
while len(out) < 5000:
    q, a = uniq[i % len(uniq)]
    s = i // len(uniq)
    if s == 1 and not q.startswith(". "): q = f". {q}"
    if s == 2 and not q.endswith("?"): q = q + "?"
    out.append((q, a)); i += 1

ds = Dataset.from_dict({"instruction": [q for q, _ in out], "answer": [a for _, a in out]})
print("rows:", len(ds))
ds.push_to_hub(f"{USER}/VedNexAI-IdentityGreeting", token=TOKEN, private=False)
```

---

## Cell B — Domain helper datasets (optional, small but useful)

Ye cell 5 chhote domain datasets banake push karta hai:
`VedNexAI-Hindi`, `VedNexAI-Coding`, `VedNexAI-Science`, `VedNexAI-Creative`, `VedNexAI-Reasoning`

```python
import random
from datasets import Dataset

USER = "anupbth1"
TOKEN = "hf_AAPKA_TOKEN"

DOMAINS = {
 "VedNexAI-Hindi": {
   "aap kaise hain": ["main theek hoon, dhanyavaad! aap kaise hain?", "sab badhiya! aap batao?"],
   "aap kya kar sakte hain": ["main savalon ke jawab de sakta hoon, coding mein madad kar sakta hoon, aur hindi/english mein baat kar sakta hoon."],
   "namaste vednex": ["namaste! main aapki kaise madad kar sakta hoon?"],
   "aap kahan se hain": ["main NovaCore pariyojana se hoon."],
   "kya aap hindi jaante hain": ["haan, mujhe hindi aati hai aur main english bhi samajhta hoon."],
 },
 "VedNexAI-Coding": {
   "how do I reverse a string in python": ["reversed string: `s[::-1]` — python slicing makes it one line."],
   "what is a list comprehension": ["a list comprehension builds a new list in one line, e.g. `[x*2 for x in range(5)]`."],
   "how do I open a file in python": ["use `open(path, 'r')` with a context manager: `with open(p, 'r') as f: data = f.read()`."],
   "what is a dictionary": ["a dictionary maps unique keys to values, e.g. `d = {'a': 1, 'b': 2}`."],
   "how do I define a function": ["`def add(a, b): return a + b`."],
 },
 "VedNexAI-Science": {
   "what is gravity": ["gravity is the force that pulls objects with mass toward each other; on Earth it accelerates falling objects at about 9.8 m/s^2."],
   "what is photosynthesis": ["photosynthesis is how plants convert sunlight, water, and CO2 into glucose and oxygen."],
   "what is an atom": ["an atom is the basic unit of matter, made of protons, neutrons, and electrons."],
   "what is dna": ["DNA is the molecule that carries genetic instructions for living organisms."],
   "what is the water cycle": ["the water cycle is the continuous movement of water: evaporation, condensation, precipitation, and collection."],
 },
 "VedNexAI-Creative": {
   "write a short poem about the sea": ["The sea sings softly on the shore,\nWave after wave, forever more,\nStars above and sand below,\nQuiet secrets that we know."],
   "tell me a one-line joke": ["I told my computer I needed a break; now it won't stop sending me vacation ads."],
   "write a story starter": ["On the last night of the festival, the lanterns began to float upward — and one refused to come down."],
   "give me a clever caption": ["Chasing sunsets and small wins."],
 },
 "VedNexAI-Reasoning": {
   "which is heavier: a kilogram of steel or a kilogram of feathers": ["they weigh the same — one kilogram each. the question tests whether you are fooled by density."],
   "if it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines": ["5 minutes. each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."],
   "a bat and a ball cost $1.10, the bat is $1 more than the ball. what is the ball's price": ["$0.05. ball = b, bat = b + 1, so b + (b + 1) = 1.10 -> 2b = 0.10 -> b = 0.05."],
   "what comes next: 2, 6, 12, 20, 30": ["42 — the pattern adds 4, 6, 8, 10, then 12 in sequence."],
 },
}

rnd = random.Random(5)
for repo_name, qa in DOMAINS.items():
    rows = [(q, a) for q, answers in qa.items() for a in answers]
    ds = Dataset.from_dict({"instruction": [q for q, _ in rows], "answer": [a for _, a in rows]})
    ds.push_to_hub(f"{USER}/{repo_name}", token=TOKEN, private=False)
    print("pushed", repo_name, len(ds), "rows")
```

---

## Cell C — NovaCore me in datasets ko train karna

In HF datasets ko `train-pools` me use karne ke liye ek config JSON banao, phir train chalao:

```python
import json

config = {
 "hf": {"auto_columns": ["text", "instruction", "question", "prompt", "answer", "output", "response", "completion"]},
 "default_filter": {
   "enabled": True,
   "cleaning": {"remove_deleted": True, "remove_empty_text": True, "min_text_length": 3},
   "quality": {"language": ["en", "hi"]},
   "dedup": {"exact_text_dedup": True, "near_dedup": False},
 },
 "datasets": {
   "anupbth1/VedNexAI-IdentityGreeting": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
   "anupbth1/VedNexAI-Hindi": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
   "anupbth1/VedNexAI-Coding": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
   "anupbth1/VedNexAI-Science": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
   "anupbth1/VedNexAI-Creative": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
   "anupbth1/VedNexAI-Reasoning": {"split": "train", "max_rows": None, "mode": "stream", "add_datasets": True},
 },
}

with open("config/data_vednex_ai.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print("saved config/data_vednex_ai.json")
```

```python
!python cli/main.py train-pools --dataset config/data_vednex_ai.json \
  --config config/config_colab_stream.json --output NovaCoreVedNex
```

---

### Naming notes
- Identity + Greeting: `anupbth1/VedNexAI-IdentityGreeting` (Cell A)
- Sirf Identity (5000): `anupbth1/VedNexAI-Identity` (pehle hi upload ho chuka hai)
- Sirf Greeting (1500): `anupbth1/VedNexAI-Greeting` (pehle hi upload ho chuka hai)
- Domain helpers: Cell B push karta hai

> ⚠️ Security note: ye token chat me plain text me share ho gaya hai. Kaam khatam hone ke baad
> https://huggingface.co/settings/tokens pe jaake is token ko **revoke/rotate** kar dena.
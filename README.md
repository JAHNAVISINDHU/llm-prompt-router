# 🤖 LLM-Powered Prompt Router

An intelligent Python service that classifies user intent using an LLM and routes each request to a specialized AI expert persona — delivering accurate, context-aware responses instead of relying on a single monolithic prompt.

---

## ✨ Features

| # | Requirement | Status |
|---|-------------|--------|
| 1 | ≥ 4 distinct expert system prompts in a configurable JSON file | ✅ |
| 2 | `classify_intent()` → `{ "intent": str, "confidence": float }` | ✅ |
| 3 | `route_and_respond()` selects correct prompt & generates response | ✅ |
| 4 | `unclear` intent triggers a clarification question (no guessing) | ✅ |
| 5 | Every request logged to `logs/route_log.jsonl` (JSON Lines) | ✅ |
| 6 | Malformed LLM JSON handled gracefully — defaults to `unclear/0.0` | ✅ |

---

## 🗂️ Project Structure

```
prompt-router/
├── main.py           # Interactive CLI entry point
├── router.py         # Core logic: classify_intent, route_and_respond, logging
├── prompts.json      # Expert system prompts (configurable)
├── run_tests.py      # Stdlib-based test suite (no extra deps needed)
├── test_router.py    # pytest-style tests (needs pytest installed)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── docker-compose.yml
├── .env.example      # Template for your API key
├── .gitignore
└── logs/
    └── route_log.jsonl  # Auto-created on first run
```

---

## 🚀 Quick Start

### Option A — Local Python (recommended for development)

#### 1. Clone / enter the project directory
```bash
cd prompt-router
```

#### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate           # Windows
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Set your OpenAI API key
```bash
cp .env.example .env
# Open .env and replace the placeholder with your real key:
# OPENAI_API_KEY=sk-...
```

Then load it into your shell:
```bash
export OPENAI_API_KEY="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"      # Linux / macOS
# set OPENAI_API_KEY=sk-your-key-here         # Windows CMD
# $env:OPENAI_API_KEY="sk-your-key-here"      # Windows PowerShell
```

#### 5. Run the interactive CLI
```bash
python main.py
```

---

### Option B — Docker

#### 1. Build the image
```bash
docker build -t prompt-router .
```

#### 2. Run interactively
```bash
docker run -it \
  -e OPENAI_API_KEY="sk-your-key-here" \
  -v "$(pwd)/logs:/app/logs" \
  prompt-router
```

#### 3. Or use docker-compose
```bash
# Copy and edit your key first
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

docker-compose up router
```

---

## 🧪 Running Tests

### Without installing pytest (stdlib only)
```bash
python3 run_tests.py
```
Expected output:
```
Ran 20 tests in 0.07s
OK
```

### With pytest (after pip install)
```bash
pytest test_router.py -v
```

### Tests via Docker
```bash
docker-compose run test
# or
docker run --rm -e OPENAI_API_KEY=sk-test prompt-router pytest test_router.py -v
```

---

## 🧠 How It Works

```
User Input
    │
    ▼
┌─────────────────────┐
│  classify_intent()  │  ← LLM call #1 (gpt-3.5-turbo, temp=0)
│  Returns:           │    Returns: { "intent": "code", "confidence": 0.93 }
│  { intent,          │
│    confidence }     │
└────────┬────────────┘
         │
         ▼
    intent == "unclear"?
         │
    YES ─┤─ NO
         │         │
         ▼         ▼
  Return      ┌─────────────────────┐
  clarif.     │  route_and_respond()│  ← LLM call #2 (expert persona)
  question    │  Selects system     │
              │  prompt from        │
              │  prompts.json       │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  log_request()      │  → logs/route_log.jsonl
              └─────────────────────┘
                       │
                       ▼
                Final Response
```

---

## 🎭 Expert Personas

| Intent | Persona | Handles |
|--------|---------|---------|
| `code` | Elite Software Engineer | Debugging, algorithms, architecture, code review |
| `data_analysis` | World-class Data Scientist | Statistics, ML, SQL, data visualization |
| `writing` | Master Writer & Editor | Essays, emails, reports, creative content |
| `career` | Seasoned Career Coach | Resumes, interviews, salary negotiation, networking |
| `unclear` | Clarification Handler | Asks guiding questions to identify user need |

---

## 📄 Log Format

Every request appends one JSON object to `logs/route_log.jsonl`:

```json
{
  "timestamp": "2025-06-15T14:32:10.123456Z",
  "user_message": "How do I reverse a linked list?",
  "intent": "code",
  "confidence": 0.96,
  "final_response": "Great question! Here's a clean Python implementation..."
}
```

View recent logs from the CLI by typing `log` at the prompt.

---

## 🔧 Configuration

To add a new expert persona, edit `prompts.json`:

```json
{
  "your_new_intent": {
    "label": "Your Expert Label",
    "system_prompt": "You are a world-class expert in..."
  }
}
```

Then update the classifier's system prompt in `router.py` (`CLASSIFIER_SYSTEM_PROMPT`) to include the new intent label.

---

## 🛡️ Error Handling

- **Malformed LLM JSON** → Gracefully caught, defaults to `unclear` / `0.0`
- **Unknown intent labels** → Normalised to `unclear`
- **Markdown-wrapped JSON** (` ```json ... ``` `) → Stripped before parsing
- **API network errors** → Caught, defaults to `unclear`, user notified
- **Missing API key** → Clear `EnvironmentError` with instructions

---

## 📋 Requirements

- Python 3.9+
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- `openai>=1.30.0`, `python-dotenv>=1.0.0`
- Optional: `pytest>=8.0.0`, `pytest-mock>=3.14.0` for test suite

---

## 💡 Example Session

```
You: how do i sort a dictionary by value in python
✅ Intent: CODE             Confidence: 97%
──────────────────────────────────────────────────────
🤖 Assistant:

Great question! Here are the most Pythonic ways to sort a dict by value...

You: im not sure what i need help with
✅ Intent: UNCLEAR          Confidence: 30%
──────────────────────────────────────────────────────
🤖 Assistant:

I'd love to help! Could you give me a bit more detail...
• Coding — debugging, writing scripts, software design
• Data Analysis — statistics, datasets, machine learning
• Writing — drafting emails, editing documents, creative content
• Career Advice — resumes, interviews, job searching
```

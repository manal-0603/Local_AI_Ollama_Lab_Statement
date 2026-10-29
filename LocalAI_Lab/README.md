# FSTT Local AI Assistant

A course assistant that runs **entirely on your machine**: a Flask web interface talks to a local LLM served by
[Ollama](https://ollama.com), grounds its answers in the course documents of `documents/` through a small
retriever (RAG), keeps the conversation history, can answer with a validated JSON structure, and reports the
response time of every answer. No API key, no Internet connection and no data leaving the computer.

Built for the practical lab *Running and Developing Local AI Applications* (PhD programme, FSTT Tangier).
The executed lab notebook with all experiments and answers is `../Local_AI_Ollama_Lab_Solution.ipynb`;
the written report is `report.pdf`.

![Web application](screenshots/06_webapp.png)

## Features

- **100 % local** – Flask is bound to `127.0.0.1`, Ollama listens on `localhost:11434`; the only network traffic is on the loopback interface.
- **Grounded answers (RAG)** – a TF-IDF keyword retriever (`rag.py`, no external library) selects the two most relevant course documents and injects them as context; the model is instructed to answer *only* from that context and to say *"Information not available."* otherwise. Sources and their scores are shown under each answer.
- **Conversation memory** – the last 6 user/assistant turns are re-sent with every request through Ollama's `/api/chat` endpoint, so follow-up questions work ("And how is it different from supervised learning?").
- **Structured responses** – a toggle switches the answer to a JSON object validated by a JSON Schema (`topic`, `summary`, `key_points[]`, `difficulty ∈ {easy, medium, hard}`), using Ollama's structured outputs.
- **Metrics** – response time, generated tokens and tokens/second (from Ollama's `eval_count` / `eval_duration`) under every answer.
- **Modern UI** – sidebar with model and knowledge base, chat bubbles, suggested questions, typing indicator, dark mode (follows the system setting), mobile layout. Pure HTML/CSS/JS, no CDN.
- **Console chatbot** and **benchmark script** included.

## Architecture

```
Browser (templates/index.html)
   │  POST /ask  {question, structured}
   ▼
Flask app.py (127.0.0.1:5001)
   ├─ rag.Retriever ──► documents/*.txt  (TF-IDF scoring → top-2 documents)
   ├─ history[]        (last 6 turns)
   ├─ messages = [system] + history + [user: CONTEXT + QUESTION]
   └─ POST http://localhost:11434/api/chat  {model, messages, think:false, format?}
                 │
                 ▼
          Ollama ──► llama.cpp runner ──► local LLM (GPU / CPU)
                 │
                 ▼
   {answer, structured, sources, seconds, tokens, tokens_per_s}  ──► Browser
```

## Project layout

```
LocalAI_Lab/
├── README.md                 this file
├── report.pdf                lab report (answers to Q1–Q15, measurements, screenshots)
├── source/
│   ├── app.py                Flask web application (RAG + memory + JSON mode + timing)
│   ├── chatbot.py            console chatbot with conversation memory (/api/chat)
│   ├── rag.py                tokeniser, TF-IDF retriever, prompt builder
│   ├── benchmark.py          performance benchmark of local models
│   └── templates/index.html  web interface
├── documents/                knowledge base (plain-text course notes)
│   ├── ai.txt · machine_learning.txt · reinforcement_learning.txt · deep_learning.txt · course.txt
└── screenshots/              01_installation … 06_webapp.png, benchmark.png
```

## Prerequisites

| Requirement | Tested with |
|---|---|
| Ollama | 0.40.12 (`brew install ollama` on macOS, installer on Windows, `curl -fsSL https://ollama.com/install.sh \| sh` on Linux) |
| Python | 3.12 (3.10+ required) |
| Hardware | Apple M2 Pro, 16 GB unified memory. Any machine with ≥ 8 GB RAM works with the models below; a GPU (Metal / CUDA) makes generation several times faster. |
| Disk | ≈ 3 GB for the two models + embeddings |

## Installation

```bash
# 1. Start the Ollama server (the desktop app starts it automatically)
ollama serve                      # or: brew services start ollama

# 2. Download the models
ollama pull qwen3.5:2b-q4_K_M     # main model: 1.9 B params, Q4_K_M, vision + tools + structured output
ollama pull gemma3:1b             # second model, used for comparison
ollama pull nomic-embed-text      # optional: embeddings for semantic retrieval

# 3. Python environment
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install flask requests psutil pandas
```

> **Why `qwen3.5:2b-q4_K_M` and not `qwen3.5:2b`?** On Apple Silicon the plain `qwen3.5:2b` tag resolves to an
> `mxfp8` build served by Ollama's MLX runner, which does **not** support JSON-schema structured output
> (`HTTP 501 – structured output is unavailable`). The GGUF build runs on llama.cpp and supports everything used here.
> On Linux/Windows either tag works. Any other chat model can be used through the `MODEL` variable.

## Running

### Web application

```bash
cd source
python app.py                     # → http://127.0.0.1:5001
```

Type a question, or click one of the suggested questions. Turn on **Structured JSON** to receive a JSON object.
**Reset** clears the conversation history on the server.

### Console chatbot

```bash
cd source
python chatbot.py                 # type 'exit' to stop
```

### Benchmark

```bash
cd source
python benchmark.py qwen3.5:2b-q4_K_M gemma3:1b
```

Each model is unloaded first (cold start), then asked the same 500-word prompt; the script prints disk size,
loaded size, VRAM, load time, response time, tokens, tokens/s and system RAM.

## Configuration

Environment variables read by `app.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen3.5:2b-q4_K_M` | Ollama model used for answers |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `PORT` | `5001` | Flask port (5000 is taken by AirPlay on macOS) |

Constants in `app.py`: `MAX_TURNS = 6` (remembered turns), `SYSTEM` (grounding instruction), `STRUCT_SCHEMA`
(JSON schema of structured answers). Thinking is disabled (`"think": false`) because reasoning models such as
qwen3.5 otherwise spend hundreds of hidden tokens before each answer.

### Knowledge base

Drop any UTF-8 `.txt` file into `documents/` and restart the app: files are indexed at start-up. The retriever is
keyword-based (TF-IDF); a question that uses different words than the document may not match. The notebook shows
an optional semantic retriever with `nomic-embed-text` embeddings and cosine similarity that solves this.

## HTTP API

| Route | Method | Body | Response |
|---|---|---|---|
| `/` | GET | – | web interface |
| `/ask` | POST | `{"question": "...", "structured": false}` | `{"answer", "structured", "sources": [{"doc", "score"}], "seconds", "tokens", "tokens_per_s"}` |
| `/reset` | POST | – | `{"ok": true}` |

Example:

```bash
curl -s http://127.0.0.1:5001/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is reinforcement learning?", "structured": true}'
```

## Results (Apple M2 Pro, 16 GB)

| Model | Loaded size | Tokens/s | Response time (500 words) | Notes |
|---|---|---|---|---|
| qwen3.5:2b-q4_K_M | 1.61 GB | 66 | 12.5 s | vision, tools, thinking, structured output |
| gemma3:1b | 0.87 GB | 90 | 9.6 s | text only, fastest |

RAG answers in the web app take 1–3 s. Full measurements, charts and the discussion of the size/quality/latency
trade-off are in `report.pdf` and in the notebook (Parts 6 and 16).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Ollama is not running` | start it: `ollama serve` (or the desktop app / `brew services start ollama`) |
| `model not found` | `ollama pull qwen3.5:2b-q4_K_M` (or set `MODEL` to a model you have: `ollama list`) |
| `501 structured output is unavailable` | the model runs on the MLX runner; use a GGUF tag such as `qwen3.5:2b-q4_K_M` |
| `Address already in use` on 5001 | another instance is running: `lsof -i :5001`, or set `PORT=5002` |
| Empty or very slow answers | thinking is enabled in your own code; add `"think": false`, or use a smaller model |
| Answers ignore the documents | small models follow grounding instructions *most* of the time; lower the temperature, shorten the history (`MAX_TURNS`), or use a larger model |

## Security note

Ollama has **no authentication** by default. Keep it on `127.0.0.1` (never set `OLLAMA_HOST=0.0.0.0` on an open
network); if remote access is needed, put a reverse proxy with TLS and authentication in front and restrict the
management routes (`/api/pull`, `/api/delete`, `/api/create`). Documents placed in `documents/` are injected into
the prompt, so untrusted documents can carry prompt-injection instructions. See report Q15.

## License

Code of this lab: MIT. Models keep their own licences (Qwen: Apache 2.0, Gemma: Gemma Terms of Use).

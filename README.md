# Laf

## An Agentic + RAG + Plugin System with Intent Routing

Laf is a modular agent framework that combines planning, intent routing, retrieval, and plugin-based tool execution into one clean orchestration system.

It is designed to be lightweight for demos but structured for long-term scalability.

Laf takes a user task, breaks it into subtasks, determines what each step represents, optionally retrieves supporting knowledge, and generates a structured response. It can also execute tools through a dynamic plugin system.

---

# What Laf Does

Given a user request like:

> "Compare LSTM and Transformers for time series forecasting."

Laf will:

* Generate structured subtasks
* Classify each step into an intent
* Optionally retrieve supporting documents
* Generate a grounded answer
* Optionally refine the response
* Optionally execute relevant tools

Laf is not just a chatbot — it is a task orchestration engine built around intent.

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

* transformers
* torch
* accelerate
* sentence-transformers
* fastapi
* uvicorn
* pydantic

---

# Running Laf

Start the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/v1/health
```

---

# Example Usage

Planning only:

```
POST /v1/plan
{
  "task": "Build a small vegetable garden in my backyard."
}
```

Route intents:

```
POST /v1/route
{
  "task": "Compare LSTM and Transformers for time series forecasting."
}
```

Full execution:

```
POST /v1/run
{
  "task": "...",
  "execute_tools": false,
  "use_rag": true,
  "reflect": true
}
```

---

# Configuration

Laf is configurable through its system configuration module.

Important settings include:

* `model_name` — The language model used for planning and generation (default: `Qwen/Qwen2.5-3B-Instruct`)
* `embed_model` — The embedding model used for similarity and retrieval
* `similarity_threshold` — Controls when a new intent should be created
* `max_steps` — Maximum number of planner steps per task
* `plugin_dir` — Directory containing external tools
* `intent_store_path` — File path where learned intents are persisted

Adjusting the similarity threshold controls how adaptive the system becomes:

* Higher value → more strict routing
* Lower value → more dynamic intent growth

---

# Future Goals

Short-Term Goals:

* Persistent vector database backend (FAISS / Qdrant / Milvus)
* Streaming generation responses
* Intent analytics and monitoring
* Confidence-based routing improvements

Mid-Term Goals:

* Multi-agent coordination
* Memory layer (episodic + semantic)
* Tool execution sandboxing
* Automated answer quality scoring

Long-Term Goals:

* Distributed intent sharing
* Self-optimizing routing thresholds
* Plugin recommendation system
* Reinforcement-style feedback loop
* Autonomous task chains

---

# Design Philosophy

Laf is built around separation of concerns and modular growth.

Each layer can evolve independently:

* Planner logic can improve without touching retrieval
* Intent routing can scale without retraining models
* Tools can be added without modifying the core engine
* Generator models can be swapped without redesigning architecture

Laf starts simple, but it is designed to scale into a powerful intent-driven orchestration system.

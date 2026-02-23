# Laf

Laf is a modular agent orchestration framework built around structured planning, intent routing, retrieval, and tool execution.

It is designed to start simple and grow into a scalable, long-term agent system.

---

## Overview

Given a user task, Laf:

1. Generates a structured plan
2. Validates the plan
3. Routes each step into an intent
4. Optionally executes tools
5. Optionally retrieves supporting documents (RAG)
6. Generates a response
7. Logs a full execution trace

Laf is not a chatbot.
It is a task orchestration engine built around intent-driven reasoning.

---

## Current Features

### Core Engine

* Structured Planner (JSON-based planning)
* Plan Reviewer (validation layer)
* Swappable plan formats (linear / tree / dag)
* Hybrid Intent Router (embedding similarity + LLM fallback)
* Dynamic intent growth
* Plugin-based tool registry
* Tool schema validation

### Retrieval & Generation

* Short-term semantic retrieval (RAG)
* Optional reflection layer (critic agent)
* Configurable generation settings

### Runtime & Observability

* Multi-instance model pool (same model, multiple instances)
* Round-robin scheduling
* Multi-run execution support
* Per-instance performance metrics
* Full execution trace (JSONL per task)
* Artifact download per task (ZIP)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Tatsuwari/laf
cd laf
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Laf as a package (important):

```bash
pip install .
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

## Running the API

Start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/v1/system/health
```

---

## Running a Task

Endpoint:

```
POST /v1/run
```

Example request:

```json
{
  "task": "Compare LSTM and Transformers for time series forecasting.",
  "execute_tools": false,
  "use_rag": true,
  "reflect": true,
  "runs": 3
}
```

The `runs` parameter allows executing the same task multiple times across model instances.

---

## Execution Traces

Each task produces a structured trace stored under:

```
data/traces/{task_id}/trace.jsonl
```

You can download artifacts via:

```
GET /v1/artifacts/{task_id}
```

This enables:

* Debugging
* Performance analysis
* Future memory construction

---

## Configuration

Laf is configured through `SystemConfig`.

Key settings:

* `model_name` – language model used for planning and generation
* `embed_model` – embedding model for routing and retrieval
* `similarity_threshold` – controls new intent creation sensitivity
* `max_steps` – planner step limit
* `plan_format` – linear, tree, or dag
* `trace_enabled` – enables execution tracing
* `trace_dir` – trace storage location

Adjusting the similarity threshold changes how adaptive the intent system becomes.

---

## Roadmap (Toward v1)

* Async execution support
* Improved plugin system and documentation
* Quantized model support
* Streaming responses
* Basic evaluation metrics
* Clean release packaging

---

## Design Philosophy

Laf is built with:

* Clear separation of concerns
* Replaceable subsystems
* Observability-first execution
* Scalable architecture

Each layer can evolve independently without breaking the rest of the system.

Laf begins as a structured task engine and is designed to evolve into a long-term adaptive agent framework.

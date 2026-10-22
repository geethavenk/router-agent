# Router Agent

A customer-support **router agent** for an online education platform, built with [LangGraph](https://langchain-ai.github.io/langgraph/) and an OpenAI-compatible LLM. Each incoming query is classified by an LLM-based router, then handed to the specialist agent best suited to answer it. Low-confidence classifications fall back to a general support agent instead of risking a wrong hand-off.

## How it works

```
                    ┌──► technical ──┐
                    │                │
START ──► router ───┼──► course ─────┼──► END
                    │                │
                    ├──► billing ────┤
                    │                │
                    └──► general ────┘
```

1. **Router node** – The LLM returns a structured `RouteDecision` (Pydantic): a `route` (`technical`, `course`, `billing`, `general`) and a `confidence` score between 0 and 1.
2. **Confidence gate** – If `confidence < 0.70` (`CONFIDENCE_THRESHOLD` in [agent.py](agent.py)), the final route is forced to `general`. The original LLM prediction is kept in state for logging.
3. **Conditional edge** – The graph branches on `final_route` to one of the specialist agents.
4. **Specialist agents** – Each has its own prompt and guardrails (e.g. no invented account, transaction, or course details):

| Route | Handles |
|-------|---------|
| `technical` | Video buffering, login problems, website/app errors |
| `course` | Course recommendations, learning paths, career-oriented advice |
| `billing` | Payments, refunds, duplicate charges, subscriptions, activation after payment |
| `general` | Everything else, plus low-confidence queries |

### Graph state

| Field | Description |
|-------|-------------|
| `query` | The user's question |
| `route` | Route predicted by the LLM |
| `confidence` | Router's confidence (0–1) |
| `final_route` | Route actually executed (after the confidence gate) |
| `response` | The specialist agent's answer |

## Project structure

```
router-agent/
├── agent.py           # LLM setup, router, specialist agents, LangGraph definition
├── main.py            # Logging, ask_agent() helper, and demo queries
├── requirements.txt   # Python dependencies
├── .env.example       # Template for environment variables
└── router_agent.log   # Generated at runtime
```

## Setup

### Prerequisites

- Python 3.10+
- An API key and base URL for an OpenAI-compatible endpoint
- (Optional) A [LangSmith](https://smith.langchain.com/) API key for tracing

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Yes | API key for the LLM provider |
| `BASE_URL` | Yes | Base URL of the OpenAI-compatible endpoint |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default in example: `router-agent-demo`) |

The model is set in [agent.py](agent.py) (`openai/gpt-oss-120b-Turbo`, temperature 0). Change it there to use a different model.

## Usage

Run the built-in demo, which sends nine sample queries covering every route plus a vague one:

```bash
python main.py
```

For each query it prints the router prediction, confidence, threshold, final route, and the agent's response. Everything is also logged to the console and to `router_agent.log`.

Use it from your own code:

```python
from main import ask_agent

result = ask_agent("I was charged twice for my subscription.")
print(result["final_route"])  # billing
print(result["response"])
```

`ask_agent` returns the final graph state, or `None` if an error occurred.

### Example output

```
USER QUERY
It's not working

ROUTER PREDICTION
technical

CONFIDENCE
0.55

CONFIDENCE THRESHOLD
0.70

FINAL ROUTE
general
```

(Exact values depend on the model.)

## Customization

- **Tune the fallback** – Adjust `CONFIDENCE_THRESHOLD` in [agent.py](agent.py). Higher values send more queries to `general`.
- **Add a department** – Add the new label to `RouteDecision.route`, describe it in the router prompt, write a new agent function, then register it with `add_node`, the conditional-edge map, and an edge to `END`.
- **Change the model or provider** – Edit the `ChatOpenAI(...)` call in [agent.py](agent.py).

## Tech stack

LangGraph · LangChain (`langchain-openai`) · Pydantic · python-dotenv · LangSmith (optional)

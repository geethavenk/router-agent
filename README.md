# Router Agent

A support router for an online education platform, built with LangGraph. An LLM classifies each query and sends it to the right specialist agent: `technical`, `course`, `billing` or `general`.

```
START ──► router ──┬──► technical ──┐
                   ├──► course ─────┤
                   ├──► billing ────┼──► END
                   └──► general ────┘
```

The router returns a route and a confidence score. If confidence is below 0.70 (`CONFIDENCE_THRESHOLD` in [agent.py](agent.py)), the query goes to `general` instead. The LLM's prediction (`route`) and the route actually used (`final_route`) are both logged.

## Setup

Requires Python 3.10+ and an OpenAI-compatible API endpoint.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your values
```

| Variable | Description |
|----------|-------------|
| `API_KEY` | LLM provider API key |
| `BASE_URL` | OpenAI-compatible endpoint URL |
| `LANGSMITH_*` | Optional LangSmith tracing |

The model is set in [agent.py](agent.py).

## Usage

```bash
python main.py
```

This runs nine demo queries and prints the prediction, confidence, final route and response. Logs are also written to `router_agent.log`.

To call it from code:

```python
from main import ask_agent

result = ask_agent("I was charged twice for my subscription.")
```

## Files

- [agent.py](agent.py): LLM setup, router, specialist agents, graph
- [main.py](main.py): logging, `ask_agent()` and demo queries

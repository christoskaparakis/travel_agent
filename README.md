# Travel Assistant Agent

An LLM-powered assistant for travel planning.

## Project layout

```text
src/travel_agent/
	agent.py       Agent orchestration
	prompts.py     System and task prompts
	guardrails.py  Input/output validation and safety rules
tests/           Automated tests
```

The LLM provider and framework are intentionally not selected yet. Add their
runtime dependencies to `pyproject.toml` when the agent implementation is
provided.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Copy `.env.example` to `.env` and fill in the values required by the chosen
provider. Do not commit `.env` or API keys.

## Run tests

```bash
pytest
```

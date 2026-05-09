# SHL Assessment Recommender

Conversational SHL Assessment Recommender for the SHL AI Intern assignment.

## Features

- FastAPI backend
- GET /health endpoint
- POST /chat endpoint
- Stateless conversation handling
- Vague-query clarification
- Catalog-grounded recommendation
- Refinement using full conversation history
- Assessment comparison
- Off-topic and prompt-injection refusal

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

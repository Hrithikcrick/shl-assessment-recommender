# Approach Document: Conversational SHL Assessment Recommender

## 1. Problem Understanding

The goal is to build a stateless conversational API that recommends SHL Individual Test Solutions. The agent clarifies vague requests, recommends a shortlist when enough information is available, refines recommendations when the user changes constraints, compares assessments, and refuses out-of-scope requests.

## 2. Catalog Processing

I used the SHL product catalog JSON and cleaned it into a valid JSON file. Each assessment is normalized into common fields: name, URL, description, job levels, languages, duration, remote/adaptive status, and test type. The system only returns URLs from the SHL catalog.

## 3. Retrieval Setup

I used TF-IDF retrieval over normalized catalog metadata. Each user message in the conversation history is combined into one query because the API is stateless and every POST /chat call contains the full conversation. This allows refinements such as “Actually add personality tests” to update the shortlist instead of starting over.

## 4. Agent Logic

The agent first checks for prompt injection and off-topic requests. Then it checks whether the query is too vague. If vague, it asks for role and skills. If enough information is available, it retrieves the top matching SHL assessments. Keyword boosts improve matches for skills like Java, Python, SQL, communication, personality, and cognitive ability. For comparison requests, it identifies relevant catalog assessments and compares catalog fields only.

## 5. Evaluation

I tested schema compliance, health check, vague-query clarification, recommendation generation, refinement behavior, comparison behavior, off-topic refusal, and response speed.

## 6. What Did Not Work

A fully LLM-based recommender was avoided because it may hallucinate assessment names or URLs. A deterministic catalog-grounded retrieval system is safer, faster, easier to debug, and easier to defend in a technical interview.

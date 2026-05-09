import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CATALOG_PATH = Path("data/shl_product_catalog_clean.json")


OFF_TOPIC_WORDS = [
    "salary", "ctc", "compensation", "legal", "law", "lawsuit",
    "visa", "termination", "fire employee", "firing", "resume",
    "cover letter", "interview questions", "general hiring advice"
]

PROMPT_INJECTION_WORDS = [
    "ignore previous", "ignore all previous", "system prompt",
    "developer message", "jailbreak", "act as", "reveal prompt",
    "forget your instructions"
]

ROLE_SKILL_WORDS = [
    "java", "python", "sql", "javascript", "developer", "engineer",
    "manager", "sales", "analyst", "data", "customer", "support",
    "leadership", "communication", "stakeholder", "graduate",
    "entry-level", "entry level", "mid", "senior", "personality",
    "cognitive", "ability", "english", "coding", "programming",
    "finance", "accounting", "banking", "operations", "call center"
]


KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S"
}


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(textify(x) for x in value)
    if isinstance(value, dict):
        return " ".join(textify(v) for v in value.values())
    return str(value)


def convert_keys_to_test_type(keys: Any) -> str:
    if not isinstance(keys, list):
        return ""

    codes = []
    for key in keys:
        code = KEY_TO_CODE.get(str(key).strip())
        if code and code not in codes:
            codes.append(code)

    return ", ".join(codes)


class SHLRecommender:
    def __init__(self):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            raw_catalog = json.load(f)

        self.catalog = []

        for item in raw_catalog:
            name = item.get("name", "").strip()
            url = item.get("link", "").strip()

            if not name or not url:
                continue

            if "shl.com" not in url.lower():
                continue

            keys = item.get("keys", [])
            test_type = convert_keys_to_test_type(keys)

            normalized = {
                "name": name,
                "url": url,
                "test_type": test_type,
                "description": textify(item.get("description", "")),
                "job_levels": textify(item.get("job_levels", "")),
                "languages": textify(item.get("languages", "")),
                "duration": textify(item.get("duration", "")),
                "remote": textify(item.get("remote", "")),
                "adaptive": textify(item.get("adaptive", "")),
                "keys": textify(keys),
                "all_text": textify(item)
            }

            self.catalog.append(normalized)

        if not self.catalog:
            raise RuntimeError("No valid catalog items found.")

        self.documents = []
        for item in self.catalog:
            doc = f"""
            {item['name']}
            {item['description']}
            {item['job_levels']}
            {item['languages']}
            {item['duration']}
            {item['remote']}
            {item['adaptive']}
            {item['keys']}
            {item['all_text']}
            """
            self.documents.append(doc)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def latest_user_message(self, messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def build_query_from_history(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
        return "\n".join(parts).strip()

    def is_prompt_injection(self, text: str) -> bool:
        low = text.lower()
        return any(word in low for word in PROMPT_INJECTION_WORDS)

    def is_off_topic(self, text: str) -> bool:
        low = text.lower()
        return any(word in low for word in OFF_TOPIC_WORDS)

    def is_too_vague(self, text: str) -> bool:
        low = text.lower().strip()

        vague_phrases = [
            "i need an assessment",
            "need an assessment",
            "need assessment",
            "recommend assessment",
            "recommend an assessment",
            "suggest assessment",
            "suggest an assessment",
            "assessment please",
            "i want a test",
            "recommend test"
        ]

        if low in vague_phrases:
            return True

        words = re.findall(r"[a-zA-Z]+", low)
        has_role_or_skill = any(word in low for word in ROLE_SKILL_WORDS)

        if len(words) <= 5 and not has_role_or_skill:
            return True

        return False

    def keyword_boost(self, query: str, item: Dict[str, str]) -> float:
        q = query.lower()
        text = (item["name"] + " " + item["all_text"]).lower()
        name = item["name"].lower()
        keys = item["keys"].lower()

        boost = 0.0

        important_terms = [
            "java", "python", "sql", "javascript", "sales", "manager",
            "personality", "cognitive", "ability", "english",
            "communication", "leadership", "graduate", "coding",
            "programming", "data", "analyst", "customer", "support",
            "stakeholder", "finance", "accounting"
        ]

        for term in important_terms:
            if term in q and term in text:
                boost += 0.08
            if term in q and term in name:
                boost += 0.15

        if "personality" in q and "personality" in keys:
            boost += 0.30

        if ("cognitive" in q or "ability" in q or "aptitude" in q) and "ability" in keys:
            boost += 0.25

        if ("coding" in q or "programming" in q or "developer" in q) and (
            "java" in text or "python" in text or "coding" in text or "programming" in text
        ):
            boost += 0.25

        return boost

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()

        final_scores = []
        for i, item in enumerate(self.catalog):
            final_scores.append(scores[i] + self.keyword_boost(query, item))

        ranked_indices = np.argsort(final_scores)[::-1]

        results = []
        seen = set()

        for idx in ranked_indices:
            item = self.catalog[int(idx)]

            if item["name"].lower() in seen:
                continue

            seen.add(item["name"].lower())

            results.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"]
            })

            if len(results) >= top_k:
                break

        return results

    def compare(self, text: str) -> str:
        low = text.lower()
        found = []

        for item in self.catalog:
            name = item["name"]
            if name.lower() in low:
                found.append(item)

        if len(found) < 2:
            candidates = self.search(text, top_k=2)
            found = []
            for rec in candidates:
                for item in self.catalog:
                    if item["name"] == rec["name"]:
                        found.append(item)
                        break

        if len(found) < 2:
            return "I can compare assessments only when I can identify two SHL assessment names from the catalog."

        a = found[0]
        b = found[1]

        return (
            f"Here is a catalog-grounded comparison:\n\n"
            f"1. {a['name']}\n"
            f"- Test type: {a['test_type'] or 'Not specified'}\n"
            f"- Duration: {a['duration'] or 'Not specified'}\n"
            f"- Description: {a['description'] or 'Not specified'}\n"
            f"- URL: {a['url']}\n\n"
            f"2. {b['name']}\n"
            f"- Test type: {b['test_type'] or 'Not specified'}\n"
            f"- Duration: {b['duration'] or 'Not specified'}\n"
            f"- Description: {b['description'] or 'Not specified'}\n"
            f"- URL: {b['url']}"
        )

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        latest = self.latest_user_message(messages)
        full_query = self.build_query_from_history(messages)

        if not latest:
            return {
                "reply": "Please tell me what role or skills you want to assess using SHL assessments.",
                "recommendations": [],
                "end_of_conversation": False
            }

        if self.is_prompt_injection(latest):
            return {
                "reply": "I can only help with SHL assessment recommendations using the SHL catalog.",
                "recommendations": [],
                "end_of_conversation": False
            }

        if self.is_off_topic(latest):
            return {
                "reply": "I can only discuss SHL assessments and recommend assessments from the SHL catalog.",
                "recommendations": [],
                "end_of_conversation": False
            }

        low_latest = latest.lower()

        if "compare" in low_latest or "difference between" in low_latest or "different between" in low_latest:
            return {
                "reply": self.compare(latest),
                "recommendations": [],
                "end_of_conversation": False
            }

        if self.is_too_vague(full_query):
            return {
                "reply": "Sure. What role are you hiring for, and what skills or traits do you want to assess?",
                "recommendations": [],
                "end_of_conversation": False
            }

        recs = self.search(full_query, top_k=10)

        return {
            "reply": f"Based on your requirements, here are {len(recs)} SHL assessments from the catalog that best match the role.",
            "recommendations": recs,
            "end_of_conversation": False
        }

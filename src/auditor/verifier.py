"""Verifies individual claims against independent web search evidence."""

import json
import os
from dataclasses import dataclass, field

import serpapi
from groq import Groq

MODEL = "openai/gpt-oss-120b"

VERIFY_PROMPT = """You are fact-checking one claim using search result snippets.

Claim: "{claim}"

Search result snippets (from independent sources, not AI-generated):
{snippets_text}

Decide one of: "supported", "contradicted", "mixed", "insufficient_evidence".
- supported: the snippets clearly back up the claim
- contradicted: the snippets clearly disagree with the claim
- mixed: some snippets support it, some contradict it
- insufficient_evidence: the snippets don't address the claim, or the claim is too
  vague/general to check (e.g. a general statement rather than a specific fact)

Return ONLY a JSON object with this exact shape, nothing else:
{{"verdict": "...", "reason": "one short sentence explaining why", "best_source": "a URL from the snippets, or empty string"}}
"""


@dataclass
class VerificationResult:
    claim: str
    verdict: str
    reason: str
    best_source: str
    error: str | None = None


def _search_for_claim(client: serpapi.Client, claim: str) -> list[dict]:
    """Run a plain Google search for a claim and return simplified snippets."""
    res = client.search(
        {
            "engine": "google",
            "q": claim,
            "num": 5,
        }
    ).as_dict()

    organic = res.get("organic_results", [])
    snippets = []
    for item in organic[:5]:
        if item.get("title") and item.get("snippet") and item.get("link"):
            snippets.append(
                {
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "link": item["link"],
                }
            )
    return snippets


def verify_claim(
    claim: str,
    serpapi_client: serpapi.Client,
    groq_api_key: str | None = None,
) -> VerificationResult:
    """Search for evidence on one claim, then ask the LLM for a verdict."""
    try:
        snippets = _search_for_claim(serpapi_client, claim)
    except Exception as e:
        return VerificationResult(
            claim=claim, verdict="insufficient_evidence", reason="", best_source="", error=str(e)
        )

    if not snippets:
        return VerificationResult(
            claim=claim,
            verdict="insufficient_evidence",
            reason="No search results found.",
            best_source="",
        )

    snippets_text = "\n\n".join(
        f"[{s['link']}]\n{s['title']}\n{s['snippet']}" for s in snippets
    )
    prompt = VERIFY_PROMPT.format(claim=claim, snippets_text=snippets_text)

    key = groq_api_key or os.environ["GROQ_API_KEY"]
    groq_client = Groq(api_key=key)

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return VerificationResult(
            claim=claim,
            verdict=data.get("verdict", "insufficient_evidence"),
            reason=data.get("reason", ""),
            best_source=data.get("best_source", ""),
        )
    except Exception as e:
        return VerificationResult(
            claim=claim, verdict="insufficient_evidence", reason="", best_source="", error=str(e)
        )
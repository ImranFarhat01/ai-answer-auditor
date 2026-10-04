"""Combines verified claims from multiple engines into one consensus answer."""

import json
import os
from dataclasses import dataclass, field

from groq import Groq

from auditor.verifier import VerificationResult

MODEL = "openai/gpt-oss-120b"

ARBITER_PROMPT = """You are producing a final, audited answer to a question by combining
fact-checked claims from multiple AI search engines.

Original question: "{query}"

Below are claims from each engine, each with a verification verdict based on independent
search evidence:

{claims_text}

Based ONLY on the claims marked "supported" or "mixed" (lean toward what the evidence
backs), write a final consensus answer. If the claims are too contradictory or too many
are "contradicted" or "insufficient_evidence", say you cannot give a confident answer.

Return ONLY a JSON object with this exact shape, nothing else:
{{
  "consensus_answer": "a short, clear answer in plain language",
  "confidence": "high" or "medium" or "low",
  "agreements": ["claim that multiple engines/evidence support", "..."],
  "disagreements": ["a short description of any conflict found", "..."],
  "abstained": true or false
}}

Set "abstained" to true only if you cannot give any confident answer at all.
"""


@dataclass
class ArbiterResult:
    query: str
    consensus_answer: str
    confidence: str
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    abstained: bool = False
    error: str | None = None


def _format_claims_for_prompt(engine_results: dict[str, list[VerificationResult]]) -> str:
    """Turn {engine_name: [VerificationResult, ...]} into readable text for the prompt."""
    lines = []
    for engine, results in engine_results.items():
        lines.append(f"--- {engine} ---")
        for r in results:
            lines.append(f'Claim: "{r.claim}"')
            lines.append(f"Verdict: {r.verdict} ({r.reason})")
            lines.append("")
    return "\n".join(lines)


def arbitrate(
    query: str,
    engine_results: dict[str, list[VerificationResult]],
    api_key: str | None = None,
) -> ArbiterResult:
    """Combine verified claims from all engines into one final consensus."""
    claims_text = _format_claims_for_prompt(engine_results)
    if not claims_text.strip():
        return ArbiterResult(
            query=query,
            consensus_answer="No claims were available to verify.",
            confidence="low",
            abstained=True,
        )

    prompt = ARBITER_PROMPT.format(query=query, claims_text=claims_text)
    key = api_key or os.environ["GROQ_API_KEY"]
    client = Groq(api_key=key)

    try:
        response = client.chat.completions.create(
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
        return ArbiterResult(
            query=query,
            consensus_answer=data.get("consensus_answer", ""),
            confidence=data.get("confidence", "low"),
            agreements=data.get("agreements", []),
            disagreements=data.get("disagreements", []),
            abstained=data.get("abstained", False),
        )
    except Exception as e:
        return ArbiterResult(
            query=query,
            consensus_answer="",
            confidence="low",
            abstained=True,
            error=str(e),
        )
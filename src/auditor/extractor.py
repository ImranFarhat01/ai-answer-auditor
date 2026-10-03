"""Extracts factual, checkable claims from an engine's answer text blocks."""

import json
import os
from dataclasses import dataclass

from groq import Groq

MODEL = "openai/gpt-oss-120b"

EXTRACTION_PROMPT = """You are extracting factual claims from an AI search engine's answer.

Below is the answer text, broken into blocks. Extract a list of specific, checkable
factual claims (numbers, dates, named facts, yes/no determinations). Ignore:
- conversational filler ("would you like to know more?")
- vague statements with nothing specific to check
- questions

Return ONLY a JSON array of strings, each one a single claim, nothing else.
Example output: ["GST on individual health insurance is 0% as of 22 September 2025.", "Group health insurance is taxed at 18%."]

Answer text blocks:
{blocks_text}
"""


@dataclass
class ExtractedClaims:
    engine: str
    claims: list[str]
    error: str | None = None


def _flatten_text_blocks(text_blocks: list) -> str:
    """Turn the nested text_blocks structure into plain readable text."""
    lines = []

    def walk(block):
        if isinstance(block, dict):
            if "snippet" in block:
                lines.append(block["snippet"])
            if "list" in block:
                for item in block["list"]:
                    walk(item)
        elif isinstance(block, list):
            for item in block:
                walk(item)

    for block in text_blocks:
        walk(block)

    return "\n".join(lines)


def extract_claims(engine: str, text_blocks: list, api_key: str | None = None) -> ExtractedClaims:
    """Ask the LLM to pull factual claims out of one engine's answer."""
    key = api_key or os.environ["GROQ_API_KEY"]
    client = Groq(api_key=key)

    blocks_text = _flatten_text_blocks(text_blocks)
    if not blocks_text.strip():
        return ExtractedClaims(engine=engine, claims=[])

    prompt = EXTRACTION_PROMPT.format(blocks_text=blocks_text)

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
        claims = json.loads(raw)
        if not isinstance(claims, list):
            raise ValueError("Expected a JSON array")
        return ExtractedClaims(engine=engine, claims=[str(c) for c in claims])
    except Exception as e:
        return ExtractedClaims(engine=engine, claims=[], error=str(e))
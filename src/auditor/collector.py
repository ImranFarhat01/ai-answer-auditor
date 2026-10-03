"""Collects answers from multiple AI search engines for a single query."""

import os
import re
from dataclasses import dataclass, field

import serpapi

ENGINES = ["google_ai_mode", "bing_copilot"]


@dataclass
class Reference:
    index: int
    title: str
    link: str
    source: str


@dataclass
class EngineAnswer:
    engine: str
    query: str
    header: str
    text_blocks: list = field(default_factory=list)
    references: list = field(default_factory=list)
    error: str | None = None


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:50]


def _parse_references(raw_refs: list) -> list[Reference]:
    refs = []
    for r in raw_refs:
        refs.append(
            Reference(
                index=r.get("index", -1),
                title=r.get("title", ""),
                link=r.get("link", ""),
                source=r.get("source", ""),
            )
        )
    return refs


def fetch_engine_answer(client: serpapi.Client, engine: str, query: str) -> EngineAnswer:
    """Call one AI engine for one query and return a normalized result."""
    try:
        res = client.search({"engine": engine, "q": query}).as_dict()
    except Exception as e:
        return EngineAnswer(engine=engine, query=query, header="", error=str(e))

    header = res.get("header", "")
    if not header and res.get("text_blocks"):
        first = res["text_blocks"][0]
        header = first.get("snippet", "")

    return EngineAnswer(
        engine=engine,
        query=query,
        header=header,
        text_blocks=res.get("text_blocks", []),
        references=_parse_references(res.get("references", [])),
    )

def fetch_all_engines(query: str, api_key: str | None = None) -> list[EngineAnswer]:
    """Call every engine in ENGINES for one query, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    key = api_key or os.environ["SERPAPI_API_KEY"]
    client = serpapi.Client(api_key=key, timeout=60)

    with ThreadPoolExecutor(max_workers=len(ENGINES)) as pool:
        futures = [
            pool.submit(fetch_engine_answer, client, engine, query)
            for engine in ENGINES
        ]
        return [f.result() for f in futures]
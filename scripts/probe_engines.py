import json
import os
import pathlib
import re

from dotenv import load_dotenv
import serpapi

load_dotenv()
client = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"], timeout=60)

ENGINES = ["google_ai_mode", "bing_copilot"]
QUERIES = [
    "What is the current repo rate set by the Reserve Bank of India?",
]

out_dir = pathlib.Path("fixtures/raw")
out_dir.mkdir(parents=True, exist_ok=True)


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:50]


for engine in ENGINES:
    for q in QUERIES:
        slug = slugify(q)
        path = out_dir / f"{engine}__{slug}.json"
        if path.exists():
            print("skip (already saved):", path)
            continue
        try:
            res = client.search({"engine": engine, "q": q})
        except Exception as e:
            print("FAILED:", engine, "|", q, "|", e)
            continue
        path.write_text(
            json.dumps(res.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("saved:", path)
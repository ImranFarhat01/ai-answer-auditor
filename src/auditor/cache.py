"""Caches SerpApi and Groq responses to disk so repeated runs cost no credits."""

import hashlib
import json
import pathlib
from functools import wraps

CACHE_DIR = pathlib.Path("fixtures/cache")


def _cache_key(*args, **kwargs) -> str:
    """Build a stable filename from the function's arguments."""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def cached(subfolder: str):
    """Decorator: caches a function's return value to disk, keyed by its arguments.

    The wrapped function must return something that can be converted to/from a
    dict via dataclasses.asdict, or a plain JSON-safe value.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            folder = CACHE_DIR / subfolder
            folder.mkdir(parents=True, exist_ok=True)
            key = _cache_key(*args, **kwargs)
            path = folder / f"{key}.json"

            if path.exists():
                cached_data = json.loads(path.read_text(encoding="utf-8"))
                return cached_data

            result = func(*args, **kwargs)
            path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
            return result

        return wrapper

    return decorator
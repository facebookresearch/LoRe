"""Content-addressed cache of raw judge responses.

The key is a sha256 over the *inputs* that determine a completion --
``(model_id, temperature, concept_set_version, order, prompt, answer_a, answer_b)`` --
NOT over the rendered prompt text. This is a deliberate, sharp trade-off:

  * Reordering or rewording prompt.py does NOT invalidate the cache. If you change
    what the judge is actually asked, you must bump CONCEPT_SET_VERSION (or clear the
    cache) or you will silently reuse stale answers.
  * Changing a concept definition IS captured, because definitions live behind
    CONCEPT_SET_VERSION.

We store only the model's raw text (plus any API error); parsing is deterministic and
re-run on read, so improving the parser does not require re-calling the API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedResponse:
    raw_text: str
    error: str | None


def compute_key(
    *,
    model_id: str,
    temperature: float | None,
    concept_set_version: str,
    order: str,
    prompt: str,
    answer_a: str,
    answer_b: str,
) -> str:
    """Deterministic sha256 hex over the completion-determining inputs."""
    payload = {
        "model_id": model_id,
        "temperature": temperature,
        "concept_set_version": concept_set_version,
        "order": order,
        "prompt": prompt,
        "answer_a": answer_a,
        "answer_b": answer_b,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """A tiny on-disk cache: one JSON file per key under ``root``.

    ``enabled=False`` turns it into a no-op (every get misses, puts are dropped) so a
    run can bypass caching without changing call sites.
    """

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return CachedResponse(raw_text=data.get("raw_text", ""), error=data.get("error"))

    def put(self, key: str, raw_text: str, error: str | None) -> None:
        if not self.enabled:
            return
        # Never cache a failed API call: it is transient, and caching it would pin the
        # failure until the cache is cleared.
        if error is not None:
            return
        tmp = self._path(key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"raw_text": raw_text, "error": error}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self._path(key))

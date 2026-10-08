"""Shared on-disk registry so repeated eval/generation runs reuse a live Vertex CachedContent
instead of re-uploading and re-caching identical video+text context every time.

Vertex's caches.create() has no content-addressed lookup of its own — every call always makes a
brand-new object, even if an identical one (same model, video, text) was created minutes ago and
is still live. Each script computes a cache_key that captures everything that affects the cached
content (model, video encoding params + source file identity, text hashes, and which flags/paths
led here), so a genuinely different run gets a different key and a matching one gets reused.
"""

import json
import time
from pathlib import Path


def _registry_path(dataset_dir: Path) -> Path:
    return dataset_dir / ".vertex_cache_registry.json"


def _load(dataset_dir: Path) -> dict:
    p = _registry_path(dataset_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(dataset_dir: Path, registry: dict) -> None:
    _registry_path(dataset_dir).write_text(json.dumps(registry, indent=2))


def get_or_create(client, types, dataset_dir: Path, cache_key: str, model: str,
                   ttl_seconds: int, build_contents, safety_margin_seconds: int = 120):
    """Reuse a live cache registered under cache_key if the server confirms it still exists and
    isn't about to expire; otherwise build fresh contents (build_contents: () -> list[Part],
    called lazily so clip extraction/encoding is skipped entirely on a cache hit) and create a
    new CachedContent, recording it for next time.

    Returns (cache_name, reused: bool).
    """
    registry = _load(dataset_dir)
    entry = registry.get(cache_key)
    if entry is not None:
        try:
            cache = client.caches.get(name=entry["name"])
            expire_ts = cache.expire_time.timestamp() if cache.expire_time else 0
            if expire_ts - safety_margin_seconds > time.time():
                return entry["name"], True
        except Exception:
            pass  # deleted / expired / otherwise gone server-side — fall through and recreate

    contents = build_contents()
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(contents=contents, ttl=f"{ttl_seconds}s"),
    )
    registry[cache_key] = {"name": cache.name, "created_at": time.time(), "cache_key": cache_key}
    _save(dataset_dir, registry)
    return cache.name, False


def encode_under_size_limit(extract_fn, initial_crf: int, max_bytes: int = 9_500_000,
                             crf_step: int = 4, max_attempts: int = 6):
    """Vertex's inline CachedContent has a hard size ceiling (~10MB) with no per-project way to
    raise it (the GCS file_uri alternative the API suggests isn't available to Express service
    accounts). A fixed CRF that fits one dataset's video can overshoot on a longer one (e.g.
    session01's ~43min video fit at crf 32; session02's ~71min video didn't), so re-encode at a
    higher CRF (smaller file, same fps/resolution) until it fits rather than hand-tuning per
    dataset. extract_fn(crf) -> bytes; called repeatedly, so each attempt re-runs ffmpeg.

    Returns (bytes, crf_used).
    """
    crf = initial_crf
    data = None
    for attempt in range(max_attempts):
        data = extract_fn(crf)
        if len(data) <= max_bytes:
            if attempt > 0:
                print(f"  clip was over the {max_bytes/1e6:.1f}MB cache limit at crf={initial_crf}; "
                      f"re-encoded at crf={crf} ({len(data)/1e6:.1f}MB)")
            return data, crf
        if attempt < max_attempts - 1:
            crf += crf_step
    print(f"  WARNING: clip still {len(data)/1e6:.1f}MB after {max_attempts} attempts up to "
          f"crf={crf}, over the {max_bytes/1e6:.1f}MB limit — cache creation will likely fail")
    return data, crf


def register_existing(dataset_dir: Path, cache_key: str, cache_name: str) -> None:
    """Retroactively record an already-created cache under cache_key, so a run already in
    flight (or one just finished) gets its cache reused by the next matching run instead of
    orphaned until TTL expiry."""
    registry = _load(dataset_dir)
    registry[cache_key] = {"name": cache_name, "created_at": time.time(), "cache_key": cache_key}
    _save(dataset_dir, registry)

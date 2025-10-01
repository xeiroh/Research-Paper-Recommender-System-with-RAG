import os
import asyncio
import time
import numpy as np
from openai import OpenAI, AsyncOpenAI
from app import settings
from fastapi import FastAPI, HTTPException
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATAPATH = settings.data_dir
CACHE_PATH = settings.cache_dir
if not os.path.exists(CACHE_PATH):
	os.makedirs(CACHE_PATH, exist_ok=True)

def _chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

batch_size = 350  # texts per request
rpm = 200        # requests per minute budget (e.g., 2500)
min_rpm = 50     # minimum RPM allowed
max_rpm = 500    # maximum RPM allowed (if None, set to rpm)
max_retries = 5   # retries on transient failures with exponential backoff

async def create_embeddings_async(data, model="text-embedding-3-small", use_cache=True, batch_size=batch_size, rpm=rpm, min_rpm=min_rpm, max_rpm=max_rpm, max_retries=max_retries):
    """
    Async embedding creator with batching and a simple RPM limiter.
    - `batch_size`: number of texts per request
    - `rpm`: requests per minute budget (e.g., 2500)
    - `min_rpm`: minimum RPM allowed
    - `max_rpm`: maximum RPM allowed (if None, set to rpm)
    - Retries on transient failures with exponential backoff.
    """
    print("Creating embeddings with OpenAI (async, batched)...")
    cache_path = os.path.join(CACHE_PATH, f"openai_{model.replace('-', '_')}.npy")
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=getattr(settings, 'openai_base_url', None) or None)
    texts = data["content"].tolist()

    if max_rpm is None:
        max_rpm = rpm
    current_rpm = max(min_rpm, min(rpm, max_rpm))

    def _interval(rpm):
        return 60.0 / max(1, rpm)

    interval = _interval(current_rpm)
    last_request_time = 0.0
    success_streak = 0
    fail_streak = 0

    def _is_rate_limit(err):
        if getattr(err, "status_code", None) == 429:
            return True
        msg = str(err).lower()
        if "429" in msg or "too many requests" in msg or "rate limit" in msg:
            return True
        return False

    def _retry_after_seconds(err):
        try:
            hdrs = getattr(getattr(err, "response", None), "headers", {}) or {}
            ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
            return float(ra) if ra is not None else None
        except Exception:
            return None

    results = []

    batches = list(_chunks(texts, batch_size))
    pbar = tqdm(batches, desc="Creating embeddings", unit="batch")
    for batch in pbar:
        attempt = 0
        while True:
            now = time.time()
            wait_for = max(0.0, last_request_time + interval - now)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            try:
                resp = await client.embeddings.create(model=model, input=batch)
                batch_vecs = [item.embedding for item in resp.data]
                results.extend(batch_vecs)
                last_request_time = time.time()
                success_streak += 1
                fail_streak = 0
                if success_streak % 8 == 0:
                    current_rpm = min(max_rpm, int(current_rpm * 1.10))
                    interval = _interval(current_rpm)
                pbar.set_postfix({"rpm": current_rpm})
                pbar.refresh()
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                server_wait = _retry_after_seconds(e)
                base = 0.5 * (2 ** (attempt - 1))  # 0.5,1,2,4,...
                if _is_rate_limit(e):
                    fail_streak += 1
                    success_streak = 0
                    # harsher downshift with floor
                    current_rpm = max(min_rpm, int(current_rpm * 0.60))
                    interval = _interval(current_rpm)
                backoff = server_wait if server_wait is not None else base
                backoff = min(6.0, backoff) * (0.9 + 0.2 * np.random.rand())  # cap and jitter
                await asyncio.sleep(backoff)

    embeddings = np.asarray(results, dtype=np.float32, order="C")
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")
    return embeddings

def create_embeddings(data, model="text-embedding-3-small", use_cache=True, batch_size=batch_size, rpm=rpm, min_rpm=min_rpm, max_rpm=max_rpm, max_retries=max_retries):
    """Sync wrapper that runs the async embedding creator."""
    return asyncio.run(create_embeddings_async(
        data=data,
        model=model,
        use_cache=use_cache,
        batch_size=batch_size,
        rpm=rpm,
        min_rpm=min_rpm,
        max_rpm=max_rpm,
        max_retries=max_retries,
    ))
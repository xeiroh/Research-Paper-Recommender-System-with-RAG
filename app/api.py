import os
import time
import numpy as np
from openai import OpenAI
from app import settings
from fastapi import FastAPI, HTTPException
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATAPATH = settings.data_dir
CACHE_PATH = settings.cache_dir
API_KEY = settings.openai_api_key

if not os.path.exists(CACHE_PATH):
	os.makedirs(CACHE_PATH, exist_ok=True)

def _chunks(lst, size):
	for i in range(0, len(lst), size):
		yield lst[i:i+size]

tpm_limit = 1000000 # tokens per minute limit for your OpenAI account
batch_size = 400  # texts per request (fast but still sane)
rpm = tpm_limit//(200 * batch_size)         # fixed requests/minute; tune based on your account
max_retries = 2   # small, constant retry count
embed_model = "text-embedding-3-small"

def get_client():
	return OpenAI(api_key=API_KEY, base_url=getattr(settings, 'openai_base_url', None) or None, timeout=120.0)

def create_embeddings(data, model=embed_model, use_cache=True, batch_size=batch_size, rpm=rpm, max_retries=max_retries):
	"""
	Sync embedding creator with batching and a simple fixed RPM limiter.
	- `batch_size`: number of texts per request
	- `rpm`: fixed requests per minute budget
	- Retries on transient failures with fixed delay.
	"""
	print("Creating embeddings with OpenAI (sync, batched)...")
	cache_path = os.path.join(CACHE_PATH, f"openai_{model.replace('-', '_')}.npy")
	embeddings, start = None, 0
	if use_cache and os.path.exists(cache_path):
		embeddings = np.load(cache_path)
		start = embeddings.shape[0]
		if start >= len(data):
			print(f"Loaded {len(embeddings)} cached embeddings from {cache_path}")
			return embeddings
		print(f"Resuming from {start} cached embeddings from {cache_path}")


	client = OpenAI(api_key=API_KEY, base_url=getattr(settings, 'openai_base_url', None) or None, timeout=120.0)
	texts = data["content"].tolist()

	def _interval(rpm):
		return 60.0 / max(1, rpm)

	interval = _interval(rpm)
	last_request_time = 0.0

	def _retry_after_seconds(err):
		try:
			hdrs = getattr(getattr(err, "response", None), "headers", {}) or {}
			ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
			return float(ra) if ra is not None else None
		except Exception:
			return None

	results = []
	if embeddings is not None and start > 0:
		results.extend(embeddings[:start].tolist())
		texts = texts[start:]

	batches = list(_chunks(texts, batch_size))
	pbar = tqdm(batches, desc="Creating embeddings", unit="batch")
	for batch in pbar:
		attempt = 0
		while True:
			now = time.time()
			wait_for = max(0.0, last_request_time + interval - now)
			if wait_for > 0:
				time.sleep(wait_for)
			try:
				resp = client.embeddings.create(model=model, input=batch)
				batch_vecs = [item.embedding for item in resp.data]
				results.extend(batch_vecs)
				if len(results) // batch_size % 10 == 0:
					tmp = np.asarray(results, dtype=np.float32, order="C")
					np.save(cache_path, tmp)
					print(f"Checkpoint saved after {len(results)//batch_size} requests")
				last_request_time = time.time()
				break
			except Exception as e:
				attempt += 1
				if attempt > max_retries:
					raise
				server_wait = _retry_after_seconds(e)
				fixed = server_wait if server_wait is not None else 2.0
				time.sleep(min(3, fixed))

	embeddings = np.asarray(results, dtype=np.float32, order="C")
	np.save(cache_path, embeddings)
	print(f"Saved embeddings to {cache_path}")
	return embeddings


def get_query_embedding(query, model=embed_model, api_key=API_KEY):
	client = OpenAI(api_key=api_key, base_url=getattr(settings, 'openai_base_url', None) or None, timeout=60.0)
	resp = client.embeddings.create(model=model, input=query)
	return np.array(resp.data[0].embedding, dtype=np.float32)

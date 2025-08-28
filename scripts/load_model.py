import pandas as pd
import os
import numpy as np
import faiss
from app import settings
from app.api import create_embeddings as create_openai_embeddings
from app.similarity_search import get_faiss_index as create_faiss_index
# from fastembed import TextEmbedding

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(ROOT, "data")
APP_PATH = os.path.join(ROOT, "app")

CACHE_PATH = settings.cache_dir
if not os.path.exists(CACHE_PATH):
	os.makedirs(CACHE_PATH, exist_ok=True)


def load_data(file_path=DATA_PATH+"paperswithcode.parquet"):
	"""Load data from a JSON file and create a `content` column."""
	print(f"Loading data from {file_path}...")
	data = pd.read_parquet(file_path)
	return data

def create_embeddings(data, model=settings.openai_embed_model, use_cache=True):
	return create_openai_embeddings(data, model=model, use_cache=use_cache)


if __name__ == "__main__":
	file_path = os.path.join(DATA_PATH, "paperswithcode.parquet")
	data = load_data(file_path)
	print(data.head())
	embeddings = create_embeddings(data)
	index = create_faiss_index(embeddings)
	print(index)

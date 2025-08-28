import os
from re import A
from IPython import embed
import numpy as np
import faiss
import pandas as pd
from app import settings
from scripts.quick_filter import FILE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = settings.data_dir
CACHE_PATH = settings.cache_dir
LOAD_PATH = os.path.join(DATA_PATH, "paperswithcode.parquet")

embedding_cache = "openai_text_embedding_3_small.npy"

EMBED_PATH = os.path.join(CACHE_PATH, embedding_cache)

def load_data():
	df = pd.read_parquet(LOAD_PATH)
	return df

def load_embeddings():
	embeddings = np.load(EMBED_PATH)
	return embeddings


def get_faiss_index(embeddings, use_cache=True, file_name="faiss_index.index"):
	"""Create a FAISS L2 index and add numpy float32 embeddings."""
	# (Optional) keep FAISS single-threaded on macOS/Apple Silicon
	# try:
	# 	faiss.omp_set_num_threads(1)
	# except Exception:
	# 	pass

	def load_faiss_index(index_path):
		if os.path.exists(index_path):
			try:
				index = faiss.read_index(index_path)
				return index
			except Exception as e:
				print(f"Error loading FAISS index from {index_path}: {e}")
		return None


	faiss_file = os.path.join(CACHE_PATH, file_name)
	index = load_faiss_index(faiss_file)
	if use_cache and index is not None:
		print(f"Loaded existing FAISS index with {index.ntotal} vectors.")
		return index

	print("No existing FAISS index found. Creating a new one...")

	d = embeddings.shape[1]
	index = faiss.IndexFlatL2(d)
	index.add(embeddings)

	# Save index to file
	faiss.write_index(index, faiss_file)
	print(
		f"FAISS index with {index.ntotal} vectors of dimension {d} created and saved to {os.path.join(CACHE_PATH, 'faiss_index.index')}"
	)

	return index

def get_lookup_table(index=None, filename="faiss_index.index"):
	df = load_data()
	embeddings = load_embeddings()
	if index is not None:
		return df, embeddings, index
	faiss_index = get_faiss_index(embeddings, file_name=filename)
	return df, embeddings, faiss_index
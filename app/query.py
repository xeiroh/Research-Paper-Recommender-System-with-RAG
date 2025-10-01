from app import settings
from app.api import get_query_embedding
from app.similarity_search import get_lookup_table
from app.mmr import maximal_marginal_relevance as mmr
from app.llm import llm_explain
import pandas as pd
import numpy as np
import os


def search(query: str, top_k: int = 5, index=None, filename="faiss_index.index", use_mmr=True, fetch_k = 25, llm=True, user=None, use_personalization=True):
	from app.users import personalize_scores, add_search_history

	q_embedding = get_query_embedding(query)
	df, embeddings, faiss_index = get_lookup_table(index=index, filename=filename)

	search_k = fetch_k if use_mmr else top_k
	if user and use_personalization:
		search_k = max(search_k, 50)

	D, I = faiss_index.search(np.array([q_embedding], dtype=np.float32), search_k)

	if user and use_personalization:
		candidate_embeddings = embeddings[I[0]]
		D[0] = personalize_scores(user, q_embedding, candidate_embeddings, D[0], df, embeddings, blend_weight=0.25)
		sorted_indices = np.argsort(D[0])
		I[0] = I[0][sorted_indices]
		D[0] = D[0][sorted_indices]

	if use_mmr:
		print("MMR Re-ranking Candidates...")
		selected_idx = mmr(q_embedding, embeddings[I[0]], top_k=top_k)
		I = I[:, selected_idx]

	results = []
	for idx in I[0]:
		row = df.iloc[int(idx)]
		item = {
			"title": row["title"],
			"abstract": row["abstract"],
			"url_pdf": row.get("url_pdf") if hasattr(row, "get") else row["url_pdf"],
			"paper_url": row.get("paper_url") if hasattr(row, "get") else row["paper_url"],
			"date": row.get("date") if hasattr(row, "get") else row["date"]
		}
		results.append(item)

	if user:
		add_search_history(user, query, results)

	explanation = None
	if llm:
		explanation = llm_explain(query, results)
	return results, df, I[0], explanation


def _test_search(query=None, use_mmr=True, filename="faiss_index.index"):
	if query is None:
		query = input("Enter a search query: ")
	results, df, indices, explanation = search(query, use_mmr=use_mmr, filename=filename)
	titles = [r["title"] for r in results]
	print("\n\n")
	for title in titles:
		print(f"{title}\n")
	print(f"\n\n{explanation}")
	# return results


# _test_search("LSTMs for time series forecasting")
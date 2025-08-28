import os
import streamlit as st
from typing import List, Dict
from app import settings, get_faiss_index
from app.query import search as search_papers
from app.get_pdf import get_pdf

try:
	from streamlit_pdf_viewer import pdf_viewer
except Exception as _e:
	pdf_viewer = None

INDEX_FILE = os.path.join(settings.cache_dir, "faiss_index.index")

st.set_page_config(page_title="Research Paper Recommender", layout="wide")
st.title("LLM-Powered Research Paper Recommender System")


with st.sidebar:
	st.header("Settings")
	top_k = st.slider("Number of Papers to Retrieve", min_value=1, max_value=20, value=5, step=1)
	use_mmr = st.checkbox("Use MMR Re-ranking", value=True)
	llm = st.checkbox("Use LLM for Explanation", value=True)
	enable_pdf = st.checkbox("Enable PDF Download to View on Query Page", value=True)
	 

query = st.text_input("Search Query", placeholder="e.g., graph neural networks for molecule property prediction")
go = st.button("Search")

def render_results(items: List[Dict], explanation):
	if not items:
		st.info("No results.")
		return

	if explanation:
		st.markdown(explanation)
		st.markdown("---")

	for i, item in enumerate(items):
		title = item.get("title", "(Untitled)")
		abstract = item.get("abstract", "(No abstract available)")
		url_pdf = item.get("url_pdf")
		paper_url = item.get("paper_url")
		date = str(item.get("date")).split()[0] if item.get("date") else "(No Date Available)"

		expander_header = f"### {i+1}. [{title}]({paper_url}) | {date}" if date else f"### {i+1}. [{title}]({paper_url})"
		with st.expander(expander_header):
			st.markdown(abstract)
			with st.expander("📄 View PDF"):
				if pdf_viewer is not None and url_pdf and enable_pdf:
					pdf_file = get_pdf(url_pdf)
					if pdf_file is None:
						st.error("Invalid PDF File; try the link instead.")
					else:
						pdf_viewer(pdf_file, width="50%", height="1200")
				else:
					st.warning("Install streamlit-pdf-viewer to view PDFs inline: pip install streamlit-pdf-viewer")
					if url_pdf:
						st.markdown(f"[Open PDF in new tab]({url_pdf})")

		st.markdown("---")

if go or query:
	try:
		index = get_faiss_index()
		with st.spinner("Searching..."):
			results, df, indices, explanation = search_papers(query, top_k=top_k, index=index, use_mmr=use_mmr, llm=llm)
		render_results(results, explanation)
	except Exception as e:
		st.error(f"Search failed: {e}")
		st.exception(e)

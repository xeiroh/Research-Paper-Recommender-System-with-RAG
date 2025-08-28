# Demo
https://github.com/user-attachments/assets/cc557e7c-4212-414a-b76a-58437d0772ec

# LLM-Powered Research Paper Recommender

An interactive system that helps researchers discover relevant academic papers using **retrieval-augmented generation (RAG)**, **vector search with FAISS**, and **MMR re-ranking** for diversity. The system integrates **LLMs for explainability**, **Personalization** for tailored results using embeddings + built user profiles, and provides a simple **Streamlit demo app**.

---

## Features
- **Personalization (Currently Fixing)**: Blends user preferences into retrieval scoring.  
- **Semantic Search**: Indexes 1M+ arXiv papers with embeddings for natural language queries.  
- **Vector Database (FAISS)**: Fast approximate nearest neighbor search.  
- **MMR Re-ranking**: Improves novelty and avoids redundancy in results.  
- **RAG with LLMs**: Provides concise rationales for why each paper was chosen.  
- **Streamlit UI**: Easy-to-use web demo for querying and exploring results.  
- **FastAPI Backend (In Progress)**: Modular design for deployment and API-based usage.  
- **Dockerized Deployment**: Container-ready for reproducibility and sharing.

---

## 🛠️ Tech Stack
- **Python 3.12**
- **FAISS** for vector similarity search
- **FastEmbed / OpenAI embeddings** for paper representation
- **Streamlit** for the demo UI
- **FastAPI** for backend endpoints
- **OpenAI GPT models** for explanations
- **Docker** for containerization

---

## 📂 Project Structure
```
Research-Paper-Recommender-System-with-RAG/
├── app/
│   ├── ui_app.py          # Streamlit frontend
│   ├── query.py           # Query Implementation
│   ├── __init__.py        # Project Initialization
│   ├── mmr.py             # MMR implementation
│   ├── api.py             # OpenAI API Handling
│   ├── llm.py             # LLM integration with GPT
│   ├── similarity_search.py # FAISS lookup logic
│   ├── get_pdf.py         # Download PDF (helper func) to display pdfs
│   └── users.py             # User Handling for Personalization
├── scripts/
│   ├── load_model.py      # Preprocessing & embeddings
│   └── build_index.py     # Build FAISS index
├── .cache/             # Cached embeddings & FAISS index
├── data/                  # Papers dataset (JSON/Parquet)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/xeiroh/Research-Paper-Recommender-System-with-RAG.git
cd Research-Paper-Recommender-System-with-RAG
pip install -r requirements.txt
```

### 2. Load FAISS Index 
- Downlood Papers With Code Dataset and place it in data/
```bash
python scripts/load_model.py
```

### Or download FAISS Index from my HuggingFace
https://huggingface.co/datasets/Xeiroh/PapersWithCode_FAISS_Index/tree/main
Place in .cache directory

```bash
streamlit run app/ui_app.py
```

## Usage
```
•	Enter a natural language query (e.g., “LSTMs vs Transformers for Medical Documentation”).
•	Get top-k papers ranked by semantic similarity.
•	Toggle MMR for diverse results.
•	Enable LLM explanations to see summaries and justifications.
•	Click PDF links to access full papers.
```

## In Progress
- Advanced personalization and user handling with search history and likes/dislikes to create "For You" page of papers.
- Live deployment on my domain for easy demo/testing.

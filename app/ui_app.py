#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__:
    from . import settings, get_faiss_index
    from .get_pdf import get_pdf
    from .query import search as search_papers
    from . import users
else:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app import settings, get_faiss_index
    from app.get_pdf import get_pdf
    from app.query import search as search_papers
    from app import users

import streamlit as st

try:
    from streamlit_pdf_viewer import pdf_viewer
except Exception as _e:
    pdf_viewer = None

INDEX_FILE = os.path.join(settings.cache_dir, "faiss_index.index")

st.set_page_config(
    page_title="Research Paper Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .paper-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .paper-meta {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .stButton>button {
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .like-button {
        background-color: transparent;
        border: none;
        cursor: pointer;
        font-size: 1.5rem;
        transition: transform 0.2s;
    }
    .like-button:hover {
        transform: scale(1.2);
    }
    div[data-testid="stSidebarNav"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .auth-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin-bottom: 1rem;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if "page" not in st.session_state:
    st.session_state.page = "search"

def login_page():
    st.markdown('<h1 class="main-header">Research Paper Recommender</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.subheader("Login to Your Account")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            if users.authenticate_user(login_username, login_password):
                st.session_state.authenticated = True
                st.session_state.username = login_username
                st.rerun()
            else:
                st.markdown('<div class="error-message">Invalid username or password</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.subheader("Create New Account")
        signup_username = st.text_input("Username", key="signup_username")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")

        if st.button("Sign Up", use_container_width=True):
            if not signup_username or not signup_password:
                st.markdown('<div class="error-message">Please fill in all fields</div>', unsafe_allow_html=True)
            elif signup_password != signup_password_confirm:
                st.markdown('<div class="error-message">Passwords do not match</div>', unsafe_allow_html=True)
            elif len(signup_password) < 6:
                st.markdown('<div class="error-message">Password must be at least 6 characters</div>', unsafe_allow_html=True)
            else:
                if users.create_user(signup_username, signup_password):
                    st.markdown('<div class="success-message">Account created successfully! Please login.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">Username already exists</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_paper_card(item: Dict, idx: int, show_like_button: bool = True):
    title = item.get("title", "(Untitled)")
    abstract = item.get("abstract", "(No abstract available)")
    url_pdf = item.get("url_pdf")
    paper_url = item.get("paper_url")
    date = str(item.get("date")).split()[0] if item.get("date") else None

    st.markdown('<div class="paper-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([0.95, 0.05])

    with col1:
        if paper_url:
            st.markdown(f'<div class="paper-title"><a href="{paper_url}" target="_blank">{idx}. {title}</a></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="paper-title">{idx}. {title}</div>', unsafe_allow_html=True)

        if date:
            st.markdown(f'<div class="paper-meta">📅 {date}</div>', unsafe_allow_html=True)

    with col2:
        if show_like_button:
            is_liked = users.is_paper_liked(st.session_state.username, paper_url)
            like_icon = "❤️" if is_liked else "🤍"

            if st.button(like_icon, key=f"like_{idx}_{paper_url}", help="Like this paper"):
                if is_liked:
                    users.unlike_paper(st.session_state.username, paper_url)
                else:
                    users.like_paper(st.session_state.username, item)
                st.rerun()

    with st.expander("📄 Abstract"):
        st.write(abstract)

    with st.expander("🔍 View PDF"):
        if pdf_viewer is not None and url_pdf and st.session_state.get("enable_pdf", True):
            pdf_file = get_pdf(url_pdf)
            if pdf_file is None:
                st.error("Invalid PDF File; try the link instead.")
            else:
                pdf_viewer(pdf_file, width="100%", height=600)
        else:
            if not pdf_viewer:
                st.warning("Install streamlit-pdf-viewer to view PDFs inline: pip install streamlit-pdf-viewer")
            if url_pdf:
                st.markdown(f"[Open PDF in new tab]({url_pdf})")

    st.markdown('</div>', unsafe_allow_html=True)

def search_page():
    st.markdown('<h1 class="main-header">🔬 Discover Research Papers</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

        st.markdown("---")

        st.markdown("### Navigation")
        if st.button("🔍 Search Papers", use_container_width=True, type="primary" if st.session_state.page == "search" else "secondary"):
            st.session_state.page = "search"
            st.rerun()

        if st.button("❤️ Liked Papers", use_container_width=True, type="primary" if st.session_state.page == "liked" else "secondary"):
            st.session_state.page = "liked"
            st.rerun()

        st.markdown("---")
        st.markdown("### Search Settings")
        st.session_state.top_k = st.slider("Number of Papers", min_value=1, max_value=20, value=5, step=1)
        st.session_state.use_mmr = st.checkbox("Use MMR Re-ranking", value=True)
        st.session_state.use_personalization = st.checkbox("Use Personalization", value=True, help="Tailor results based on your liked papers")
        st.session_state.llm = st.checkbox("LLM Explanations", value=True)
        st.session_state.enable_pdf = st.checkbox("Enable PDF Viewer", value=True)

    query = st.text_input("🔎 Search for research papers", placeholder="e.g., graph neural networks for molecule property prediction", key="search_query")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)

    if search_button and query:
        try:
            index = get_faiss_index()
            with st.spinner("Searching for papers..."):
                results, df, indices, explanation = search_papers(
                    query,
                    top_k=st.session_state.top_k,
                    index=index,
                    use_mmr=st.session_state.use_mmr,
                    llm=st.session_state.llm,
                    user=st.session_state.username,
                    use_personalization=st.session_state.use_personalization
                )

            if not results:
                st.info("No results found. Try a different query.")
                return

            if explanation:
                st.markdown("### 📊 Analysis")
                st.markdown(explanation)
                st.markdown("---")

            st.markdown(f"### Results ({len(results)} papers)")

            for i, item in enumerate(results):
                render_paper_card(item, i + 1, show_like_button=True)

        except Exception as e:
            st.error(f"Search failed: {e}")
            st.exception(e)

def liked_papers_page():
    st.markdown('<h1 class="main-header">❤️ Your Liked Papers</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

        st.markdown("---")

        st.markdown("### Navigation")
        if st.button("🔍 Search Papers", use_container_width=True, type="primary" if st.session_state.page == "search" else "secondary"):
            st.session_state.page = "search"
            st.rerun()

        if st.button("❤️ Liked Papers", use_container_width=True, type="primary" if st.session_state.page == "liked" else "secondary"):
            st.session_state.page = "liked"
            st.rerun()

        st.markdown("---")
        st.session_state.enable_pdf = st.checkbox("Enable PDF Viewer", value=True, key="liked_pdf_toggle")

    liked_papers = users.get_liked_papers(st.session_state.username)

    if not liked_papers:
        st.info("You haven't liked any papers yet. Search for papers and click the ❤️ button to save them here!")
        return

    st.markdown(f"### You have {len(liked_papers)} liked papers")
    st.markdown("---")

    for i, item in enumerate(liked_papers):
        render_paper_card(item, i + 1, show_like_button=True)

def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.page == "search":
            search_page()
        elif st.session_state.page == "liked":
            liked_papers_page()

if __name__ == "__main__":
    main()

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from app import settings

USERS_DIR = os.path.join(settings.root, ".users")
os.makedirs(USERS_DIR, exist_ok=True)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _get_user_file(username: str) -> str:
    return os.path.join(USERS_DIR, f"{username}.json")

def create_user(username: str, password: str) -> bool:
    if not username or not password:
        return False

    user_file = _get_user_file(username)
    if os.path.exists(user_file):
        return False

    user_data = {
        "username": username,
        "password": _hash_password(password),
        "created_at": datetime.now().isoformat(),
        "liked_papers": [],
        "search_history": [],
        "preferences": {}
    }

    with open(user_file, 'w') as f:
        json.dump(user_data, f, indent=2)

    return True

def authenticate_user(username: str, password: str) -> bool:
    user_file = _get_user_file(username)
    if not os.path.exists(user_file):
        return False

    with open(user_file, 'r') as f:
        user_data = json.load(f)

    return user_data["password"] == _hash_password(password)

def get_user_data(username: str) -> Optional[Dict]:
    user_file = _get_user_file(username)
    if not os.path.exists(user_file):
        return None

    with open(user_file, 'r') as f:
        return json.load(f)

def _save_user_data(username: str, user_data: Dict) -> None:
    user_file = _get_user_file(username)
    with open(user_file, 'w') as f:
        json.dump(user_data, f, indent=2)

def like_paper(username: str, paper: Dict) -> bool:
    user_data = get_user_data(username)
    if not user_data:
        return False

    paper_id = paper.get("paper_url") or paper.get("title")

    for liked in user_data["liked_papers"]:
        if liked.get("paper_url") == paper.get("paper_url"):
            return False

    date_value = paper.get("date")
    if date_value is not None:
        date_str = str(date_value)
    else:
        date_str = None

    liked_paper = {
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "url_pdf": paper.get("url_pdf"),
        "paper_url": paper.get("paper_url"),
        "date": date_str,
        "liked_at": datetime.now().isoformat()
    }

    user_data["liked_papers"].append(liked_paper)
    _save_user_data(username, user_data)
    return True

def unlike_paper(username: str, paper_url: str) -> bool:
    user_data = get_user_data(username)
    if not user_data:
        return False

    user_data["liked_papers"] = [
        p for p in user_data["liked_papers"]
        if p.get("paper_url") != paper_url
    ]

    _save_user_data(username, user_data)
    return True

def is_paper_liked(username: str, paper_url: str) -> bool:
    user_data = get_user_data(username)
    if not user_data:
        return False

    return any(p.get("paper_url") == paper_url for p in user_data["liked_papers"])

def get_liked_papers(username: str) -> List[Dict]:
    user_data = get_user_data(username)
    if not user_data:
        return []

    return user_data.get("liked_papers", [])

def add_search_history(username: str, query: str, results: List[Dict]) -> None:
    user_data = get_user_data(username)
    if not user_data:
        return

    search_entry = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results_count": len(results),
        "top_result": results[0].get("title") if results else None
    }

    user_data["search_history"].append(search_entry)

    if len(user_data["search_history"]) > 100:
        user_data["search_history"] = user_data["search_history"][-100:]

    _save_user_data(username, user_data)

def get_search_history(username: str) -> List[Dict]:
    user_data = get_user_data(username)
    if not user_data:
        return []

    return user_data.get("search_history", [])

def compute_user_preference_vector(username: str, embeddings_array: np.ndarray, df, indices_map: Dict) -> Optional[np.ndarray]:
    user_data = get_user_data(username)
    if not user_data or not user_data.get("liked_papers"):
        return None

    liked_indices = []
    for liked_paper in user_data["liked_papers"]:
        paper_url = liked_paper.get("paper_url")
        if not paper_url:
            continue

        matching_rows = df[df["paper_url"] == paper_url]
        if not matching_rows.empty:
            idx = matching_rows.index[0]
            liked_indices.append(idx)

    if not liked_indices:
        return None

    liked_embeddings = embeddings_array[liked_indices]
    user_vector = np.mean(liked_embeddings, axis=0)

    return user_vector

def personalize_scores(username: str, query_embedding: np.ndarray, candidate_embeddings: np.ndarray,
                       distances: np.ndarray, df, full_embeddings: np.ndarray, blend_weight: float = 0.3) -> np.ndarray:
    user_vector = compute_user_preference_vector(username, full_embeddings, df, {})

    if user_vector is None:
        return distances

    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-12)

    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_normalized = candidate_embeddings / (candidate_norms + 1e-12)

    user_similarities = candidate_normalized @ user_vector
    user_distances = 1.0 - user_similarities

    personalized_distances = (1 - blend_weight) * distances + blend_weight * user_distances

    return personalized_distances

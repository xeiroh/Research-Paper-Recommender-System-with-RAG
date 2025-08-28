import numpy as np

def l2_normalize(x, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def maximal_marginal_relevance(query_vec, doc_vecs, lambda_param=0.7, top_k=5, fetch_k=20):
    """
    Vectorized MMR that is typically much faster for medium/large candidate sets.

    Args:
        query_vec: np.ndarray of shape (d,)
        doc_vecs: np.ndarray of shape (n, d)
        lambda_param: float between 0 and 1
        top_k: number of results to return
        fetch_k: number of initial candidates to consider
    Returns:
        indices of selected documents in doc_vecs
    """
    if doc_vecs.ndim != 2:
        raise ValueError("doc_vecs must be 2D (n, d)")
    n, d = doc_vecs.shape
    if n == 0:
        return []

    # Normalize once (in-place-safe copy as float32 for speed/memory)
    q = l2_normalize(query_vec.reshape(1, -1).astype(np.float32, copy=False), axis=1)
    X = l2_normalize(doc_vecs.astype(np.float32, copy=False), axis=1)

    # Cosine similarity to query (n,)
    sim_q = (X @ q.T).ravel()

    fk = min(fetch_k, n)
    if fk <= 0:
        return []
    if fk < n:
        cand_idx = np.argpartition(-sim_q, fk - 1)[:fk]
    else:
        cand_idx = np.arange(n)

    # Work in the candidate subspace
    C = X[cand_idx]                   
    sim_q_c = sim_q[cand_idx].copy()  


    s_max = np.full(fk, -np.inf, dtype=np.float32)

    # Keep a mask of available candidates
    alive = np.ones(fk, dtype=bool)

    out = []
    k = min(top_k, fk)

    # Greedy selection
    last_sel_local = None
    for t in range(k):
        if t == 0:
            # First pick: purely by query similarity
            # Masked argmax
            masked = np.where(alive, sim_q_c, -np.inf)
            i = int(np.argmax(masked))
        else:
            # Update s_max using ONLY the newly added vector (incremental, O(fk*d))
            # Then take elementwise max with previous s_max
            s_max = np.maximum(s_max, C @ C[last_sel_local])

            # Compute MMR scores for alive items, ignore removed by setting -inf
            mmr = lambda_param * sim_q_c - (1.0 - lambda_param) * s_max
            masked = np.where(alive, mmr, -np.inf)
            i = int(np.argmax(masked))

        # Map back to global index and record
        out.append(int(cand_idx[i]))

        # Remove from future consideration
        alive[i] = False
        sim_q_c[i] = -np.inf
        s_max[i] = np.inf
        last_sel_local = i

    return out
# # app/services/rag/vector_store.py
# import os
# import pickle
# from typing import List, Dict, Any, Optional
# import numpy as np
# import faiss

# VECTOR_DIR = "vector_store"
# os.makedirs(VECTOR_DIR, exist_ok=True)

# INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
# META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

# # Globals
# _index: Optional[faiss.Index] = None
# _metadata: List[Dict[str, Any]] = []
# _dim: Optional[int] = None


# def _load():
#     """Load faiss index + metadata if present."""
#     global _index, _metadata, _dim
#     if os.path.exists(INDEX_PATH):
#         _index = faiss.read_index(INDEX_PATH)
#         if os.path.exists(META_PATH):
#             with open(META_PATH, "rb") as f:
#                 _metadata = pickle.load(f)
#         else:
#             _metadata = []
#         _dim = int(_index.d)
#     else:
#         _index = None
#         _metadata = []
#         _dim = None


# def _save():
#     """Persist index and metadata to disk."""
#     global _index, _metadata
#     if _index is not None:
#         faiss.write_index(_index, INDEX_PATH)
#     with open(META_PATH, "wb") as f:
#         pickle.dump(_metadata, f)


# def _ensure_index(d: int):
#     """Create index if missing with dimension d."""
#     global _index, _dim
#     if _index is None:
#         # simple Flat index (L2). For large datasets consider IVF or HNSW.
#         _index = faiss.IndexFlatL2(d)
#         _dim = d
#     else:
#         if _dim is None:
#             _dim = int(_index.d)
#         if _dim != d:
#             raise ValueError(f"Inconsistent embedding dimension: index expects {_dim}, got {d}")


# def add_embeddings(vectors, metadatas: List[Dict[str, Any]]):
#     """
#     Add embeddings to FAISS and persist metadata.
#     `vectors` may be:
#       - list of lists
#       - numpy array shape (n,d)
#       - single 1-D numpy array (d,) or list (will be treated as single vector)
#     `metadatas` must be a list of dicts of length n.
#     """
#     global _index, _metadata, _dim

#     # load existing index if not yet loaded
#     if _index is None and os.path.exists(INDEX_PATH):
#         _load()

#     # coerce to numpy float32 array
#     vecs = np.asarray(vectors, dtype=np.float32)

#     # handle 1-D -> treat as single vector
#     if vecs.ndim == 1:
#         vecs = vecs.reshape(1, -1)

#     if vecs.ndim != 2:
#         raise ValueError("Embeddings must be 2-D (n_vectors, dim) or 1-D (dim,)")

#     n, d = vecs.shape

#     # validate metadatas length
#     if not isinstance(metadatas, list) or len(metadatas) != n:
#         raise ValueError(f"metadata length ({len(metadatas) if isinstance(metadatas, list) else 'NA'}) "
#                          f"must match number of vectors ({n})")

#     # ensure index exists and dimension matches
#     _ensure_index(d)

#     # finally add
#     try:
#         _index.add(vecs)   # faiss expects numpy.float32 array
#     except Exception as e:
#         raise RuntimeError(f"Faiss add failed: {e}")

#     # extend metadata
#     _metadata.extend(metadatas)

#     # persist
#     _save()

#     # return total vectors in index (useful)
#     return int(_index.ntotal)


# def similarity_search(query_vector, k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Return top-k metadata dicts for a query_vector.
#     `query_vector` can be 1-D or 2-D numpy/list.
#     """
#     global _index, _metadata

#     if _index is None:
#         # try to load saved index
#         _load()
#         if _index is None:
#             return []

#     q = np.asarray(query_vector, dtype=np.float32)
#     if q.ndim == 1:
#         q = q.reshape(1, -1)
#     if q.ndim != 2:
#         raise ValueError("query_vector must be 1-D or 2-D array")

#     # dimension check
#     if q.shape[1] != int(_index.d):
#         raise ValueError(f"Query vector dim {q.shape[1]} does not match index dim {_index.d}")

#     # perform search
#     D, I = _index.search(q, k)  # D: distances, I: indices
#     results = []
#     for idx in I[0]:
#         if idx < 0:
#             continue
#         if idx < len(_metadata):
#             results.append(_metadata[idx])
#     return results

###### more optimized

# app/services/rag/vector_store.py
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss

VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

_index: Optional[faiss.Index] = None
_metadata: List[Dict] = []
_dim: Optional[int] = None

def _load():
    global _index, _metadata, _dim
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                _metadata[:] = pickle.load(f)
        _dim = _index.d

def _save():
    global _index, _metadata
    if _index is not None:
        faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

def add_embeddings(vectors: List[List[float]], metadatas: List[Dict]):
    """
    Accepts `vectors` as list-of-lists or numpy array.
    Ensures dtype=float32 and 2D array for faiss.
    """
    global _index, _metadata, _dim

    if vectors is None or len(vectors) == 0:
        return

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n, d = arr.shape

    if _index is None:
        _dim = d
        _index = faiss.IndexFlatL2(_dim)

    # If index dim mismatches, re-create or raise
    if _index.d != d:
        raise ValueError(f"Index dimension {_index.d} != embeddings dimension {d}")

    _index.add(arr)                # faiss expects numpy array, not list
    _metadata.extend(metadatas or [])
    _save()

def similarity_search(query_vector, k=5):
    global _index, _metadata
    if _index is None:
        _load()
    if _index is None:
        return []

    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    D, I = _index.search(q, k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        if idx < len(_metadata):
            results.append(_metadata[idx])
    return results

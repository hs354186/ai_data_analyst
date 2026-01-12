# from openai import OpenAI
# from typing import List
# import os

# client = OpenAI()

# EMBEDDING_MODEL = os.environ.get(
#     "OPENAI_EMBEDDING_MODEL",
#     "text-embedding-3-small"  # ✅ fast + cheap
# )

# BATCH_SIZE = 64  # ⬅️ speed + reliability

# def embed_texts(texts: List[str]) -> List[List[float]]:
#     if not texts:
#         return []

#     embeddings = []

#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i:i + BATCH_SIZE]

#         response = client.embeddings.create(
#             model=EMBEDDING_MODEL,
#             input=batch
#         )

#         embeddings.extend([d.embedding for d in response.data])

#     return embeddings

########## above was also working .. but below is used for avoiding api rate limit error

# app/services/rag/embeddings.py
import os
import time
from typing import List, Optional
import numpy as np
from openai import OpenAI

# choose an embedding model (change via env if desired)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI()  # assumes OPENAI_API_KEY in env

def embed_texts(texts: List[str], batch_size: int = 64, max_retries: int = 5, backoff_base: float = 1.0) -> np.ndarray:
    """
    Create embeddings for a list of texts in batches.
    Returns a numpy array of shape (n_texts, dim) dtype=float32.
    - batch_size: how many inputs to send per request
    - max_retries: retry attempts for transient errors
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    clean_texts = [t if (t is not None) else "" for t in texts]

    all_embs = []
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch
                )
                # SDK response: resp.data is a list of objects with .embedding
                for item in resp.data:
                    all_embs.append(item.embedding)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Embedding failed after {max_retries} retries: {e}")
                # exponential backoff with jitter
                sleep = min(backoff_base * (2 ** (attempt - 1)), 30.0)
                time.sleep(sleep + (0.1 * attempt))
    arr = np.asarray(all_embs, dtype=np.float32)
    # If everything okay, arr should be (n, d)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

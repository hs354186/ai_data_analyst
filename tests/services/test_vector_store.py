import os
from app.services.rag.vector_store import add_embeddings, query, load_index_and_meta
import numpy as np

def test_vector_store_roundtrip(tmp_path, monkeypatch):
    # create two 3-dim vectors and meta and add to a fresh index
    vecs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    metas = [{"source":"a","chunk_id":"c1"},{"source":"b","chunk_id":"c2"}]
    # ensure fresh index files removed
    from app.services.rag.vector_store import INDEX_PATH, META_PATH
    try:
        os.remove(INDEX_PATH)
    except Exception:
        pass
    try:
        os.remove(META_PATH)
    except Exception:
        pass

    total = add_embeddings(vecs, metas)
    assert total >= 2

    # query using vector similar to first
    res = query([1.0, 0.0, 0.0], k=1)
    assert len(res) >= 1
    assert res[0]["source"] in ["a","b"]



# from fastapi import APIRouter, HTTPException
# from pathlib import Path
# import pandas as pd

# from app.services.rag.chunker import chunk_dataframe
# from app.services.rag.embeddings import embed_texts
# from app.services.rag.vector_store import add_embeddings, similarity_search

# router = APIRouter()
# PARQUET_DIR = Path("data/parquet")

# # -------------------------
# # INGEST DATASET
# # -------------------------
# @router.post("/rag/ingest/{dataset_id}")
# def ingest_dataset(dataset_id: str):
#     parquet_path = PARQUET_DIR / f"{dataset_id}.parquet"
#     if not parquet_path.exists():
#         raise HTTPException(status_code=404, detail="Dataset not found")

#     df = pd.read_parquet(parquet_path)

#     chunks = chunk_dataframe(df, dataset_id)

#     valid_chunks = [
#         c for c in chunks
#         if isinstance(c.get("text"), str) and c["text"].strip()
#     ]

#     if not valid_chunks:
#         raise HTTPException(
#             status_code=400,
#             detail="No valid chunks generated"
#         )

#     texts = [c["text"] for c in valid_chunks]
#     metadatas = [c["metadata"] for c in valid_chunks]

#     embeddings = embed_texts(texts)
#     add_embeddings(embeddings, metadatas)

#     return {
#         "dataset_id": dataset_id,
#         "rows": len(df),
#         "chunks_ingested": len(texts)
#     }

# # -------------------------
# # QUERY (RAG SEARCH)
# # -------------------------
# @router.post("/rag/query")
# def rag_query(query_text: str, k: int = 5):
#     if not query_text.strip():
#         raise HTTPException(status_code=400, detail="Query text is empty")

#     query_embedding = embed_texts([query_text])[0]

#     results = similarity_search(
#         query_embedding=query_embedding,
#         k=k
#     )

#     return {
#         "query": query_text,
#         "results": results
#     }


# app/api/routes/rag.py
from fastapi import APIRouter, HTTPException, Body
from pathlib import Path
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple

from app.services.rag.chunker import chunk_dataframe
from app.services.rag.embeddings import embed_texts
from app.services.rag.vector_store import add_embeddings, similarity_search

load_dotenv()

router = APIRouter()
PARQUET_DIR = Path("data/parquet")

# OpenAI client (SDK v1 style you've used elsewhere)
client = OpenAI()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# -------------------------
# INGEST DATASET
# -------------------------
@router.post("/rag/ingest/{dataset_id}")
def ingest_dataset(dataset_id: str):
    parquet_path = PARQUET_DIR / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # read parquet
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {e}")

    # create chunks
    chunks = chunk_dataframe(df, dataset_id)

    # keep only textful chunks
    valid_chunks = [
        c for c in chunks
        if isinstance(c.get("text"), str) and c["text"].strip()
    ]

    if not valid_chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid chunks generated"
        )

    texts = [c["text"] for c in valid_chunks]
    metadatas = [c["metadata"] for c in valid_chunks]

    # IMPORTANT: store the chunk text in metadata so RAG retrieval is fast
    metadatas_with_text = []
    for m, t in zip(metadatas, texts):
        md = dict(m) if isinstance(m, dict) else {}
        md["text"] = t
        # also include a 'source' field for traceability if not present
        if "source" not in md:
            md["source"] = md.get("dataset_id", dataset_id)
        metadatas_with_text.append(md)

    # 1) create embeddings (embed_texts should return a list/ndarray of vectors)
    embeddings = embed_texts(texts)

    # make sure embeddings are numpy arrays (faiss expects ndarray when adding)
    embeddings_arr = np.asarray(embeddings, dtype=np.float32)

    # 2) add embeddings and metadata to vector store
    try:
        add_embeddings(embeddings_arr, metadatas_with_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add embeddings: {e}")

    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "chunks_ingested": len(texts)
    }


# -------------------------
# Helper: extract chunk text from metadata (fast) or reconstruct from parquet
# -------------------------
def _get_chunk_text_from_meta(meta: dict) -> str:
    """
    Prefer meta['text'] (fast). If not present, try to reconstruct from parquet
    using dataset_id + row_start/row_end. Returns empty string on failure.
    """
    if not isinstance(meta, dict):
        return ""

    # direct text included at ingest time: ideal
    if "text" in meta and isinstance(meta["text"], str):
        return meta["text"]

    # attempt reconstruction (slow)
    ds = meta.get("dataset_id")
    if not ds:
        return ""

    parquet_path = PARQUET_DIR / f"{ds}.parquet"
    if not parquet_path.exists():
        return ""

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return ""

    start = meta.get("row_start", None)
    end = meta.get("row_end", None)
    if start is None or end is None:
        sample_df = df.head(20)
    else:
        sample_df = df.iloc[start:end]

    lines = []
    for _, row in sample_df.iterrows():
        pairs = []
        for col in sample_df.columns:
            v = row[col]
            if pd.isna(v) or v == "":
                continue
            pairs.append(f"{col}: {v}")
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def _assemble_context(metas: List[dict], max_chars: int = 3000) -> Tuple[str, List[Tuple[dict, str]]]:
    """
    Build a context string by concatenating retrieved chunk texts (prefixed by source).
    Stops when max_chars limit reached. Returns (context_string, [(meta, text), ...]).
    """
    ctx_parts = []
    contexts = []
    chars = 0

    for m in metas:
        txt = _get_chunk_text_from_meta(m)
        if not txt:
            continue
        src = m.get("source") or m.get("dataset_id") or "unknown"
        part = f"Source: {src}\n{txt}\n---\n"
        part_len = len(part)
        # if adding this part would exceed limit
        if chars + part_len > max_chars:
            if chars == 0:
                # accept a truncated first part
                truncated = part[: max_chars - 10] + "\n...[truncated]\n"
                ctx_parts.append(truncated)
                contexts.append((m, txt[: max_chars - 10]))
            break
        ctx_parts.append(part)
        contexts.append((m, txt))
        chars += part_len

    return "\n".join(ctx_parts), contexts


# -------------------------
# QUERY (RAG Search + LLM)
# -------------------------
@router.post("/rag/query")
def rag_query(query_text: str = Body(..., embed=True), k: int = 5):
    """
    RAG query:
      - embed query
      - retrieve top-k chunks (metadata) from vector store
      - assemble compact context
      - call LLM with context + question
      - return llm_raw and source snippets
    """
    if not query_text or not query_text.strip():
        raise HTTPException(status_code=400, detail="Query text is empty")

    # 1) embed query
    try:
        q_emb = embed_texts([query_text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {e}")

    # make sure it's a numpy vector for similarity_search
    try:
        q_vec = np.asarray(q_emb, dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=500, detail="Query embedding format invalid")

    # 2) retrieve top-k metadata items
    try:
        # vector_store.similarity_search expects a numpy vector
        results_meta = similarity_search(q_vec, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    if not results_meta:
        return {"query": query_text, "results": [], "llm_raw": "", "sources": []}

    # 3) assemble context (bounded)
    context_str, contexts = _assemble_context(results_meta, max_chars=3000)

    # 4) form prompt
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful data assistant. Use the provided CONTEXT (dataset excerpts) to answer the question. "
            "If the context doesn't contain enough information, say so rather than guessing. Cite sources using 'Source: <source>'."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {query_text}\n\nAnswer concisely and reference sources when relevant."
    }

    # 5) call LLM
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=600,
        )
        llm_raw = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # 6) return answer + source snippets (trimmed)
    sources_out = []
    for m, txt in contexts:
        sources_out.append({
            "meta": m,
            "snippet": txt[:1000]
        })

    return {
        "query": query_text,
        "llm_raw": llm_raw,
        "sources": sources_out
    }

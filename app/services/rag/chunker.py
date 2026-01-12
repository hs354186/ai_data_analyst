# import pandas as pd
# from typing import List, Dict

# MAX_CHARS = 1500   # ⬅️ critical for embedding safety

# def chunk_dataframe(
#     df: pd.DataFrame,
#     dataset_id: str,
#     rows_per_chunk: int = 5
# ) -> List[Dict]:
#     chunks = []

#     total_rows = len(df)

#     for start in range(0, total_rows, rows_per_chunk):
#         end = min(start + rows_per_chunk, total_rows)
#         subset = df.iloc[start:end]

#         text_lines = []
#         for _, row in subset.iterrows():
#             pairs = [
#                 f"{col}: {str(row[col])}"
#                 for col in df.columns
#                 if pd.notna(row[col]) and str(row[col]).strip()
#             ]
#             if pairs:
#                 text_lines.append(" | ".join(pairs))

#         if not text_lines:
#             continue

#         chunk_text = "\n".join(text_lines)

#         # 🔒 HARD LIMIT TEXT SIZE
#         if len(chunk_text) > MAX_CHARS:
#             chunk_text = chunk_text[:MAX_CHARS]

#         chunks.append({
#             "text": chunk_text,
#             "metadata": {
#                 "dataset_id": dataset_id,
#                 "row_start": start,
#                 "row_end": end
#             }
#         })

#     return chunks

####### more optimized

# app/services/rag/chunker.py
import pandas as pd
from typing import List, Dict
import uuid

MAX_CHARS = 2000   # ✅ SAFE for embeddings (≈500 tokens)

def chunk_dataframe(
    df: pd.DataFrame,
    dataset_id: str,
    rows_per_chunk: int = 25
) -> List[Dict]:

    chunks = []
    total_rows = len(df)
    if total_rows == 0:
        return chunks

    cols = list(df.columns)

    for start in range(0, total_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, total_rows)
        subset = df.iloc[start:end]

        lines = []
        for _, row in subset.iterrows():
            pairs = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    continue
                pairs.append(f"{col}: {str(val)}")
            if pairs:
                lines.append(" | ".join(pairs))

        if not lines:
            continue

        text = "\n".join(lines)

        # 🔒 HARD LIMIT — THIS FIXES YOUR ERROR
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        chunks.append({
            "text": text,
            "metadata": {
                "dataset_id": dataset_id,
                "row_start": start,
                "row_end": end,
                "chunk_id": str(uuid.uuid4())
            }
        })

    return chunks

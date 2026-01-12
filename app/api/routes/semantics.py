# app/api/routes/semantics.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import json
import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
router = APIRouter()

PARQUET_DIR = Path("data/parquet")
METADATA_DIR = Path("data/metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client (SDK v1)
client = OpenAI()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
import os
print("OPENAI_API_KEY exists:", "OPENAI_API_KEY" in os.environ)

def sample_column_values(df: pd.DataFrame, col: str, max_samples: int = 8) -> List[str]:
    if col not in df.columns:
        return []

    s = df[col].dropna().astype(str)
    if s.empty:
        return []

    if len(s) > max_samples:
        s = s.sample(n=max_samples, random_state=42)

    vals = list(dict.fromkeys(s.tolist()))
    return vals[:max_samples]


def build_prompt(dataset_id: str, df: pd.DataFrame) -> str:
    lines = []
    lines.append(
        f"You are a data understanding assistant. "
        f"For dataset id `{dataset_id}`, return a JSON array describing each column."
    )
    lines.append("Each element must include:")
    lines.append("- original_name")
    lines.append("- semantic_meaning (10–40 words)")
    lines.append("- semantic_role: one of [id, dimension, measure, date, text, currency, categorical, boolean, other]")
    lines.append("- confidence (0.0–1.0)")
    lines.append("- note (optional)")
    lines.append("")
    lines.append("ONLY return valid JSON. No explanations.")
    lines.append("")
    lines.append("Columns and example values:")

    for col in df.columns:
        vals = sample_column_values(df, col)
        short_vals = [v[:80] + "..." if len(v) > 80 else v for v in vals]
        lines.append(f"Column: {col}")
        lines.append(f"Examples: {short_vals}")
        lines.append("")

    return "\n".join(lines)


def call_llm_for_semantics(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                # "content": "You are a data semantics inference engine. Respond ONLY with valid JSON."
                "content": (
                    "You are a data semantics inference engine. "
                    "Return ONLY valid JSON. "
                    "Do NOT use markdown, code fences, or explanations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=900,
    )

    return response.choices[0].message.content

# def clean_llm_json(text: str) -> Any:
#     """
#     Handles:
#     1. Markdown fenced JSON
#     2. JSON returned as escaped string
#     3. Normal JSON
#     """
#     text = text.strip()

#     # Remove markdown fences
#     if text.startswith("```"):
#         text = text.split("```")[1].strip()

#     # If this is a JSON-encoded string, decode it once
#     try:
#         decoded = json.loads(text)
#         if isinstance(decoded, str):
#             # JSON inside a string → decode again
#             return json.loads(decoded)
#         return decoded
#     except json.JSONDecodeError:
#         # Last resort: try raw
#         return json.loads(text)

import re

def parse_llm_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extracts multiple JSON objects from raw LLM output
    even if they are not wrapped in an array.
    """
    text = text.strip()

    # Remove markdown if any
    if text.startswith("```"):
        text = text.split("```")[1]

    # Unescape once if needed
    try:
        text = json.loads(text)
    except Exception:
        pass

    objects = []
    for match in re.finditer(r"\{[\s\S]*?\}", text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            continue

    return objects


@router.post("/datasets/{dataset_id}/semantics")
def infer_dataset_semantics(dataset_id: str):
    parquet_path = PARQUET_DIR / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {e}")

    prompt = build_prompt(dataset_id, df)

    try:
        llm_raw = call_llm_for_semantics(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    parsed = []
    parse_error = None

    try:
        # candidate = json.loads(llm_raw)
        # cleaned = clean_llm_json(llm_raw)
        # candidate = json.loads(cleaned)

        candidate = parse_llm_objects(llm_raw)

        if isinstance(candidate, dict):
            candidate = [candidate]

        for item in candidate:
            if not isinstance(item, dict):
                continue

            parsed.append({
                "original_name": item.get("original_name"),
                "semantic_meaning": item.get("semantic_meaning", ""),
                "semantic_role": item.get("semantic_role", "other"),
                "confidence": float(item.get("confidence", 0.0)),
                "note": item.get("note", "")
            })

    except Exception as e:
        parse_error = str(e)
        parsed = [
            {
                "original_name": col,
                "semantic_meaning": "",
                "semantic_role": "other",
                "confidence": 0.0,
                "note": "LLM JSON parse failed"
            }
            for col in df.columns
        ]

    metadata = {
        "dataset_id": dataset_id,
        "domain": "unknown",
        "columns": parsed,
        "llm_raw": llm_raw,
        "parse_error": parse_error,
    }

    meta_path = METADATA_DIR / f"{dataset_id}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata

# import json
# from pathlib import Path
# from app.core.llm import llm_chat
# from app.services.semantics.prompts import SEMANTIC_INFERENCE_PROMPT

# METADATA_DIR = Path("data/metadata")

# def infer_semantics(metadata: dict) -> dict:
#     # prompt = SEMANTIC_INFERENCE_PROMPT.format(metadata=json.dumps(metadata))

#     prompt = SEMANTIC_INFERENCE_PROMPT.format(
#     metadata=json.dumps(metadata.get("columns", []), indent=2)
#     )


#     response = llm_chat(
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )

#     try:
#         semantics = json.loads(response)
#     except json.JSONDecodeError as e:
#         print("LLM RAW RESPONSE:\n", response)
#         raise ValueError("LLM returned invalid JSON") from e

#     return semantics


# def infer_and_save(dataset_id: str) -> dict:
#     meta_path = METADATA_DIR / f"{dataset_id}.json"
#     if not meta_path.exists():
#         raise FileNotFoundError("Metadata not found")

#     metadata = json.loads(meta_path.read_text())
#     semantics = infer_semantics(metadata)

#     out_path = METADATA_DIR / f"{dataset_id}_semantics.json"
#     out_path.write_text(json.dumps(semantics, indent=2))

#     return semantics


################# New

import json
import re
from pathlib import Path
from app.core.llm import llm_chat
from app.services.semantics.prompts import SEMANTIC_INFERENCE_PROMPT

METADATA_DIR = Path("data/metadata")
SEMANTICS_SUFFIX = "_semantics.json"

# helper: try safe JSON parse, then try to extract a {...} substring and repair common issues
def _safe_parse_json(text: str):
    # 1) try strict parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) try to find the largest {...} block (first { to last })
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        # quick fixes: replace single quotes with double quotes (only if safe),
        # remove trailing commas before } or ],
        # Note: these are heuristics — not perfect but often works for model output.
        candidate_fixed = candidate

        # remove newlines that break JSON (keep them though usually ok)
        candidate_fixed = candidate_fixed.strip()

        # remove trailing commas like {...,}
        candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate_fixed)

        # if there are single quotes instead of double quotes, convert them carefully:
        # only convert when it's likely JSON: ensure there are no double quotes present meaningfully
        # (this avoids breaking valid JSON)
        if "'" in candidate_fixed and '"' not in candidate_fixed:
            candidate_fixed = candidate_fixed.replace("'", '"')

        try:
            return json.loads(candidate_fixed)
        except Exception:
            # fall through to last resort
            pass

    # nothing worked
    raise ValueError("Could not parse JSON from LLM response")


def infer_semantics(metadata: dict) -> dict:
    """
    Calls the LLM with a prompt and returns a semantics dict.
    Uses a robust parsing fallback for noisy LLM responses.
    """
    # keep prompt focused: only include columns array to reduce size
    columns = metadata.get("columns", [])
    prompt = SEMANTIC_INFERENCE_PROMPT.format(metadata=json.dumps(columns, indent=2))

    # Make the instruction explicit to return JSON only
    system_msg = {
        "role": "system",
        "content": "You are a careful assistant. **Return ONLY valid JSON** with keys: domain, columns (array of objects with original_name, meaning, role, confidence). No extra commentary."
    }

    user_msg = {"role": "user", "content": prompt}

    # Try once, then retry once if parsing fails
    attempts = 2
    last_response_text = None
    for attempt in range(attempts):
        response_text = llm_chat([system_msg, user_msg], temperature=0)
        last_response_text = response_text

        # debug print of raw response — useful during development (remove or log later)
        print("LLM RAW RESPONSE (attempt", attempt + 1, "):\n", response_text)

        try:
            semantics = _safe_parse_json(response_text)
            # optionally: validate minimal shape
            if "columns" not in semantics or "domain" not in semantics:
                raise ValueError("Parsed JSON missing required keys")
            return semantics
        except Exception as e:
            # if last attempt, raise a clear error, otherwise retry
            print("JSON parse attempt failed:", str(e))
            if attempt == attempts - 1:
                # final fallback = conservative default semantics
                fallback = {
                    "domain": "unknown",
                    "columns": [
                        {
                            "original_name": c.get("name", ""),
                            "meaning": "unknown",
                            "role": "attribute",
                            "confidence": 0.0,
                        }
                        for c in columns
                    ],
                    "confidence": 0.0,
                    "note": "fallback due to parsing errors",
                    "llm_raw": response_text[:2000]  # store a snippet for debugging
                }
                print("Returning fallback semantics due to repeated parse failures.")
                return fallback
            # else loop to retry once

    # unreachable
    raise RuntimeError("Semantic inference failed unexpectedly")


def infer_and_save(dataset_id: str) -> dict:
    """
    Load metadata for dataset_id, infer semantics, save to metadata file with _semantics suffix.
    Returns the saved semantics dict.
    """
    meta_path = METADATA_DIR / f"{dataset_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError("Metadata file not found for dataset_id")

    metadata = json.loads(meta_path.read_text())
    semantics = infer_semantics(metadata)

    sem_path = METADATA_DIR / f"{dataset_id}{SEMANTICS_SUFFIX}"
    sem_path.write_text(json.dumps(semantics, indent=2))
    return semantics

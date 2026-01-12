# app/api/routes/upload.py

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion.reader import (
    read_tabular_file,
    sanitize_dataframe,
)

router = APIRouter()

PARQUET_DIR = Path("data/parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())

        # 1. Read file
        df = read_tabular_file(file)

        # 2. Sanitize for parquet
        df = sanitize_dataframe(df)

        # 3. Write parquet
        parquet_path = PARQUET_DIR / f"{dataset_id}.parquet"
        df.to_parquet(
            parquet_path,
            engine="fastparquet",
            index=False
        )

        return {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

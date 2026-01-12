# app/services/ingestion/reader.py

import pandas as pd
from fastapi import UploadFile


def read_tabular_file(file: UploadFile) -> pd.DataFrame:
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file, low_memory=False)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file.file)
    else:
        raise ValueError("Unsupported file format")

    return df


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dataframe SAFE for parquet:
    - No mixed object/numeric columns
    - No NaN / None
    - Convert everything unstable → string
    """

    for col in df.columns:
        # If column is object OR mixed → force string
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .replace(["nan", "None", "NaN"], "")
            )
        else:
            # numeric columns: replace NaN safely
            df[col] = df[col].fillna(0)

    return df

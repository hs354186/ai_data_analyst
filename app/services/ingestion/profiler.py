import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> dict:
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "sample_values": df[col]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            }
            for col in df.columns
        ],
    }

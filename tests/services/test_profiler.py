import pandas as pd
from app.services.ingestion.profiler import profile_dataframe

def test_profile_dataframe():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

    profile = profile_dataframe(df)

    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["columns"][0]["name"] == "A"

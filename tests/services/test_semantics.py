from app.services.semantics.infer import infer_semantics

def test_semantics_structure():
    metadata = {
        "columns": [
            {"name": "Vendor", "dtype": "object"},
            {"name": "Net Value", "dtype": "float"},
            {"name": "Posting Date", "dtype": "datetime"},
        ]
    }

    semantics = infer_semantics(metadata)

    assert "domain" in semantics
    assert "columns" in semantics
    assert isinstance(semantics["columns"], list)

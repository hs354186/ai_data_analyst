import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_then_semantics():
    csv = "Vendor,Net Value,Date\nA,100,2024-01-01\n"
    file = io.BytesIO(csv.encode())

    res = client.post("/upload", files={"file": ("test.csv", file)})
    dataset_id = res.json()["dataset_id"]

    res2 = client.post(f"/datasets/{dataset_id}/semantics")
    assert res2.status_code == 200
    assert "semantics" in res2.json()

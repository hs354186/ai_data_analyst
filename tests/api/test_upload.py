from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_upload_csv():
    csv_content = "col1,col2\n1,10\n2,20\n"
    file = io.BytesIO(csv_content.encode("utf-8"))

    response = client.post(
        "/upload",
        files={"file": ("test.csv", file, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["rows"] == 2
    assert len(data["columns"]) == 2

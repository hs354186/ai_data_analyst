from app.services.rag.chunker import chunk_text

def test_chunker_small():
    t = "a" * 3000  # long string
    chunks = chunk_text(t, chunk_size=1200, overlap=200)
    assert len(chunks) >= 2
    assert all("text" in c for c in chunks)

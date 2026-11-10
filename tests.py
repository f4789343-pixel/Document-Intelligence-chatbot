import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_upload():
  files = [
    ("files", ("test.txt", b"Machine learning is a branch of artificial intelligence.", "text/plain"))]
  response = client.post("/upload", files=files)
  assert response.status_code == 200

def test_chat():
  response = client.post("/chat", json={"question": "what is machine learning?"})
  assert response.status_code == 200
  assert "response" in response.json()



  
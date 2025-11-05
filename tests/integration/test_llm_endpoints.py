"""Integration tests for LLM API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app
from src.llm.schemas import LLMResponse

client = TestClient(app)

@pytest.fixture
def mock_llm_services():
    """Mock LLM services for testing endpoints."""
    with patch('src.api.v1.endpoints.llm.llm_service') as mock_llm, \
         patch('src.api.v1.endpoints.llm.embedding_service') as mock_embedding:
        
        # Mock LLM service responses
        mock_llm.generate_text.return_value = ("Test response", {"prompt_tokens": 10, "completion_tokens": 20})
        mock_llm.stream_text.return_value = ["Chunk1", "Chunk2"]
        
        # Mock Embedding service responses
        mock_embedding.get_embeddings.return_value = (
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            {"prompt_tokens": 15, "total_tokens": 15}
        )
        
        yield mock_llm, mock_embedding

def test_generate_text_endpoint(mock_llm_services):
    """Test the text generation endpoint."""
    mock_llm, _ = mock_llm_services
    
    response = client.post(
        "/api/v1/llm/generate",
        json={
            "prompt": "Test prompt",
            "temperature": 0.7,
            "max_tokens": 100
        },
        headers={"X-API-Key": "test-key"}
    )
    
    assert response.status_code == 200
    assert "text" in response.json()
    assert response.json()["model"] == "llama-2-7b"
    mock_llm.generate_text.assert_called_once()

def test_stream_text_endpoint(mock_llm_services):
    """Test the streaming text generation endpoint."""
    mock_llm, _ = mock_llm_services
    
    response = client.post(
        "/api/v1/llm/stream",
        json={"prompt": "Test prompt"},
        headers={"X-API-Key": "test-key"}
    )
    
    assert response.status_code == 200
    assert b"Chunk1" in response.content
    assert b"Chunk2" in response.content
    mock_llm.stream_text.assert_called_once()

def test_embeddings_endpoint(mock_llm_services):
    """Test the embeddings endpoint."""
    _, mock_embedding = mock_llm_services
    
    response = client.post(
        "/api/v1/llm/embeddings",
        json={"texts": ["text1", "text2"]},
        headers={"X-API-Key": "test-key"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 3
    assert data["model"] == "m2bert"
    mock_embedding.get_embeddings.assert_called_once()

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

"""Tests for Apriel Generation Service using mocks.

This test module contains unit tests for the AprielGenerationService.
It uses mocks to avoid hitting external API rate limits.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import List, Dict, Any

from src.llm.generation import AprielGenerationService
from src.llm.schemas import LLMResponse, LLMModelType, LLMProvider

# Test configuration
TEST_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TEST_API_KEY = "test-api-key-123"

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "output": {"choices": [{"text": "Mocked response"}]},
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }

@pytest.fixture
def generation_service():
    """Create and configure a generation service instance for testing with mocks."""
    with patch('src.llm.generation.AsyncClient') as mock_client:
        mock_client.return_value = AsyncMock()
        service = AprielGenerationService(
            api_key=TEST_API_KEY,
            model=TEST_MODEL,
            system_prompt="You are a helpful AI assistant.",
            timeout=30
        )
        return service

@pytest.mark.asyncio
async def test_generate_single_turn(generation_service, mock_llm_response):
    """Test single-turn text generation with mocked response."""
    # Setup mock
    mock_response = AsyncMock()
    mock_response.json.return_value = mock_llm_response
    generation_service.client.post.return_value = mock_response
    
    # Call the method
    response = await generation_service.generate(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=100
    )
    
    # Verify response
    assert response == "Mocked response"
    
    # Verify the API was called with correct parameters
    generation_service.client.post.assert_called_once()
    _, kwargs = generation_service.client.post.call_args
    assert kwargs["json"]["prompt"] == "Test prompt"
    assert kwargs["json"]["temperature"] == 0.7
    assert kwargs["json"]["max_tokens"] == 100

@pytest.mark.asyncio
async def test_chat_with_history(generation_service, mock_llm_response):
    """Test chat completion with conversation history using mocks."""
    # Setup mock
    mock_response = AsyncMock()
    mock_response.json.return_value = mock_llm_response
    generation_service.client.post.return_value = mock_response
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Ethereum?"}
    ]
    
    # Call the method
    response = await generation_service.chat(
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )
    
    # Verify response
    assert response == "Mocked response"
    
    # Verify the API was called with correct parameters
    generation_service.client.post.assert_called_once()
    _, kwargs = generation_service.client.post.call_args
    assert kwargs["json"]["messages"] == messages
    assert kwargs["json"]["temperature"] == 0.7
    assert kwargs["json"]["max_tokens"] == 100

@pytest.mark.asyncio
async def test_generate_with_retries(generation_service):
    """Test that the service retries on transient errors."""
    # Setup mock to fail once then succeed
    mock_response = AsyncMock()
    mock_response.json.side_effect = [
        Exception("Temporary error"),
        {"output": {"choices": [{"text": "Retry success"}]}, "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}
    ]
    generation_service.client.post.return_value = mock_response
    
    # Call the method with retries
    response = await generation_service.generate(
        prompt="Test retry",
        max_retries=2
    )
    
    # Verify response after retry
    assert response == "Retry success"
    assert generation_service.client.post.call_count == 2

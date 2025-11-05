"""Tests for Apriel Generation Service.

This test module contains unit tests for the AprielGenerationService.
It uses mocks to test the service without hitting external APIs.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock, ANY
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
def mock_together():
    """Create a mock AsyncTogether client."""
    with patch('together.AsyncTogether') as mock_together:
        mock_client = AsyncMock()
        mock_together.return_value = mock_client
        yield mock_together

@pytest.mark.asyncio
async def test_generate_single_turn(mock_together, mock_llm_response):
    """Test single-turn text generation with mocked response."""
    # Setup mock
    mock_client = mock_together.return_value
    mock_client.chat.completions.create.return_value = AsyncMock()
    mock_client.chat.completions.create.return_value.choices = [
        AsyncMock(message=AsyncMock(content="Mocked response"))
    ]
    
    # Create service with mocked client
    service = AprielGenerationService(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        system_prompt="You are a helpful AI assistant.",
        timeout=30
    )
    
    # Call the method
    response = await service.generate(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=100
    )
    
    # Verify response
    assert isinstance(response, str)
    assert response == "Mocked response"
    
    # Verify the API was called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"][-1]["content"] == "Test prompt"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100

@pytest.mark.asyncio
async def test_chat_with_history(mock_together, mock_llm_response):
    """Test chat completion with conversation history using mocks."""
    # Setup mock
    mock_client = mock_together.return_value
    mock_client.chat.completions.create.return_value = AsyncMock()
    mock_client.chat.completions.create.return_value.choices = [
        AsyncMock(message=AsyncMock(content="Mocked response"))
    ]
    
    # Create service with mocked client
    service = AprielGenerationService(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        system_prompt="You are a helpful assistant.",
        timeout=30
    )
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Ethereum?"}
    ]
    
    # Call the method
    response = await service.chat(
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )
    
    # Verify response
    assert isinstance(response, str)
    assert response == "Mocked response"
    
    # Verify the API was called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"] == messages
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100

@pytest.mark.asyncio
async def test_generate_with_retries(mock_together):
    """Test that the service retries on transient errors."""
    # Setup mock to fail once then succeed
    mock_client = mock_together.return_value
    
    # First call raises an exception, second call succeeds
    mock_client.chat.completions.create.side_effect = [
        Exception("Temporary error"),
        AsyncMock(choices=[AsyncMock(message=AsyncMock(content="Retry success"))])
    ]
    
    # Create service with mocked client
    service = AprielGenerationService(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        system_prompt="You are a helpful assistant.",
        timeout=30
    )
    
    # Call the method with retries
    response = await service.generate(
        prompt="Test retry",
        max_retries=2
    )
    
    # Verify response after retry
    assert response == "Retry success"
    assert mock_client.chat.completions.create.call_count == 2

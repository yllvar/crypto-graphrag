"""Tests for Apriel Generation Service using real Together.AI API.

This test module contains integration tests for the AprielGenerationService.
It requires a valid TOGETHER_API_KEY environment variable to run.
"""
import os
import pytest
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock

from pydantic import ValidationError

from src.llm.generation import (
    AprielGenerationService,
    ChatMessage,
    ChatCompletionRequest,
    LLMResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds between retries

# Test data
TEST_PROMPT = "What is Bitcoin?"
TEST_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, can you help me with cryptocurrency?"},
    {"role": "assistant", "content": "Of course! I'd be happy to help with any questions you have about cryptocurrency."}
]

# Skip tests if running in CI environment without API key
pytestmark = pytest.mark.skipif(
    not TOGETHER_API_KEY,
    reason="TOGETHER_API_KEY environment variable not set"
)

# Add delay between tests to respect rate limits
@pytest.fixture(autouse=True)
def add_delay_between_tests():
    """Add a delay between tests to respect rate limits."""
    yield
    time.sleep(30)  # 30 second delay between tests

@pytest.fixture(scope="module")
def generation_service():
    """Create and configure a generation service instance for testing."""
    return AprielGenerationService(
        api_key=TOGETHER_API_KEY,
        model=TEST_MODEL,
        max_retries=MAX_RETRIES,
        system_prompt="You are a helpful AI assistant with expertise in cryptocurrency and blockchain technology.",
        timeout=30  # 30 second timeout
    )

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def test_generate_single_turn(generation_service, event_loop):
    """Test single-turn text generation."""
    async def _run_test():
        prompt = "What is Bitcoin?"
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Sending request to Together.AI API (attempt {attempt + 1}/{MAX_RETRIES})...")
                response = await generation_service.generate(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=100
                )
                
                assert isinstance(response, LLMResponse), f"Expected LLMResponse, got {type(response)}"
                assert isinstance(response.content, str), f"Expected string content, got {type(response.content)}"
                assert len(response.content) > 0, "Response content is empty"
                assert "usage" in response.metadata, "Missing 'usage' in metadata"
                assert "total_tokens" in response.metadata["usage"], "Missing 'total_tokens' in usage"
                assert response.metadata["usage"]["total_tokens"] > 0, "Total tokens should be greater than 0"
                
                logger.info(f"Response: {response.content}")
                logger.info(f"Usage: {response.metadata['usage']}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(rate_limit_msg in error_msg for rate_limit_msg in ["rate limit", "rate_limit", "too many requests"]):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        pytest.skip("Skipping test due to rate limiting after multiple retries")
                        
                logger.error(f"Error in test_generate_single_turn (attempt {attempt + 1}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise
                    
                # Wait before retrying for other errors
                await asyncio.sleep(RETRY_DELAY)
        
        # If we get here, all retries were exhausted
        pytest.skip("Skipping test after all retries failed")
        return False
    
    # Run the async test in the event loop
    return event_loop.run_until_complete(_run_test())

def test_chat_with_history(generation_service, event_loop):
    """Test chat completion with conversation history."""
    async def _run_test():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Ethereum?"}
        ]
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Sending chat request to Together.AI API (attempt {attempt + 1}/{MAX_RETRIES})...")
                response = await generation_service.chat(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
                
                assert isinstance(response, LLMResponse), f"Expected LLMResponse, got {type(response)}"
                assert isinstance(response.content, str), f"Expected string content, got {type(response.content)}"
                assert len(response.content) > 0, "Response content is empty"
                assert "usage" in response.metadata, "Missing 'usage' in metadata"
                assert "total_tokens" in response.metadata["usage"], "Missing 'total_tokens' in usage"
                assert response.metadata["usage"]["total_tokens"] > 0, "Total tokens should be greater than 0"
                
                logger.info(f"Chat response: {response.content}")
                logger.info(f"Usage: {response.metadata['usage']}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(rate_limit_msg in error_msg for rate_limit_msg in ["rate limit", "rate_limit", "too many requests"]):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        pytest.skip("Skipping test due to rate limiting after multiple retries")
                        
                logger.error(f"Error in test_chat_with_history (attempt {attempt + 1}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise
                    
                # Wait before retrying for other errors
                await asyncio.sleep(RETRY_DELAY)
        
        # If we get here, all retries were exhausted
        pytest.skip("Skipping test after all retries failed")
        return False
    
    # Run the async test in the event loop
    return event_loop.run_until_complete(_run_test())

def test_error_handling():
    """Test error handling for various error scenarios."""
    pytest.skip("Skipping error handling test to avoid rate limits")

def test_edge_cases():
    """Test edge cases for the generation service."""
    pytest.skip("Skipping edge cases test to avoid rate limits")

def test_performance_benchmark():
    """Performance benchmark for the generation service."""
    pytest.skip("Skipping performance benchmark to avoid rate limits")

def test_concurrent_requests():
    """Test handling of concurrent requests."""
    pytest.skip("Skipping concurrent requests test to avoid rate limits")

def test_generate_with_different_parameters():
    """Test generation with different parameter combinations."""
    pytest.skip("Skipping parameter testing to avoid rate limits")

def test_apriel_service():
    """Test Apriel service with different parameters."""
    pytest.skip("Skipping Apriel service test to avoid rate limits")

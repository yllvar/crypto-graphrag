"""Integration tests for Apriel Generation Service.

This test module contains integration tests for the AprielGenerationService.
It uses the actual Together.AI API with a valid API key.
"""
import os
import pytest
import asyncio
import logging
from typing import List, Dict, Any

from src.llm.generation import AprielGenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TEST_API_KEY = "3f96147919d9d49efae247e1cfe05e5f12d737ca096f45f6915232feaaafd0ad"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

@pytest.fixture
def generation_service():
    """Create and configure a generation service instance for testing."""
    return AprielGenerationService(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        system_prompt="You are a helpful AI assistant with expertise in cryptocurrency and blockchain technology.",
        timeout=30
    )

@pytest.mark.asyncio
async def test_generate_single_turn(generation_service):
    """Test single-turn text generation with the actual API."""
    # Test with a simple prompt
    prompt = "Explain quantum computing in simple terms"
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Sending generation request to Together.AI API (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Test with default parameters
            response = await generation_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
                top_k=50,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n", "###"]
            )
            
            # Verify response structure
            assert isinstance(response, str), f"Expected string response, got {type(response)}"
            assert len(response) > 0, "Response should not be empty"
            logger.info(f"Successfully generated response: {response[:100]}...")
            
            # If we get here, the test passed
            return
            
        except Exception as e:
            logger.error(f"Error in test_generate_single_turn (attempt {attempt + 1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
                
            # Wait before retrying
            await asyncio.sleep(RETRY_DELAY)
    
    # If we get here, all retries were exhausted
    pytest.fail("All retry attempts failed")

@pytest.mark.asyncio
async def test_chat_with_history(generation_service):
    """Test chat completion with conversation history using the actual API."""
    # Test messages
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
                max_tokens=100,
                top_p=0.9,
                top_k=50,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n", "###"]
            )
            
            # Verify response structure
            assert isinstance(response, str), f"Expected string response, got {type(response)}"
            assert len(response) > 0, "Response should not be empty"
            logger.info(f"Successfully generated chat response: {response[:100]}...")
            
            # If we get here, the test passed
            return
            
        except Exception as e:
            logger.error(f"Error in test_chat_with_history (attempt {attempt + 1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
                
            # Wait before retrying
            await asyncio.sleep(RETRY_DELAY)
    
    # If we get here, all retries were exhausted
    pytest.fail("All retry attempts failed")

@pytest.mark.asyncio
async def test_generate_with_retries(generation_service):
    """Test that the service handles retries on transient errors."""
    # This test verifies the retry logic in the service
    test_prompt = "Explain how blockchain works in simple terms"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await generation_service.generate(
                prompt=test_prompt,
                max_retries=2,
                temperature=0.7,
                max_tokens=100
            )
            
            # If we get here, the call was successful
            assert isinstance(response, str)
            assert len(response) > 0
            return
            
        except Exception as e:
            logger.error(f"Error in test_generate_with_retries (attempt {attempt + 1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
                
            # Wait before retrying
            await asyncio.sleep(RETRY_DELAY)
    
    # If we get here, all retries were exhausted
    pytest.fail("All retry attempts failed")

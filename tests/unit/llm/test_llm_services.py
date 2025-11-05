"""Tests for LLM services using real Together.AI API.

This test module contains integration tests for the LLM services.
It requires a valid TOGETHER_API_KEY environment variable to run.
"""
import os
import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Tuple, Optional

from src.llm.embedding import M2BertEmbeddingService
from src.llm.generation import LlamaGenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip tests if running in CI environment without API key
pytestmark = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY environment variable not set"
)

# Test configuration
TEST_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Model for text generation
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"  # Model for embeddings
MAX_RETRIES = 3  # Maximum number of retries for rate-limited requests
RETRY_DELAY = 5  # Delay between retries in seconds

# Test data
TEST_PROMPTS = [
    "Explain what blockchain is in one sentence.",
    "What is the difference between Bitcoin and Ethereum?",
    "How does proof of stake work?"
]

TEST_TEXTS = [
    "Bitcoin is a decentralized digital currency.",
    "Ethereum is a platform for decentralized applications.",
    "Cryptocurrency markets are highly volatile.",
    "Smart contracts are self-executing contracts with the terms directly written into code.",
    "DeFi stands for decentralized finance and aims to recreate traditional financial systems without intermediaries."
]

@pytest.fixture(scope="module")
def test_texts() -> List[str]:
    """Provide sample texts for testing."""
    return TEST_TEXTS
    
@pytest.fixture
def embedding_service() -> M2BertEmbeddingService:
    """Create and configure an embedding service instance for testing."""
    return M2BertEmbeddingService(
        api_key=os.getenv("TOGETHER_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    )
    
@pytest.fixture
def generation_service() -> LlamaGenerationService:
    """Create and configure a generation service instance for testing."""
    return LlamaGenerationService(
        api_key=os.getenv("TOGETHER_API_KEY"),
        model=TEST_MODEL
    )

@pytest.fixture
def mock_embedding_response() -> List[float]:
    """Provide a mock embedding response for testing."""
    # Return a simple embedding vector for testing
    return [0.1 * i for i in range(768)]  # Assuming 768-dimensional embeddings

@pytest.fixture
def mock_generation_response() -> Tuple[str, Dict[str, Any]]:
    """Provide a mock generation response for testing."""
    return (
        "Blockchain is a decentralized digital ledger that records transactions across many computers.",
        {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    )

@pytest.mark.asyncio
async def test_embedding_service_generate(embedding_service, test_texts):
    """Test M2BertEmbeddingService generate method with real API."""
    text = test_texts[0]
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} - Generating embedding for text: {text[:50]}...")
            embedding = await embedding_service.generate(text)
            
            # Validate the embedding
            assert isinstance(embedding, list), f"Expected list, got {type(embedding)}"
            assert len(embedding) > 0, "Embedding should not be empty"
            assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
            
            # If we got here, the test passed
            return
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["rate limit", "overloaded", "503"]):
                if attempt == MAX_RETRIES - 1:  # Last attempt
                    pytest.skip(f"Skipping test after {MAX_RETRIES} attempts due to rate limiting: {str(e)}")
                continue
            else:
                pytest.fail(f"Embedding generation failed with error: {str(e)}")
    
    pytest.skip("Max retries reached without completing the test")


@pytest.mark.asyncio
async def test_embedding_service_batch(embedding_service, test_texts):
    """Test batch embedding generation with multiple texts."""
    if not hasattr(embedding_service, 'batch_generate'):
        pytest.skip("Batch generation not implemented in this version")
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} - Generating batch embeddings for {len(test_texts)} texts")
            
            # Test with a subset of texts to avoid hitting rate limits
            batch = test_texts[:2]  # Only use first two texts
            embeddings = await embedding_service.batch_generate(batch)
            
            # Validate the response
            assert isinstance(embeddings, list), "Expected list of embeddings"
            assert len(embeddings) == len(batch), "Number of embeddings should match input"
            
            for emb in embeddings:
                assert isinstance(emb, list), "Each embedding should be a list"
                assert len(emb) > 0, "Embedding should not be empty"
                assert all(isinstance(x, float) for x in emb), "All elements should be floats"
            
            return
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["rate limit", "overloaded", "503"]):
                if attempt == MAX_RETRIES - 1:
                    pytest.skip(f"Skipping test after {MAX_RETRIES} attempts due to rate limiting: {str(e)}")
                continue
            elif "not implemented" in error_msg:
                pytest.skip("Batch generation not implemented")
            else:
                pytest.fail(f"Batch embedding generation failed: {str(e)}")
    
    pytest.skip("Max retries reached without completing the test")


@pytest.mark.asyncio
async def test_generation_service_generate(generation_service):
    """Test LlamaGenerationService generate method with real API."""
    prompt = TEST_PROMPTS[0]
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} - Generating text for prompt: {prompt[:50]}...")
            
            # The response is a tuple of (content, metadata)
            response, metadata = await generation_service.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )
            
            # Handle response format
            if isinstance(response, tuple) and len(response) == 2 and response[0] == 'content':
                response = response[1]
            
            # Verify response
            assert isinstance(response, str), f"Expected string response, got {type(response)}"
            assert len(response) > 0, "Response should not be empty"
            
            # Handle metadata
            if isinstance(metadata, tuple) and len(metadata) == 2 and metadata[0] == 'metadata':
                metadata = metadata[1]
            
            # Extract usage
            usage = metadata.get('usage', metadata) if isinstance(metadata, dict) else metadata
            
            # Verify usage
            assert isinstance(usage, dict), f"Expected dict usage, got {type(usage)}"
            required_keys = ["prompt_tokens", "completion_tokens", "total_tokens"]
            missing_keys = [key for key in required_keys if key not in usage]
            assert not missing_keys, f"Missing required keys in usage: {missing_keys}"
            
            # If we got here, the test passed
            return
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["rate limit", "overloaded", "503"]):
                if attempt == MAX_RETRIES - 1:  # Last attempt
                    pytest.skip(f"Skipping test after {MAX_RETRIES} attempts due to rate limiting: {str(e)}")
                continue
            else:
                pytest.fail(f"Text generation failed with error: {str(e)}")
    
    pytest.skip("Max retries reached without completing the test")


@pytest.mark.asyncio
async def test_generation_with_different_parameters(generation_service):
    """Test text generation with different parameters."""
    test_cases = [
        {"temperature": 0.5, "max_tokens": 30, "top_p": 0.9},
        {"temperature": 0.8, "max_tokens": 50, "top_k": 50},
        {"temperature": 1.0, "max_tokens": 20, "repetition_penalty": 1.2}
    ]
    
    for params in test_cases:
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                
                logger.info(f"Testing with params: {params}")
                
                response, _ = await generation_service.generate(
                    prompt=TEST_PROMPTS[1],
                    **params
                )
                
                # Basic validation
                if isinstance(response, tuple) and len(response) == 2 and response[0] == 'content':
                    response = response[1]
                
                assert isinstance(response, str), f"Expected string response, got {type(response)}"
                assert len(response) > 0, "Response should not be empty"
                
                # If we got here, this test case passed
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(err in error_msg for err in ["rate limit", "overloaded", "503"]):
                    if attempt == MAX_RETRIES - 1:
                        pytest.skip(f"Skipping test case after {MAX_RETRIES} attempts: {str(e)}")
                    continue
                else:
                    pytest.fail(f"Generation with params {params} failed: {str(e)}")
        else:
            pytest.skip("Max retries reached for this test case")
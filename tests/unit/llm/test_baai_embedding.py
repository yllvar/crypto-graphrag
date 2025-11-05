"""Tests for BAAI Embedding Service using real Together.AI API.

This test module contains integration tests for the BAAIEmbeddingService.
It requires a valid TOGETHER_API_KEY environment variable to run.
"""
import os
import pytest
import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from pydantic import ValidationError

from src.llm.embedding import BAAIEmbeddingService, EmbeddingRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip tests if running in CI environment without API key
pytestmark = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY environment variable not set"
)

# Test configuration
TEST_MODEL = "BAAI/bge-base-en-v1.5"
MAX_RETRIES = 3
RETRY_DELAY = 5

# Test data
TEST_TEXTS = [
    "Bitcoin is a decentralized digital currency.",
    "Ethereum is a platform for decentralized applications.",
    "Cryptocurrency markets are highly volatile.",
    "Smart contracts are self-executing contracts with the terms directly written into code.",
    "DeFi stands for decentralized finance and aims to recreate traditional financial systems without intermediaries."
]

@pytest.fixture(scope="module")
def embedding_service():
    """Create and configure an embedding service instance for testing."""
    return BAAIEmbeddingService(
        model=TEST_MODEL,
        batch_size=3,  # Small batch size for testing
        max_retries=MAX_RETRIES
    )

@pytest.mark.asyncio
async def test_generate_embedding(embedding_service):
    """Test generating a single embedding."""
    text = "This is a test sentence for embedding."
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await embedding_service.generate(text)
            
            # Verify response structure
            assert isinstance(response.content, list)
            assert len(response.content) == 768  # BAAI/bge-base-en-v1.5 has 768-dimensional embeddings
            assert all(isinstance(x, float) for x in response.content)
            
            # Verify metadata
            assert "model" in response.metadata
            assert response.metadata["model"] == TEST_MODEL
            assert "vector_size" in response.metadata
            assert response.metadata["vector_size"] == 768
            
            return  # Test passed
            
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                logger.warning(f"Rate limited, attempt {attempt + 1}/{MAX_RETRIES}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    
    pytest.skip("Skipping test due to rate limiting")

@pytest.mark.asyncio
async def test_batch_generate_embeddings(embedding_service):
    """Test batch generation of embeddings."""
    texts = TEST_TEXTS[:3]  # Test with first 3 texts
    
    for attempt in range(MAX_RETRIES):
        try:
            responses = await embedding_service.batch_generate(texts)
            
            # Verify response structure
            assert isinstance(responses, list)
            assert len(responses) == len(texts)
            
            # Verify each embedding
            for i, response in enumerate(responses):
                assert isinstance(response.content, list)
                assert len(response.content) == 768
                assert all(isinstance(x, float) for x in response.content)
                
                # Verify metadata
                assert response.metadata["batch_index"] == i
                assert "usage" in response.metadata
                
            return  # Test passed
            
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                logger.warning(f"Rate limited, attempt {attempt + 1}/{MAX_RETRIES}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    
    pytest.skip("Skipping test due to rate limiting")

@pytest.mark.asyncio
async def test_large_batch(embedding_service):
    """Test with a batch larger than the service's batch size."""
    # Create a larger batch of texts
    texts = TEST_TEXTS * 2  # 10 texts total
    
    for attempt in range(MAX_RETRIES):
        try:
            responses = await embedding_service.batch_generate(
                texts,
                batch_size=2  # Force small batch size
            )
            
            # Verify all embeddings were generated
            assert len(responses) == len(texts)
            
            # Verify each embedding
            for response in responses:
                assert isinstance(response.content, list)
                assert len(response.content) == 768
                
            return  # Test passed
            
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                logger.warning(f"Rate limited, attempt {attempt + 1}/{MAX_RETRIES}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    
    pytest.skip("Skipping test due to rate limiting")

@pytest.mark.asyncio
async def test_edge_cases(embedding_service):
    """Test edge cases like empty string and very long text."""
    # Test empty string
    with pytest.raises(ValueError):
        await embedding_service.generate("")
    
    # Test very long text (should be truncated)
    long_text = "This is a very long text. " * 1000
    response = await embedding_service.generate(long_text, truncate=True)
    assert len(response.content) == 768
    
    # Test very long text without truncation (should fail)
    with pytest.raises(ValueError):
        await embedding_service.generate(long_text, truncate=False)
    
    # Test with special characters and emojis
    special_text = "Testing special chars: 😊 #test @user 123!@#"
    response = await embedding_service.generate(special_text)
    assert len(response.content) == 768

@pytest.mark.asyncio
async def test_invalid_inputs(embedding_service):
    """Test handling of invalid inputs."""
    # Test with invalid API key
    with patch('together.AsyncTogether') as mock_together:
        mock_together.return_value.embeddings.create.side_effect = Exception("API Error")
        service = BAAIEmbeddingService(api_key="invalid_key")
        
        with pytest.raises(RuntimeError):
            await service.generate("Test text")
    
    # Test with invalid model
    with pytest.raises(ValueError):
        BAAIEmbeddingService(model="invalid-model")

@pytest.mark.asyncio
async def test_batch_processing(embedding_service):
    """Test batch processing with various batch sizes."""
    # Create test texts with different lengths
    test_texts = [
        "Short text",
        "Medium length text with some more words",
        "A very long text with many words " * 10,
        "Another text with special characters: !@#$%^&*()",
        "Text with emoji 😊 and numbers 123"
    ]
    
    # Test with different batch sizes
    for batch_size in [1, 2, 5]:
        responses = await embedding_service.batch_generate(
            test_texts,
            batch_size=batch_size
        )
        
        # Verify all texts were processed
        assert len(responses) == len(test_texts)
        
        # Verify embeddings have correct dimensions
        for response in responses:
            assert len(response.content) == 768
            assert all(isinstance(x, float) for x in response.content)

@pytest.mark.asyncio
async def test_performance_benchmark(embedding_service):
    """Performance benchmark for the embedding service."""
    # Generate test data
    short_text = "Short text"
    medium_text = "This is a medium length text with more words and context."
    long_text = "A very long text. " * 100
    
    test_cases = [
        ("single_short", [short_text] * 10),
        ("single_medium", [medium_text] * 10),
        ("single_long", [long_text] * 5),
        ("mixed_lengths", [short_text, medium_text, long_text] * 3)
    ]
    
    results = []
    
    for name, texts in test_cases:
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            start_time = time.time()
            
            # Process the batch
            responses = await embedding_service.batch_generate(
                texts,
                batch_size=batch_size
            )
            
            total_time = time.time() - start_time
            total_tokens = sum(
                len(text.split()) * 1.3  # Approximate tokens
                for text in texts
            )
            
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            results.append({
                "test_case": name,
                "batch_size": batch_size,
                "num_texts": len(texts),
                "total_chars": sum(len(t) for t in texts),
                "total_time": total_time,
                "tokens_per_second": tokens_per_second,
                "texts_per_second": len(texts) / total_time if total_time > 0 else 0
            })
    
    # Log the results
    logger.info("\nEmbedding Performance Benchmark Results:")
    for result in results:
        logger.info(
            f"{result['test_case']} (batch_size={result['batch_size']}): "
            f"{result['total_time']:.2f}s, "
            f"{result['tokens_per_second']:.1f} tokens/sec, "
            f"{result['texts_per_second']:.1f} texts/sec"
        )
    
    # Assert that performance is reasonable
    assert all(r["tokens_per_second"] > 10 for r in results), "Performance test failed: Too slow"

@pytest.mark.asyncio
async def test_concurrent_requests(embedding_service):
    """Test handling of concurrent embedding requests."""
    async def embed_text(text: str) -> List[float]:
        response = await embedding_service.generate(text)
        return response.content
    
    # Create test texts
    test_texts = [f"Test text {i}" for i in range(5)]
    
    # Create and run tasks
    tasks = [embed_text(text) for text in test_texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify results
    assert len(results) == len(test_texts)
    for i, result in enumerate(results):
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

@pytest.mark.asyncio
async def test_embedding_quality(embedding_service):
    """Test the quality of embeddings by comparing similarities."""
    # Similar texts should have similar embeddings
    text1 = "The price of Bitcoin is going up"
    text2 = "Bitcoin's value is increasing"
    text3 = "The weather is nice today"
    
    # Get embeddings
    emb1 = (await embedding_service.generate(text1)).content
    emb2 = (await embedding_service.generate(text2)).content
    emb3 = (await embedding_service.generate(text3)).content
    
    # Calculate cosine similarities
    def cosine_sim(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0
        return np.dot(a, b) / (a_norm * b_norm)
    
    sim_similar = cosine_sim(emb1, emb2)
    sim_different = cosine_sim(emb1, emb3)
    
    logger.info(f"Similar texts similarity: {sim_similar:.4f}")
    logger.info(f"Different texts similarity: {sim_different:.4f}")
    
    # Similar texts should be more similar than different ones
    assert sim_similar > sim_different, "Similarity test failed"
    assert sim_similar > 0.7, "Similar texts should be very similar"
    assert sim_different < 0.5, "Different texts should be less similar"

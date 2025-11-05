"""BAAI Embedding Service.

Implements text embedding functionality using BAAI/bge-base-en-v1.5 model.
"""
import logging
from typing import List, Optional, Dict, Any, Union

from together import AsyncTogether
from pydantic import BaseModel, Field

from .base import BaseLLMService, LLMResponse
from ..utils.config import settings

logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    model: str
    input: Union[str, List[str]]
    truncate: bool = True

class BAAIEmbeddingService(BaseLLMService):
    """Service for generating text embeddings using BAAI/bge-base-en-v1.5 model."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
        max_retries: int = 3,
        **kwargs
    ):
        """Initialize the BAAI embedding service.
        
        Args:
            api_key: Together.AI API key. If None, uses TOGETHER_API_KEY from settings.
            model: Model name. Defaults to "BAAI/bge-base-en-v1.5".
            batch_size: Number of texts to process in a single batch.
            max_retries: Maximum number of retries for failed requests.
            **kwargs: Additional parameters for the Together client.
        """
        self.api_key = api_key or settings.TOGETHER_API_KEY
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.client = AsyncTogether(api_key=self.api_key, **kwargs)
        
        # Model configuration
        self.vector_size = 768  # BAAI/bge-base-en-v1.5 has 768-dimensional embeddings
        self.max_seq_length = 512  # Maximum sequence length for the model
        
        logger.info("Initialized BAAIEmbeddingService with model: %s", self.model)
    
    async def generate(self, text: str, **kwargs) -> LLMResponse:
        """Generate embeddings for a single text.
        
        Args:
            text: Input text to embed
            **kwargs: Additional parameters for the embedding request
                - truncate: Whether to truncate the input text if it's too long (default: True)
                - normalize: Whether to normalize the embeddings (default: True)
            
        Returns:
            LLMResponse containing the embedding vector and metadata
            
        Raises:
            ValueError: If the input text is empty or too long
            RuntimeError: If the API request fails after max_retries
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        if len(text) > self.max_seq_length * 4:  # Rough estimate of token count
            if kwargs.get("truncate", True):
                text = text[:self.max_seq_length * 4]  # Truncate long text
            else:
                raise ValueError(f"Input text is too long (max {self.max_seq_length * 4} characters)")
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    **{k: v for k, v in kwargs.items() if k != "truncate"}
                )
                
                # Extract the embedding
                embedding = response.data[0].embedding
                
                # Ensure the embedding has the expected dimensions
                if len(embedding) != self.vector_size:
                    logger.warning(
                        f"Unexpected embedding size: {len(embedding)} (expected {self.vector_size})"
                    )
                
                return LLMResponse(
                    content=embedding,
                    metadata={
                        "model": self.model,
                        "vector_size": len(embedding),
                        "usage": {
                            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                            "total_tokens": getattr(response.usage, "total_tokens", 0),
                        },
                        "params": kwargs
                    }
                )
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    logger.warning(
                        f"Attempt {retry_count}/{self.max_retries} failed: {str(e)}. Retrying..."
                    )
                    continue
                
                error_msg = f"Failed to generate embedding after {self.max_retries} attempts: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    async def batch_generate(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in a single batch. If None, uses the instance's batch_size.
            **kwargs: Additional parameters for the embedding request
                - truncate: Whether to truncate the input text if it's too long (default: True)
                - normalize: Whether to normalize the embeddings (default: True)
                
        Returns:
            List of LLMResponse objects containing the embedding vectors and metadata
            
        Raises:
            ValueError: If any input text is empty or too long
            RuntimeError: If the API request fails after max_retries
        """
        if not texts:
            return []
            
        batch_size = batch_size or self.batch_size
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    **kwargs
                )
                
                # Process the batch of embeddings
                for j, embedding in enumerate(response.data):
                    results.append(
                        LLMResponse(
                            content=embedding.embedding,
                            metadata={
                                "model": self.model,
                                "vector_size": len(embedding.embedding),
                                "batch_index": i + j,
                                "usage": {
                                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                                },
                                "params": kwargs
                            }
                        )
                    )
                    
            except Exception as e:
                error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        return results
        
    def get_embedding_size(self) -> int:
        """Get the size of the embedding vectors produced by this model."""
        return self.vector_size
        
    def get_max_sequence_length(self) -> int:
        """Get the maximum sequence length supported by this model."""
        return self.max_seq_length
        
    async def get_token_count(self, text: str) -> int:
        """Estimate the number of tokens in the given text.
        
        Note: This is an approximation as the exact tokenization is model-specific.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimate: 1 token ≈ 4 characters in English
        return max(1, len(text) // 4)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        responses = await self.batch_generate(texts)
        return [response.content for response in responses]

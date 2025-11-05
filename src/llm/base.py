"""
Base LLM Service Interface

Defines the abstract base class for all LLM services in the system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized response format for LLM operations."""
    content: Any
    metadata: Dict[str, Any] = {}

class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text based on the given prompt.
        
        Args:
            prompt: The input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the given text.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass

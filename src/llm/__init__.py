"""
LLM Module for Agent-Based Graph RAG System

This module provides interfaces and implementations for working with various LLM services.
"""

from .base import BaseLLMService, LLMResponse
from .generation import AprielGenerationService, ChatMessage, ChatCompletionRequest
from .embedding import BAAIEmbeddingService, EmbeddingRequest

__all__ = [
    # Base classes
    'BaseLLMService',
    'LLMResponse',
    
    # Generation
    'AprielGenerationService',
    'ChatMessage',
    'ChatCompletionRequest',
    
    # Embedding
    'BAAIEmbeddingService',
    'EmbeddingRequest',
    
    # Legacy (for backward compatibility)
    'LlamaGenerationService',  # Deprecated: Use AprielGenerationService
    'M2BertEmbeddingService',  # Deprecated: Use BAAIEmbeddingService
]

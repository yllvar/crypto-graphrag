"""
LLM Schema Definitions

Defines Pydantic models for LLM request/response handling.
"""
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    TOGETHER_AI = "together_ai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMModelType(str, Enum):
    """Types of LLM models."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

class LLMRequest(BaseModel):
    """Base request model for LLM generation."""
    prompt: str = Field(..., description="The input prompt or message")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: List[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")

class LLMResponse(BaseModel):
    """Base response model for LLM generation."""
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""
    texts: List[str] = Field(..., description="List of texts to embed")
    model: str = Field(..., description="Embedding model identifier")
    truncate: bool = Field(True, description="Whether to truncate long texts")

class EmbeddingResponse(BaseModel):
    """Response model for text embeddings."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model used for embeddings")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

class TokenizeRequest(BaseModel):
    """Request model for tokenization."""
    text: str = Field(..., description="Text to tokenize")
    model: str = Field(..., description="Model identifier for tokenizer")

class TokenizeResponse(BaseModel):
    """Response model for tokenization."""
    tokens: List[int] = Field(..., description="List of token IDs")
    token_count: int = Field(..., description="Number of tokens")
    model: str = Field(..., description="Model used for tokenization")

class ModelInfo(BaseModel):
    """Model information and capabilities."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name")
    provider: LLMProvider = Field(..., description="Provider of the model")
    type: LLMModelType = Field(..., description="Type of the model")
    context_length: int = Field(..., description="Maximum context length in tokens")
    description: Optional[str] = Field(None, description="Model description")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities")

class LLMHealthCheck(BaseModel):
    """Health check response for LLM services."""
    status: str = Field(..., description="Service status")
    models: List[str] = Field(..., description="Available models")
    provider: str = Field(..., description="Service provider")
    version: str = Field(..., description="Service version")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ["healthy", "degraded", "unavailable"]:
            raise ValueError("Status must be one of: healthy, degraded, unavailable")
        return v

class StreamChunk(BaseModel):
    """A chunk of streaming response."""
    content: str = Field(..., description="Content chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage if final chunk")

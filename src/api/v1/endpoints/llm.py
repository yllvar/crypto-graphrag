"""
LLM API Endpoints

This module provides FastAPI endpoints for interacting with LLM services.
"""
import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from src.llm.generation import AprielGenerationService
from src.llm.embedding import BAAIEmbeddingService
from src.utils.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Initialize services
generation_service = AprielGenerationService()
embedding_service = BAAIEmbeddingService()

# Create router
router = APIRouter()

# Constants
MAX_BATCH_SIZE = 50  # Maximum number of texts to process in a single batch

# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": ["\n", "###"]
            }
        }
    )
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The input prompt for text generation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the generation (0.0 = deterministic, 2.0 = most random)"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0 to 1.0)"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling parameter (1-100)"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for frequent tokens (-2.0 to 2.0)"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for new tokens (-2.0 to 2.0)"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="List of sequences where the API will stop generating further tokens"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate that prompt is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()

class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "texts": ["This is a test sentence", "And another one"],
                "model": "BAAI/bge-base-en-v1.5"
            }
        }
    )
    
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of texts to generate embeddings for (max {MAX_BATCH_SIZE} texts per request)"
    )
    model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="The model to use for generating embeddings"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_texts(cls, data):
        if isinstance(data, dict) and not data.get("texts"):
            raise ValueError('Texts list cannot be empty')
        return data

class GenerationResponse(BaseModel):
    """Response model for text generation."""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "text": "Quantum computing is a type of computation that...",
                "model": "ServiceNow-AI/Apriel-1.5-15b-Thinker",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 100,
                    "total_tokens": 110
                }
            }
        }
    )
    
    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage information")

class EmbeddingResponse(BaseModel):
    """Response model for text embeddings."""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "embeddings": [[0.1, -0.2, 0.3, ...], [-0.1, 0.2, 0.4, ...]],
                "model": "BAAI/bge-base-en-v1.5",
                "usage": {
                    "prompt_tokens": 25,
                    "total_tokens": 25
                }
            }
        }
    )
    
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="The model used for generating embeddings")
    usage: Dict[str, int] = Field(..., description="Token usage information")

# Authentication
async def verify_api_key(api_key: str = Depends(api_key_header)) -> None:
    """Verify the API key.
    
    In a production environment, you would validate the API key against a database.
    For now, we'll just check if it matches the one in settings.
    """
    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )

# Rate limiting decorator (placeholder)
def rate_limit(requests_per_minute: int = 60):
    """Decorator to implement rate limiting."""
    # In a real implementation, use something like Redis for distributed rate limiting
    # This is a simple in-memory version for demonstration
    from fastapi import Request, HTTPException
    from fastapi import status
    import time
    
    last_called = {}
    
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "127.0.0.1"
            current_time = time.time()
            
            # Initialize or update request count
            if client_ip not in last_called:
                last_called[client_ip] = {
                    'count': 1,
                    'window_start': current_time
                }
            else:
                # Reset counter if window has passed
                if current_time - last_called[client_ip]['window_start'] > 60:
                    last_called[client_ip] = {
                        'count': 1,
                        'window_start': current_time
                    }
                else:
                    last_called[client_ip]['count'] += 1
            
            # Check rate limit
            if last_called[client_ip]['count'] > requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Endpoints
@router.post(
    "/generate",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate text",
    description="Generate text using the LLM model",
    dependencies=[Security(verify_api_key)]
)
@rate_limit(requests_per_minute=30)  # 30 requests per minute per IP
async def generate_text(
    request: GenerationRequest,
) -> Dict[str, Any]:
    """Generate text based on the given prompt."""
    try:
        # Log the request
        logger.info(f"Generating text with params: {request.dict()}")
        
        # Call the LLM service
        response = await llm_service.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # Log successful generation
        logger.info(f"Successfully generated {len(response.content)} characters")
        
        return {
            "text": response.content,
            "model": response.metadata.get("model", "unknown"),
            "usage": {
                "prompt_tokens": response.metadata.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.metadata.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.metadata.get("usage", {}).get("total_tokens", 0),
            }
        }
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating text: {str(e)}"
        )

@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Get embeddings",
    description="Get embeddings for the given texts",
    dependencies=[Security(verify_api_key)]
)
@rate_limit(requests_per_minute=60)  # 60 requests per minute per IP
async def get_embeddings(
    request: EmbeddingRequest,
) -> Dict[str, Any]:
    """Get embeddings for the given texts."""
    try:
        # Log the request
        logger.info(f"Generating embeddings for {len(request.texts)} texts")
        
        # Call the embedding service
        embeddings = await embedding_service.get_embeddings(request.texts)
        
        # Log successful generation
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        
        return {
            "embeddings": embeddings,
            "model": embedding_service.model,
            "usage": {
                "texts_processed": len(embeddings),
                "total_tokens": sum(len(text.split()) for text in request.texts)  # Approximate
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        # Simple health check - in a real app, you might want to check DB connections, etc.
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

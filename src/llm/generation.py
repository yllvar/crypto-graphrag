"""
Apriel Generation Service

Implements chat-based text generation functionality using ServiceNow-AI/Apriel-1.5-15b-Thinker model.
"""
import logging
from typing import Dict, Any, Optional, List, Union, Literal

from together import AsyncTogether
from pydantic import BaseModel, Field

from .base import BaseLLMService, LLMResponse
from ..utils.config import settings

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: bool = False

class AprielGenerationService(BaseLLMService):
    """Service for generating text using ServiceNow-AI/Apriel-1.5-15b-Thinker model."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "ServiceNow-AI/Apriel-1.5-15b-Thinker",
        system_prompt: str = "You are a helpful AI assistant.",
        **default_params
    ):
        """Initialize the Apriel generation service.
        
        Args:
            api_key: Together.AI API key. If None, uses TOGETHER_API_KEY from settings.
            model: Model name. Defaults to "ServiceNow-AI/Apriel-1.5-15b-Thinker".
            system_prompt: The system message to set the behavior of the assistant.
            **default_params: Default generation parameters
        """
        self.api_key = api_key or settings.TOGETHER_API_KEY
        self.model = model
        self.system_prompt = system_prompt
        self.client = AsyncTogether(api_key=self.api_key)
        
        # Set default generation parameters
        self.default_params = {
            "max_tokens": getattr(settings, 'LLM_MAX_TOKENS', 2048),
            "temperature": getattr(settings, 'LLM_TEMPERATURE', 0.7),
            "top_p": 0.9,
            "stop": ["</s>", "###"],
            **default_params
        }
        
        logger.info("Initialized AprielGenerationService with model: %s", self.model)
    
    async def generate(
        self, 
        prompt: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text based on the given prompt and chat history.
        
        Args:
            prompt: The user's input text
            chat_history: List of previous messages in the conversation
                Each message should be a dict with 'role' and 'content' keys
                Example: [{"role": "user", "content": "Hello!"}]
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Merge default parameters with provided ones
        params = {**self.default_params, **kwargs}
        
        # Prepare messages with system prompt and chat history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
            
        # Add the current user message
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Call the Together.AI chat completions API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                stop=params.get("stop"),
                stream=False
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            # Extract usage information if available
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            
            # Prepare metadata
            metadata = {
                "model": self.model,
                "usage": usage,
                "params": {
                    k: v for k, v in params.items()
                    if k in ["temperature", "max_tokens", "top_p", "stop"]
                }
            }
            
            return LLMResponse(
                content=generated_text,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion based on the conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Merge default parameters with provided ones
        params = {**self.default_params, **kwargs}
        
        try:
            # Call the Together.AI chat completions API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                stop=params.get("stop"),
                stream=False
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            # Extract usage information if available
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            
            # Prepare metadata
            metadata = {
                "model": self.model,
                "usage": usage,
                "params": {
                    k: v for k, v in params.items()
                    if k in ["temperature", "max_tokens", "top_p", "stop"]
                }
            }
            
            return LLMResponse(
                content=generated_text,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error in chat completion: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the M2-BERT model.
        
        This is a convenience method that creates an instance of M2BertEmbeddingService
        to maintain the BaseLLMService interface.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        from .embedding import M2BertEmbeddingService
        embedding_service = M2BertEmbeddingService()
        return await embedding_service.get_embeddings(texts)
    
    async def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the given text.
        
        Note: This is an estimate as the actual tokenization happens on the server.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimate: ~4 characters per token on average for English text
        return max(1, len(text) // 4)
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream generated text in real-time.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text as they become available
        """
        try:
            # Merge default params with any provided overrides
            params = {**self.default_params, **kwargs, "stream": True}
            
            # Make the streaming API call
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            # Stream the response
            async for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("Error in streaming generation: %s", str(e))
            raise

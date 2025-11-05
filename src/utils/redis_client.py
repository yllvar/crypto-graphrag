import json
import asyncio
import logging
from typing import Any, Dict, Optional, AsyncGenerator
from redis.asyncio import Redis
from ..utils.config import settings
from ..utils.logger import logger

class RedisClient:
    """Redis client wrapper for handling pub/sub and stream operations"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.pubsub = None
        self._is_connected = False
    
    async def connect(self):
        """Initialize Redis connection"""
        if not self._is_connected:
            try:
                self.redis = Redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis.ping()
                self.pubsub = self.redis.pubsub()
                self._is_connected = True
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Redis connection error: {e}")
                raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis and self._is_connected:
            await self.redis.close()
            self._is_connected = False
            logger.info("Disconnected from Redis")
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish message to a channel"""
        if not self._is_connected:
            await self.connect()
        
        try:
            return await self.redis.publish(
                channel,
                json.dumps(message)
            )
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")
            raise
    
    async def subscribe(self, channel: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to a channel and yield messages"""
        if not self._is_connected:
            await self.connect()
        
        try:
            await self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            
            while True:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message:
                    try:
                        data = json.loads(message["data"])
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {message}")
                
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            await self.pubsub.unsubscribe(channel)
            logger.info(f"Unsubscribed from channel: {channel}")
        except Exception as e:
            logger.error(f"Error in Redis subscription: {e}")
            raise
    
    async def add_to_stream(
        self,
        stream_name: str,
        data: Dict[str, Any],
        max_length: int = 1000
    ) -> str:
        """Add data to a Redis stream"""
        if not self._is_connected:
            await self.connect()
        
        try:
            # Add data to stream
            stream_id = await self.redis.xadd(
                name=stream_name,
                fields={"data": json.dumps(data)},
                maxlen=max_length,
                approximate=True
            )
            return stream_id
        except Exception as e:
            logger.error(f"Error adding to Redis stream: {e}")
            raise
    
    async def read_stream(
        self,
        stream_name: str,
        last_id: str = "0-0",
        count: Optional[int] = None,
        block: Optional[int] = None
    ) -> list:
        """Read data from a Redis stream"""
        if not self._is_connected:
            await self.connect()
        
        try:
            messages = await self.redis.xread(
                streams={stream_name: last_id},
                count=count,
                block=block
            )
            
            result = []
            for stream in messages:
                for message_id, message in stream[1]:
                    try:
                        data = json.loads(message["data"])
                        result.append({
                            "id": message_id,
                            "data": data
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid message format: {message}")
            
            return result
        except Exception as e:
            logger.error(f"Error reading from Redis stream: {e}")
            raise

# Global Redis client instance
redis_client = RedisClient()

# Initialize Redis connection on import
async def init_redis():
    await redis_client.connect()

# Clean up on exit
async def close_redis():
    await redis_client.disconnect()

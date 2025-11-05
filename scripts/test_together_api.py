"""Test script to verify Together.AI API connectivity."""
import asyncio
import os
import ssl
import certifi
import aiohttp
from together import AsyncTogether

# Configuration
API_KEY = "3f96147919d9d49efae247e1cfe05e5f12d737ca096f45f6915232feaaafd0ad"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Set up SSL context for all HTTP requests
ssl_context = ssl.create_default_context(cafile=certifi.where())

async def test_api_connection():
    """Test connection to Together.AI API."""
    try:
        # Configure aiohttp to use our SSL context
        conn = aiohttp.TCPConnector(ssl=ssl_context)
        
        # Initialize the client with a custom HTTP client
        client = AsyncTogether(
            api_key=API_KEY,
            timeout=30  # 30 second timeout
        )
        
        print("Testing API connection...")
        
        # Test a simple completion
        print("Sending test request...")
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=50,
            temperature=0.7
        )
        
        print("\nAPI Connection Successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        # Print usage information
        if hasattr(response, 'usage'):
            print("\nUsage:")
            for k, v in response.usage.items():
                print(f"  {k}: {v}")
        
    except Exception as e:
        print(f"\nError connecting to Together.AI API: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify the API key is correct")
        print("3. Check if the model name is correct")
        print("4. Try running 'python -m pip install --upgrade certifi'")
        print("5. Check if your system time is correct")
        print("6. Try running 'python -m pip install --upgrade together'")
        
        # Print more detailed error information
        import traceback
        print("\nDetailed error traceback:")
        traceback.print_exc()
    finally:
        # Clean up the connector
        if 'conn' in locals():
            await conn.close()

if __name__ == "__main__":
    print("Testing Together.AI API connectivity...")
    print(f"Using model: {MODEL}")
    asyncio.run(test_api_connection())

"""Test script to verify Together.AI API connectivity using requests."""
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
API_KEY = "3f96147919d9d49efae247e1cfe05e5f12d737ca096f45f6915232feaaafd0ad"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
API_URL = "https://api.together.xyz/v1/chat/completions"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def test_api_connection():
    """Test connection to Together.AI API using requests."""
    try:
        # Create a session
        session = create_session()
        
        # Prepare the request data
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print("Testing API connection...")
        print(f"Sending request to: {API_URL}")
        print(f"Using model: {MODEL}")
        
        # Make the request
        response = session.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=30,
            verify=True  # Enable SSL verification
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        print("\nAPI Connection Successful!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        
        # Print usage information if available
        if 'usage' in result:
            print("\nUsage:")
            for k, v in result['usage'].items():
                print(f"  {k}: {v}")
        
        return True
        
    except requests.exceptions.SSLError as e:
        print("\nSSL Certificate Verification Failed!")
        print("This usually means your system's SSL certificates are outdated.")
        print("You can try one of the following solutions:")
        print("1. Update your system's CA certificates")
        print("2. Run 'pip install --upgrade certifi'")
        print("3. On macOS, run '/Applications/Python\ 3.11/Install\ Certificates.command'")
        print(f"\nError details: {str(e)}")
        
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to Together.AI API: {str(e)}")
        
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Response text: {e.response.text}")
    
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        print("\nDetailed error traceback:")
        traceback.print_exc()
    
    return False

if __name__ == "__main__":
    print("Testing Together.AI API connectivity using requests...")
    success = test_api_connection()
    
    if not success:
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify the API key is correct")
        print("3. Try running 'pip install --upgrade certifi'")
        print("4. On macOS, run '/Applications/Python\\ 3.11/Install\\ Certificates.command'")
        print("5. Check if your system time is correct")

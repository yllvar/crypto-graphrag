"""Script to run LLM tests one at a time with delays to respect rate limits."""
import asyncio
import time
import subprocess
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_FILES = [
    "tests/unit/llm/test_apriel_service.py",
    "tests/unit/llm/test_baai_embedding.py"
]

# Delay between tests in seconds (adjust based on rate limits)
DELAY_BETWEEN_TESTS = 30  # 30 seconds between tests
MAX_RETRIES = 3
RETRY_DELAY = 60  # 60 seconds between retries

def run_test(test_file: str, test_name: str = None) -> bool:
    """Run a single test with retry logic."""
    cmd = ["poetry", "run", "pytest", "-v"]
    
    if test_name:
        cmd.extend(["-k", test_name])
    
    cmd.append(test_file)
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Running test: {test_file}{'::' + test_name if test_name else ''} (Attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Run the test
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log the output
            logger.info(f"Test output:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Test error output:\n{result.stderr}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            
            # Check for rate limiting
            if any(rate_limit_msg in error_msg.lower() 
                  for rate_limit_msg in ["rate limit", "rate_limit", "too many requests"]):
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"Rate limited. Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)
                continue
                
            # For other errors, log and re-raise
            logger.error(f"Test failed with error: {error_msg}")
            if attempt == MAX_RETRIES - 1:  # Last attempt
                logger.error(f"Test failed after {MAX_RETRIES} attempts")
                return False
                
            # Wait before retrying for other errors
            time.sleep(RETRY_DELAY)
    
    return False

async def run_tests_sequentially():
    """Run tests one by one with delays in between."""
    results = {}
    
    for test_file in TEST_FILES:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting tests in {test_file}")
        logger.info(f"{'='*80}")
        
        # First, discover all test names in the file
        try:
            discover_cmd = [
                "poetry", "run", "pytest",
                "--collect-only", "-q",
                test_file
            ]
            
            result = subprocess.run(
                discover_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Extract test names
            test_lines = [
                line.strip() 
                for line in result.stdout.split('\n') 
                if '::' in line and 'test_' in line
            ]
            
            if not test_lines:
                logger.warning(f"No tests found in {test_file}")
                results[test_file] = "No tests found"
                continue
                
            # Run each test individually
            for test_line in test_lines:
                test_name = test_line.split('::')[-1]
                logger.info(f"\n{'*'*40}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'*'*40}")
                
                # Run the test
                start_time = time.time()
                success = run_test(test_file, test_name)
                duration = time.time() - start_time
                
                # Store the result
                results[f"{test_file}::{test_name}"] = {
                    "status": "PASSED" if success else "FAILED",
                    "duration": f"{duration:.2f}s"
                }
                
                # Add delay between tests
                if test_line != test_lines[-1]:  # Don't delay after the last test
                    logger.info(f"Waiting {DELAY_BETWEEN_TESTS} seconds before next test...")
                    await asyncio.sleep(DELAY_BETWEEN_TESTS)
                    
        except Exception as e:
            logger.error(f"Error running tests in {test_file}: {str(e)}")
            results[test_file] = f"ERROR: {str(e)}"
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test, result in results.items():
        if isinstance(result, dict):
            logger.info(f"{test}: {result['status']} ({result['duration']})")
        else:
            logger.info(f"{test}: {result}")
    
    logger.info("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(run_tests_sequentially())

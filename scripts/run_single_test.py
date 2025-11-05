"""Script to run a single test with proper rate limiting."""
import os
import sys
import time
import asyncio
import pytest
import importlib.util
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test configuration
TEST_DELAY = 30  # seconds between tests
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds between retries

def get_test_functions(module_path):
    """Get all test functions from a test module."""
    # Load the test module
    spec = importlib.util.spec_from_file_location("test_module", module_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Find all test functions
    test_functions = []
    for name in dir(test_module):
        if name.startswith('test_'):
            func = getattr(test_module, name)
            if callable(func) and hasattr(func, '__code__'):
                test_functions.append((name, func))
    
    return test_functions

async def run_single_test(module_path, test_name):
    """Run a single test with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\n{'='*80}")
            print(f"Running test: {test_name} (Attempt {attempt + 1}/{MAX_RETRIES})")
            print(f"{'='*80}")
            
            # Run the test
            result = pytest.main([
                "-xvs",  # -x: exit on first failure, -v: verbose, -s: show output
                f"{module_path}::{test_name}",
                "--tb=short"  # shorter traceback
            ])
            
            if result == 0:  # Test passed
                return True
                
            # If we got here, the test failed
            if attempt < MAX_RETRIES - 1:
                print(f"\nTest failed, waiting {RETRY_DELAY} seconds before retry...")
                await asyncio.sleep(RETRY_DELAY)
                
        except Exception as e:
            print(f"Error running test: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {RETRY_DELAY} seconds before retry...")
                await asyncio.sleep(RETRY_DELAY)
    
    return False

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_test.py <test_file_path> [test_function_name]")
        sys.exit(1)
    
    test_file = sys.argv[1]
    test_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)
    
    if test_name:
        # Run a specific test
        success = await run_single_test(test_file, test_name)
        print("\n" + "="*80)
        print(f"Test {'PASSED' if success else 'FAILED'}")
        print("="*80)
    else:
        # Run all tests in the file, one by one
        test_functions = get_test_functions(test_file)
        if not test_functions:
            print(f"No test functions found in {test_file}")
            sys.exit(1)
        
        print(f"Found {len(test_functions)} test(s) in {test_file}\n")
        
        results = {}
        for i, (name, _) in enumerate(test_functions, 1):
            print(f"\n{'*'*80}")
            print(f"Running test {i}/{len(test_functions)}: {name}")
            print(f"{'*'*80}")
            
            success = await run_single_test(test_file, name)
            results[name] = "PASSED" if success else "FAILED"
            
            # Add delay between tests, but not after the last one
            if i < len(test_functions):
                print(f"\nWaiting {TEST_DELAY} seconds before next test...")
                await asyncio.sleep(TEST_DELAY)
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for name, result in results.items():
            print(f"{name}: {result}")
        
        # Exit with non-zero code if any test failed
        if "FAILED" in results.values():
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

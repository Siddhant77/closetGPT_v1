"""
Main test runner for ClosetGPT v1 tests
Enables running tests as: python -m tests
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all test suites with detailed reporting"""
    print("=" * 60)
    print("🧪 CLOSETGPT V1 TEST SUITE")
    print("=" * 60)
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    
    # Load all tests from the tests directory
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Use the standard unittest runner
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures) + len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n❌ Failed Tests:")
        for test, error in result.failures + result.errors:
            test_name = str(test).split()[0]
            error_msg = error.split('\n')[-2] if '\n' in error else error
            print(f"  - {test_name}: {error_msg}")
    else:
        print("\n🎉 ALL TESTS PASSED!")
    
    print("=" * 60)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test_class(class_name):
    """Run a specific test class"""
    print(f"🧪 Running {class_name} tests...")
    
    # Import the specific test class
    if class_name == "TestCLIPEmbedder":
        from .test_clip_embedder import TestCLIPEmbedder
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCLIPEmbedder)
    elif class_name == "TestDataManager":
        from .test_data_manager import TestDataManager
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDataManager)
    elif class_name == "TestIntegration":
        from .test_integration import TestIntegration
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    else:
        print(f"❌ Unknown test class: {class_name}")
        return False
    
    # Run the specific test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main entry point for test runner"""
    if len(sys.argv) > 1:
        # Run specific test class
        class_name = sys.argv[1]
        success = run_specific_test_class(class_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
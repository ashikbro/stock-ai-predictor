"""
Test script to verify all modules can be imported correctly.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    print("-" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test config
    try:
        from src import config
        print("✓ src.config")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.config - {e}")
        tests_failed += 1
    
    # Test data modules
    try:
        from src.data import data_fetcher
        print("✓ src.data.data_fetcher")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.data.data_fetcher - {e}")
        tests_failed += 1
    
    try:
        from src.data import preprocessor
        print("✓ src.data.preprocessor")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.data.preprocessor - {e}")
        tests_failed += 1
    
    # Test RL agent modules
    try:
        from src.rl_agent import environment
        print("✓ src.rl_agent.environment")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.rl_agent.environment - {e}")
        tests_failed += 1
    
    try:
        from src.rl_agent import dqn_agent
        print("✓ src.rl_agent.dqn_agent")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.rl_agent.dqn_agent - {e}")
        tests_failed += 1
    
    print("-" * 60)
    print(f"\nResults: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("✓ All module imports successful!")
        return True
    else:
        print("✗ Some module imports failed. Install dependencies with:")
        print("  pip install -r requirements.txt")
        return False


def test_config_values():
    """Test that configuration values are properly loaded."""
    print("\nTesting configuration values...")
    print("-" * 60)
    
    try:
        from src import config
        
        assert hasattr(config, 'STOCK_SYMBOLS'), "STOCK_SYMBOLS not found"
        assert hasattr(config, 'INITIAL_BALANCE'), "INITIAL_BALANCE not found"
        assert hasattr(config, 'LEARNING_RATE'), "LEARNING_RATE not found"
        assert hasattr(config, 'GAMMA'), "GAMMA not found"
        
        print(f"✓ Stock Symbols: {config.STOCK_SYMBOLS}")
        print(f"✓ Initial Balance: ${config.INITIAL_BALANCE:,.2f}")
        print(f"✓ Learning Rate: {config.LEARNING_RATE}")
        print(f"✓ Gamma: {config.GAMMA}")
        print(f"✓ Window Size: {config.WINDOW_SIZE}")
        
        print("-" * 60)
        print("✓ All configuration values properly loaded!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        print("-" * 60)
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    print("-" * 60)
    
    required_dirs = ['src', 'src/data', 'src/rl_agent', 'models', 'data']
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - NOT FOUND")
            all_exist = False
    
    print("-" * 60)
    if all_exist:
        print("✓ All required directories exist!")
        return True
    else:
        print("✗ Some directories missing. Run setup.sh to create them.")
        return False


def test_files_exist():
    """Test that required files exist."""
    print("\nTesting required files...")
    print("-" * 60)
    
    required_files = [
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.env.example',
        'app.py',
        'train.py',
        'fetch_data.py',
        'src/config.py',
        'src/data/data_fetcher.py',
        'src/data/preprocessor.py',
        'src/rl_agent/environment.py',
        'src/rl_agent/dqn_agent.py'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False
    
    print("-" * 60)
    if all_exist:
        print("✓ All required files exist!")
        return True
    else:
        print("✗ Some files missing!")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("STOCK AI PREDICTOR - VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Required Files", test_files_exist()))
    results.append(("Module Imports", test_imports()))
    results.append(("Configuration", test_config_values()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThe Stock AI Predictor is properly configured.")
        print("\nYou can now:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Fetch data: python fetch_data.py")
        print("3. Train a model: python train.py --symbol AAPL --episodes 100")
        print("4. Launch dashboard: streamlit run app.py")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review the errors above and fix any issues.")
    
    print()
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

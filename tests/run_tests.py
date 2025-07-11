"""
Test runner script for AutoCare backend unit tests.

This script provides convenient commands to run different categories of tests
and generate coverage reports.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main test runner function."""
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Available test commands
    test_commands = {
        "all": {
            "cmd": ["python", "-m", "pytest", "tests/unit/", "-v"],
            "description": "Run all unit tests"
        },
        "models": {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_models_*.py", "-v"],
            "description": "Run model/schema tests"
        },
        "services": {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_services_*.py", "-v"],
            "description": "Run service layer tests"
        },
        "utils": {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_utils_*.py", "-v"],
            "description": "Run utility function tests"
        },
        "crud": {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_crud_*.py", "-v"],
            "description": "Run CRUD operation tests"
        },
        "controllers": {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_main_*.py", "-v"],
            "description": "Run controller/API tests"
        },
        "coverage": {
            "cmd": ["python", "-m", "pytest", "tests/unit/", "--cov=app", "--cov-report=html", "--cov-report=term-missing"],
            "description": "Run all tests with coverage report"
        },
        "fast": {
            "cmd": ["python", "-m", "pytest", "tests/unit/", "-x", "--tb=short"],
            "description": "Run tests until first failure (fast feedback)"
        }
    }
    
    if len(sys.argv) < 2:
        print("AutoCare Backend Unit Test Runner")
        print("=" * 40)
        print("Usage: python run_tests.py <command>")
        print("\nAvailable commands:")
        for cmd, info in test_commands.items():
            print(f"  {cmd:12} - {info['description']}")
        print("\nExamples:")
        print("  python run_tests.py all        # Run all unit tests")
        print("  python run_tests.py models     # Run only model tests")
        print("  python run_tests.py coverage   # Run with coverage report")
        print("  python run_tests.py fast       # Fast feedback mode")
        return
    
    command = sys.argv[1].lower()
    
    if command not in test_commands:
        print(f"Error: Unknown command '{command}'")
        print(f"Available commands: {', '.join(test_commands.keys())}")
        return
    
    # Run the selected test command
    cmd_info = test_commands[command]
    success = run_command(cmd_info["cmd"], cmd_info["description"])
    
    if success:
        print(f"\n✅ {cmd_info['description']} completed successfully!")
        
        # Additional information for coverage command
        if command == "coverage":
            print("\n📊 Coverage report generated:")
            print("   - HTML report: htmlcov/index.html")
            print("   - Open in browser to view detailed coverage")
    else:
        print(f"\n❌ {cmd_info['description']} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

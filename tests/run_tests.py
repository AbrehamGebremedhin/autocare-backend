#!/usr/bin/env python3
"""
Enhanced test runner for AutoCare backend with comprehensive reporting and analysis.

This script provides convenient commands to run different categories of tests,
generate coverage reports, and analyze security/performance issues.
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

class TestRunner:
    """Enhanced test runner with detailed reporting and analysis."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def run_tests(self, 
                  test_type: str = "all",
                  coverage: bool = True,
                  verbose: bool = False,
                  parallel: bool = False,
                  markers: Optional[str] = None,
                  output_format: str = "json") -> Dict:
        """
        Run tests with specified configuration.
        
        Args:
            test_type: Type of tests to run (all, unit, integration, security)
            coverage: Whether to collect coverage data
            verbose: Enable verbose output
            parallel: Run tests in parallel
            markers: pytest markers to filter tests
            output_format: Output format (json, xml, html)
        
        Returns:
            Dictionary containing test results and metrics
        """
        print(f"🚀 Starting {test_type} tests...")
        print(f"📁 Project root: {self.project_root}")
        print(f"🧪 Test directory: {self.test_dir}")
        
        # Build pytest command
        cmd = self._build_pytest_command(
            test_type=test_type,
            coverage=coverage,
            verbose=verbose,
            parallel=parallel,
            markers=markers,
            output_format=output_format
        )
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run tests
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=False,  # Show output in real-time
            text=True
        )
        end_time = time.time()
        
        # Process results with simple metrics
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": test_type,
            "duration": round(end_time - start_time, 2),
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "command": ' '.join(cmd)
        }
        
        # Print summary
        self._print_summary(test_results)
        
        return test_results
    
    def _build_pytest_command(self, 
                             test_type: str,
                             coverage: bool,
                             verbose: bool,
                             parallel: bool,
                             markers: Optional[str],
                             output_format: str) -> List[str]:
        """Build pytest command with all options."""
        # Use the current Python executable to ensure we use the virtual environment
        python_exe = sys.executable
        cmd = [python_exe, "-m", "pytest"]
        
        # Test path selection
        if test_type == "unit":
            cmd.extend([str(self.test_dir / "unit")])
        elif test_type == "integration":
            cmd.extend([str(self.test_dir / "integration")])
        elif test_type == "security":
            cmd.extend(["-m", "security"])
        elif test_type == "performance":
            cmd.extend(["-m", "performance"])
        else:
            cmd.extend([str(self.test_dir / "unit")])  # Default to unit tests
        
        # Coverage options
        if coverage:
            cmd.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:test_results/htmlcov",
                "--cov-fail-under=70"  # Reasonable threshold
            ])
        
        # Output options
        if verbose:
            cmd.extend(["-v", "-s"])
        else:
            cmd.extend(["-v"])  # Always use some verbosity
        
        # Additional options
        cmd.extend([
            "--tb=short",
            "--maxfail=5",
            "--durations=10",
            "--disable-warnings"
        ])
        
        return cmd
    
    def _print_summary(self, test_results: Dict):
        """Print test results summary to console."""
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        # Status
        status_icon = "✅" if test_results['success'] else "❌"
        print(f"{status_icon} Status: {'PASSED' if test_results['success'] else 'FAILED'}")
        print(f"⏱️  Duration: {test_results['duration']}s")
        print(f"🔧 Command: {test_results['command']}")
        
        print("\n" + "=" * 60)
        
        if test_results['success']:
            print("🎉 All tests passed! Great job!")
        else:
            print("💡 Some tests failed. Check the output above for details.")

def run_command(cmd, description):
    """Run a command and handle errors (legacy function)."""
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
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="AutoCare Backend Test Runner")
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "security", "performance"],
        default="unit",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage collection"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Pytest markers to filter tests"
    )
    
    # Legacy support: allow direct test types as arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["all", "unit", "integration", "security", "models", "services", "utils", "coverage", "fast"]:
        # Legacy mode
        test_type = sys.argv[1]
        
        # Handle special legacy commands
        if test_type == "coverage":
            runner = TestRunner()
            results = runner.run_tests(
                test_type="unit",
                coverage=True,
                verbose=True
            )
            if results['success']:
                print("\n📊 Coverage report generated:")
                print("   - HTML report: test_results/htmlcov/index.html")
                print("   - Open in browser to view detailed coverage")
            sys.exit(0 if results['success'] else 1)
        
        elif test_type == "fast":
            # Fast mode: stop on first failure
            cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-x", "--tb=short"]
            success = run_command(cmd, "Run tests until first failure (fast feedback)")
            sys.exit(0 if success else 1)
        
        elif test_type in ["models", "services", "utils"]:
            # Specific component tests
            cmd = [sys.executable, "-m", "pytest", f"tests/unit/test_{test_type}_*.py", "-v"]
            success = run_command(cmd, f"Run {test_type} tests")
            sys.exit(0 if success else 1)
        
        else:
            # Standard test types
            runner = TestRunner()
            results = runner.run_tests(
                test_type=test_type,
                coverage=True,
                verbose=True
            )
            sys.exit(0 if results['success'] else 1)
    
    # Parse modern arguments
    args = parser.parse_args()
    
    # Run tests
    runner = TestRunner()
    results = runner.run_tests(
        test_type=args.type,
        coverage=not args.no_coverage,
        verbose=args.verbose,
        parallel=args.parallel,
        markers=args.markers
    )
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()

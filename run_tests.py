#!/usr/bin/env python3
"""
Test runner script for Simtune project.

This script provides convenient commands for running different types of tests
and generating coverage reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🚀 {description}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def run_unit_tests():
    """Run only unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit", "-v"]
    return run_command(cmd, "Running unit tests")


def run_integration_tests():
    """Run only integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration", "-v"]
    return run_command(cmd, "Running integration tests")


def run_all_tests():
    """Run all tests with coverage."""
    cmd = ["python", "-m", "pytest"]
    return run_command(cmd, "Running all tests with coverage")


def run_fast_tests():
    """Run tests excluding slow ones."""
    cmd = ["python", "-m", "pytest", "-m", "not slow"]
    return run_command(cmd, "Running fast tests (excluding slow tests)")


def run_coverage_report():
    """Generate detailed coverage report."""
    cmd = ["python", "-m", "pytest", "--cov-report=html", "--cov-report=term"]
    success = run_command(cmd, "Generating coverage report")

    if success:
        html_report = Path("htmlcov/index.html")
        if html_report.exists():
            print(f"📊 HTML coverage report generated: {html_report.absolute()}")
        else:
            print("⚠️  HTML coverage report not found")

    return success


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path, "-v"]
    return run_command(cmd, f"Running specific test: {test_path}")


def check_dependencies():
    """Check if all required test dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-typer",
        "responses",
        "freezegun",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required test dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ All test dependencies are installed")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Simtune Test Runner")
    parser.add_argument(
        "command",
        choices=["unit", "integration", "all", "fast", "coverage", "deps", "specific"],
        help="Test command to run",
    )
    parser.add_argument(
        "--test-path", help="Specific test path (for 'specific' command)"
    )

    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    success = True

    if args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "fast":
        success = run_fast_tests()
    elif args.command == "coverage":
        success = run_coverage_report()
    elif args.command == "deps":
        success = check_dependencies()
    elif args.command == "specific":
        if not args.test_path:
            print("❌ --test-path is required for 'specific' command")
            sys.exit(1)
        success = run_specific_test(args.test_path)

    if not success:
        sys.exit(1)

    print("\n🎉 Test execution completed successfully!")


if __name__ == "__main__":
    main()

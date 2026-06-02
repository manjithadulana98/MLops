"""
Quick setup and test script for VT-Detection MLOps Pipeline

This script validates the installation and demonstrates the workflow.

Usage:
    python setup_verify.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd: str, description: str = "") -> bool:
    """Execute a shell command and return success status."""
    if description:
        print(f"\n{'='*70}")
        print(f"▶ {description}")
        print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\n" * 2)
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "VT-Detection MLOps Pipeline — Setup Verification".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    checks = []
    
    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(("Python version", f"{py_version}" if int(py_version[0]) >= 3 else "FAIL"))
    
    # 2. Project structure
    required_dirs = [
        "src/ingestion", "src/training", "src/evaluation",
        "tests", "configs", "models/artifacts", "data/MIT_dataset"
    ]
    for d in required_dirs:
        checks.append((f"Directory: {d}", "✓" if Path(d).exists() else "✗"))
    
    # 3. Required files
    required_files = [
        "requirements.txt", "dvc.yaml", "configs/training_config.yaml",
        "src/ingestion/load_mit_dataset.py", "src/training/model.py",
        "src/training/train.py", "src/evaluation/evaluate.py",
    ]
    for f in required_files:
        checks.append((f"File: {f}", "✓" if Path(f).exists() else "✗"))
    
    # Print checklist
    print("\n📋 VALIDATION CHECKLIST")
    print("-" * 70)
    for item, status in checks:
        symbol = "✓" if status == "✓" else ("✗" if status == "✗" else "ℹ")
        print(f"  {symbol} {item:.<60} {status}")
    
    # 4. Package imports
    print("\n📦 PACKAGE IMPORTS")
    print("-" * 70)
    packages = ["numpy", "pandas", "tensorflow", "sklearn", "dvc", "mlflow", "wfdb", "yaml"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — RUN: pip install -r requirements.txt")
    
    # 5. Run unit tests
    print("\n🧪 RUNNING UNIT TESTS")
    print("-" * 70)
    success = run_command(
        "pytest tests/test_validate.py -v --tb=short",
        "Unit Tests: Data Validation (TC-001 to TC-009)"
    )
    print("  ✓ All validation tests passed" if success else "  ✗ Some tests failed")
    
    # 6. Dataset check
    print("\n📊 DATASET CHECK")
    print("-" * 70)
    mit_path = Path("data/MIT_dataset")
    if mit_path.exists():
        record_files = list(mit_path.glob("*.hea"))
        print(f"  ✓ MIT-BIH dataset found: {len(record_files)} records")
    else:
        print("  ✗ MIT-BIH dataset not found at data/MIT_dataset")
    
    # 7. Quick integration test
    print("\n🚀 QUICK INTEGRATION TEST (5% data, 2 epochs)")
    print("-" * 70)
    success = run_command(
        "pytest tests/test_mlops_integration.py -v --tb=short -k 'not MLOps' --maxfail=1",
        "Integration Tests: Workflow Components"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ SETUP VERIFICATION COMPLETE".center(70))
    print("=" * 70)
    print("\n📖 NEXT STEPS:\n")
    print("  1. View the comprehensive MLOps guide:")
    print("     open docs/MLOPS_WORKFLOW.md\n")
    print("  2. Run the full training pipeline:")
    print("     dvc repro\n")
    print("  3. Monitor experiments in MLflow:")
    print("     mlflow ui\n")
    print("  4. Evaluate the trained model:")
    print("     python -m src.evaluation.pipeline --model models/artifacts/vgg16_best.keras\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

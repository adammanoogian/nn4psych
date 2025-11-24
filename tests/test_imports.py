#!/usr/bin/env python3
"""
Quick test script to verify all imports work after reorganization
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

print("=" * 60)
print("TESTING NN4PSYCH IMPORT STRUCTURE")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Project root: {PROJECT_ROOT}")
print()

# Track results
results = []

def test_import(module_path, description):
    """Test a single import"""
    try:
        if "from" in module_path:
            exec(module_path)
        else:
            __import__(module_path)
        print(f"✓ {description}: SUCCESS")
        results.append(True)
        return True
    except ImportError as e:
        print(f"✗ {description}: FAILED - {e}")
        results.append(False)
        return False
    except Exception as e:
        print(f"✗ {description}: ERROR - {e}")
        results.append(False)
        return False

# Test core package imports
print("\n1. Core Package Imports")
print("-" * 40)
test_import("nn4psych", "Main package")
test_import("from nn4psych import __version__", "Package version")

# Test model imports
print("\n2. Model Imports")
print("-" * 40)
test_import("from nn4psych.models import ActorCritic", "ActorCritic model")
test_import("from nn4psych.models.actor_critic import ActorCritic", "Direct model import")

# Test environment imports
print("\n3. Environment Imports")
print("-" * 40)
test_import("from envs import PIE_CP_OB_v2", "Environment module")
test_import("from envs.pie_environment import PIE_CP_OB_v2", "Direct environment import")

# Test training imports
print("\n4. Training Module Imports")
print("-" * 40)
test_import("from nn4psych.training.configs import ExperimentConfig", "ExperimentConfig")
test_import("from nn4psych.training.configs import ModelConfig", "ModelConfig")
test_import("from nn4psych.training.configs import TaskConfig", "TaskConfig")
test_import("from nn4psych.training.configs import TrainingConfig", "TrainingConfig")

# Test analysis imports
print("\n5. Analysis Module Imports")
print("-" * 40)
test_import("from nn4psych.analysis.behavior import extract_behavior", "Behavior extraction")
test_import("from nn4psych.analysis.hyperparams import HyperparamAnalyzer", "Hyperparameter analyzer")

# Test utility imports
print("\n6. Utility Module Imports")
print("-" * 40)
test_import("from nn4psych.utils.io import save_model", "Save model utility")
test_import("from nn4psych.utils.io import load_model", "Load model utility")
test_import("from nn4psych.utils.metrics import get_lrs", "Learning rate metric")
test_import("from nn4psych.utils.metrics import get_lrs_v2", "Learning rate v2 metric")
test_import("from nn4psych.utils.plotting import plot_behavior", "Plotting utility")

# Test config import
print("\n7. Configuration Import")
print("-" * 40)
test_import("import config", "Main config module")

# Test Bayesian model imports (from scripts)
print("\n8. Bayesian Model Imports (from scripts)")
print("-" * 40)
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
test_import("from bayesian.bayesian_models import BayesianModel", "Bayesian model")
test_import("from bayesian.pyem_models import PyEMModel", "PyEM model")

# Summary
print("\n" + "=" * 60)
print("IMPORT TEST SUMMARY")
print("=" * 60)
success_count = sum(results)
total_count = len(results)
success_rate = (success_count / total_count * 100) if total_count > 0 else 0

print(f"Successful imports: {success_count}/{total_count} ({success_rate:.1f}%)")

if success_count == total_count:
    print("\n✅ All imports successful! The reorganization is working correctly.")
    sys.exit(0)
else:
    print(f"\n⚠️ {total_count - success_count} imports failed. Please check the errors above.")
    sys.exit(1)
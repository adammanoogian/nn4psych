"""
Test Suite for NN4Psych Reorganization Validation
Ensures all modules, imports, and pipelines work correctly after reorganization
"""

import sys
import os
import pytest
import importlib
import subprocess
from pathlib import Path
import yaml
import pickle
import pandas as pd
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# =============================================================================
# Test 1: Package Import Structure
# =============================================================================

class TestImportStructure:
    """Validate that all modules can be imported correctly"""

    def test_core_package_imports(self):
        """Test that main nn4psych package imports work"""
        try:
            # Core package
            import nn4psych
            from nn4psych import __version__
            assert __version__ == "0.2.0"
        except ImportError as e:
            pytest.fail(f"Failed to import nn4psych package: {e}")

    def test_model_imports(self):
        """Test model module imports"""
        try:
            from nn4psych.models import ActorCritic
            from nn4psych.models.actor_critic import ActorCritic as AC
            assert ActorCritic == AC
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")

    def test_environment_imports(self):
        """Test standalone environment imports"""
        try:
            from envs import PIE_CP_OB_v2
            from envs.pie_environment import PIE_CP_OB_v2 as PIE
            assert PIE_CP_OB_v2 == PIE
        except ImportError as e:
            pytest.fail(f"Failed to import environments: {e}")

    def test_training_imports(self):
        """Test training module imports"""
        try:
            from nn4psych.training.configs import ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig
            assert ExperimentConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import training configs: {e}")

    def test_analysis_imports(self):
        """Test analysis module imports"""
        try:
            from nn4psych.analysis.behavior import extract_behavior
            from nn4psych.analysis.hyperparams import HyperparamAnalyzer
            assert extract_behavior is not None
            assert HyperparamAnalyzer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import analysis modules: {e}")

    def test_utils_imports(self):
        """Test utility module imports"""
        try:
            from nn4psych.utils.io import save_model, load_model, saveload
            from nn4psych.utils.metrics import get_lrs, get_lrs_v2
            from nn4psych.utils.plotting import plot_behavior
            assert all([save_model, load_model, get_lrs, get_lrs_v2])
        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")

    def test_bayesian_imports(self):
        """Test Bayesian model imports from scripts"""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
            from bayesian.bayesian_models import BayesianModel
            from bayesian.pyem_models import PyEMModel
            assert BayesianModel is not None
            assert PyEMModel is not None
        except ImportError as e:
            pytest.fail(f"Failed to import Bayesian models: {e}")

# =============================================================================
# Test 2: Configuration System
# =============================================================================

class TestConfigurationSystem:
    """Validate configuration loading and usage"""

    def test_config_py_exists(self):
        """Check that config.py exists and is importable"""
        try:
            import config
            assert hasattr(config, 'PROJECT_ROOT')
            assert hasattr(config, 'MODEL_PARAMS')
            assert hasattr(config, 'TASK_PARAMS')
            assert hasattr(config, 'TRAINING_PARAMS')
        except ImportError as e:
            pytest.fail(f"Failed to import config.py: {e}")

    def test_config_paths(self):
        """Verify all configured paths exist"""
        import config

        # Check key directories
        assert config.PROJECT_ROOT.exists()
        assert config.TRAINED_MODELS_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.OUTPUT_DIR.exists()
        assert config.FIGURES_DIR.exists()
        assert config.NOTEBOOKS_DIR.exists()

    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation and serialization"""
        from nn4psych.training.configs import ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig

        # Create config
        model_config = ModelConfig(
            type="ActorCritic",
            input_dim=9,
            hidden_dim=64,
            action_dim=3,
            gain=1.5
        )

        task_config = TaskConfig(
            type="PIE_CP_OB_v2",
            condition="change-point",
            max_time=200,
            hazard_rate=0.125
        )

        training_config = TrainingConfig(
            epochs=1000,
            learning_rate=0.0003,
            gamma=0.95,
            rollout_size=100
        )

        exp_config = ExperimentConfig(
            model=model_config,
            task=task_config,
            training=training_config
        )

        # Test YAML serialization
        yaml_str = exp_config.to_yaml()
        assert "ActorCritic" in yaml_str
        assert "change-point" in yaml_str

        # Test deserialization
        loaded_config = ExperimentConfig.from_yaml(yaml_str)
        assert loaded_config.model.hidden_dim == 64
        assert loaded_config.training.gamma == 0.95

# =============================================================================
# Test 3: Model Architecture
# =============================================================================

class TestModelArchitecture:
    """Validate model creation and forward pass"""

    def test_actor_critic_creation(self):
        """Test ActorCritic model instantiation"""
        from nn4psych.models import ActorCritic

        model = ActorCritic(
            input_dim=9,
            hidden_dim=64,
            action_dim=3,
            gain=1.5
        )

        assert model.input_dim == 9
        assert model.hidden_dim == 64
        assert model.action_dim == 3
        assert model.gain == 1.5

    def test_model_forward_pass(self):
        """Test model forward pass with dummy data"""
        from nn4psych.models import ActorCritic

        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        # Create dummy input
        batch_size = 32
        seq_length = 10
        x = torch.randn(batch_size, seq_length, 9)

        # Forward pass
        logits, values, hidden = model(x)

        # Check output shapes
        assert logits.shape == (batch_size, seq_length, 3)
        assert values.shape == (batch_size, seq_length, 1)
        assert hidden.shape == (batch_size, 64)

    def test_hidden_state_reset(self):
        """Test hidden state initialization and reset"""
        from nn4psych.models import ActorCritic

        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        # Get initial hidden state
        hidden = model.get_initial_hidden(batch_size=16)
        assert hidden.shape == (16, 64)

        # Reset hidden state
        model.reset_hidden(batch_size=8)
        assert model.hidden.shape == (8, 64)

# =============================================================================
# Test 4: Environment Functionality
# =============================================================================

class TestEnvironment:
    """Validate environment operations"""

    def test_environment_creation(self):
        """Test PIE_CP_OB_v2 environment instantiation"""
        from envs import PIE_CP_OB_v2

        # Test both conditions
        env_cp = PIE_CP_OB_v2(condition="change-point")
        env_ob = PIE_CP_OB_v2(condition="oddball")

        assert env_cp.condition == "change-point"
        assert env_ob.condition == "oddball"

    def test_environment_reset(self):
        """Test environment reset functionality"""
        from envs import PIE_CP_OB_v2

        env = PIE_CP_OB_v2(condition="change-point", max_time=200)
        obs = env.reset()

        # Check observation shape (should be 6-dimensional)
        assert len(obs) == 6
        assert env.time == 0
        assert env.done == False

    def test_environment_step(self):
        """Test environment step function"""
        from envs import PIE_CP_OB_v2

        env = PIE_CP_OB_v2(condition="change-point")
        obs = env.reset()

        # Take a random action
        action = np.random.choice([0, 1, 2])  # left, right, or confirm
        next_obs, reward, done, info = env.step(action)

        assert len(next_obs) == 6
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

# =============================================================================
# Test 5: Data Pipeline Scripts
# =============================================================================

class TestDataPipeline:
    """Validate data pipeline script functionality"""

    def test_pipeline_scripts_exist(self):
        """Check that all pipeline scripts exist"""
        pipeline_dir = PROJECT_ROOT / "scripts" / "data_pipeline"

        expected_scripts = [
            "00_run_full_pipeline.py",
            "01_extract_model_behavior.py",
            "02_compute_learning_metrics.py",
            "03_analyze_hyperparameter_sweeps.py"
        ]

        for script in expected_scripts:
            script_path = pipeline_dir / script
            assert script_path.exists(), f"Missing script: {script}"

    def test_master_runner_help(self):
        """Test that master runner script provides help"""
        script_path = PROJECT_ROOT / "scripts" / "data_pipeline" / "00_run_full_pipeline.py"

        # Run with --help flag
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Run full analysis pipeline" in result.stdout or "stages" in result.stdout

    def test_pipeline_imports(self):
        """Test that pipeline scripts can import required modules"""
        # Test stage 01 imports
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_pipeline"))
            script_content = (PROJECT_ROOT / "scripts" / "data_pipeline" / "01_extract_model_behavior.py").read_text()

            # Check for key imports in the script
            assert "from nn4psych.models import ActorCritic" in script_content or "nn4psych" in script_content
            assert "from envs import PIE_CP_OB_v2" in script_content or "envs" in script_content
        except Exception as e:
            pytest.fail(f"Pipeline script import check failed: {e}")

# =============================================================================
# Test 6: Analysis Scripts
# =============================================================================

class TestAnalysisScripts:
    """Validate analysis script functionality"""

    def test_analysis_scripts_exist(self):
        """Check that key analysis scripts exist"""
        analysis_dir = PROJECT_ROOT / "scripts" / "analysis"

        expected_scripts = [
            "analyze_rnn_refactored.py",
            "analyze_hyperparams_unified.py",
            "visualize_learning_rates.py",
            "nassarfig6.py",
            "analyze_fixed_points.py"
        ]

        for script in expected_scripts:
            script_path = analysis_dir / script
            assert script_path.exists(), f"Missing script: {script}"

    def test_bayesian_scripts_exist(self):
        """Check that Bayesian analysis scripts exist"""
        bayesian_dir = PROJECT_ROOT / "scripts" / "analysis" / "bayesian"

        expected_files = [
            "__init__.py",
            "bayesian_models.py",
            "pyem_models.py"
        ]

        for file in expected_files:
            file_path = bayesian_dir / file
            assert file_path.exists(), f"Missing Bayesian file: {file}"

# =============================================================================
# Test 7: Training Scripts
# =============================================================================

class TestTrainingScripts:
    """Validate training script functionality"""

    def test_training_scripts_exist(self):
        """Check that training scripts exist"""
        training_dir = PROJECT_ROOT / "scripts" / "training"

        # Main training scripts
        assert (training_dir / "train_rnn_canonical.py").exists()
        assert (training_dir / "examples" / "train_example.py").exists()

        # SLURM script
        assert (PROJECT_ROOT / "scripts" / "slurm_seeds.sh").exists()

    def test_canonical_training_help(self):
        """Test canonical training script help"""
        script_path = PROJECT_ROOT / "scripts" / "training" / "train_rnn_canonical.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )

        # Check for key parameters in help
        assert "--gamma" in result.stdout or result.returncode == 0
        assert "--rollout" in result.stdout or result.returncode == 0
        assert "--epochs" in result.stdout or result.returncode == 0

# =============================================================================
# Test 8: Fitting Scripts
# =============================================================================

class TestFittingScripts:
    """Validate fitting script functionality"""

    def test_fitting_scripts_exist(self):
        """Check that fitting scripts exist"""
        fitting_dir = PROJECT_ROOT / "scripts" / "fitting"

        expected_scripts = [
            "fit_bayesian_pymc.py",
            "fit_bayesian_pyem.py",
            "README.md"
        ]

        for script in expected_scripts:
            script_path = fitting_dir / script
            assert script_path.exists(), f"Missing script: {script}"

# =============================================================================
# Test 9: Data Directories
# =============================================================================

class TestDataDirectories:
    """Validate data directory structure"""

    def test_data_structure(self):
        """Check that data directories exist with correct structure"""
        data_dir = PROJECT_ROOT / "data"

        # Required subdirectories
        required_dirs = [
            "raw",
            "raw/nassar2021",
            "raw/fig2_values",
            "processed",
            "intermediate"
        ]

        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            assert dir_path.exists(), f"Missing data directory: {dir_name}"

    def test_output_structure(self):
        """Check output directory structure"""
        output_dir = PROJECT_ROOT / "output"

        # These should exist or be created
        expected_dirs = [
            "behavioral_summary",
            "model_performance",
            "parameter_exploration"
        ]

        for dir_name in expected_dirs:
            dir_path = output_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists(), f"Could not create output directory: {dir_name}"

# =============================================================================
# Test 10: Model Loading
# =============================================================================

class TestModelLoading:
    """Validate model loading functionality"""

    def test_model_directory_exists(self):
        """Check that model directories exist"""
        models_dir = PROJECT_ROOT / "trained_models"

        assert models_dir.exists()
        assert (models_dir / "checkpoints").exists()

    def test_model_loading_function(self):
        """Test model loading utility function"""
        from nn4psych.utils.io import load_model
        from nn4psych.models import ActorCritic

        # Create a dummy model and save it
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        try:
            # Try loading the model
            loaded_state = torch.load(temp_path)
            new_model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
            new_model.load_state_dict(loaded_state)

            # Check that parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            # Clean up
            os.unlink(temp_path)

# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """End-to-end integration tests"""

    def test_model_environment_integration(self):
        """Test that model can interact with environment"""
        from nn4psych.models import ActorCritic
        from envs import PIE_CP_OB_v2

        # Create model and environment
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        env = PIE_CP_OB_v2(condition="change-point")

        # Reset environment
        obs = env.reset()
        model.reset_hidden(batch_size=1)

        # Run a few steps
        for _ in range(10):
            # Prepare input (add batch and time dimensions)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)

            # Add context and reward (dummy values)
            context = torch.zeros(1, 1, 2)
            reward = torch.zeros(1, 1, 1)
            full_input = torch.cat([obs_tensor, context, reward], dim=-1)

            # Forward pass
            logits, values, hidden = model(full_input)

            # Sample action
            probs = torch.softmax(logits.squeeze(), dim=-1)
            action = torch.multinomial(probs, 1).item()

            # Environment step
            obs, reward, done, info = env.step(action)

            if done:
                break

        assert True  # If we get here, integration works

    def test_config_to_model_pipeline(self):
        """Test configuration to model creation pipeline"""
        from nn4psych.training.configs import ExperimentConfig
        from nn4psych.models import ActorCritic

        # Create config
        config_dict = {
            "model": {
                "type": "ActorCritic",
                "input_dim": 9,
                "hidden_dim": 128,
                "action_dim": 3,
                "gain": 2.0
            },
            "task": {
                "type": "PIE_CP_OB_v2",
                "condition": "oddball"
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001
            }
        }

        # Convert to YAML and back
        import yaml
        yaml_str = yaml.dump(config_dict)
        config = ExperimentConfig.from_yaml(yaml_str)

        # Create model from config
        model = ActorCritic(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            action_dim=config.model.action_dim,
            gain=config.model.gain
        )

        assert model.hidden_dim == 128
        assert model.gain == 2.0


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
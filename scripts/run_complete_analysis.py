#!/usr/bin/env python3
"""
Complete Analysis Runner for NN4Psych Project
Comprehensive script to run all available analysis pipelines and workflows
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json
import datetime
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_PIPELINE_DIR = SCRIPTS_DIR / "data_pipeline"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis"
TRAINING_DIR = SCRIPTS_DIR / "training"
FITTING_DIR = SCRIPTS_DIR / "fitting"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'analysis_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Pipeline Definitions
# =============================================================================

ANALYSIS_PIPELINES = {
    'data': {
        'name': 'Data Processing Pipeline',
        'description': 'Extract behavior, compute metrics, analyze hyperparameters',
        'stages': [
            {
                'name': 'Behavior Extraction',
                'script': str(DATA_PIPELINE_DIR / '01_extract_model_behavior.py'),
                'args': [],
                'description': 'Extract behavioral data from trained models'
            },
            {
                'name': 'Metrics Computation',
                'script': str(DATA_PIPELINE_DIR / '02_compute_learning_metrics.py'),
                'args': [],
                'description': 'Compute learning rates and performance metrics'
            },
            {
                'name': 'Hyperparameter Analysis',
                'script': str(DATA_PIPELINE_DIR / '03_analyze_hyperparameter_sweeps.py'),
                'args': [],
                'description': 'Analyze hyperparameter sweep results'
            }
        ]
    },

    'training': {
        'name': 'Model Training Pipeline',
        'description': 'Train RNN models with various configurations',
        'stages': [
            {
                'name': 'Single Model Training',
                'script': str(TRAINING_DIR / 'train_rnn_canonical.py'),
                'args': ['--gamma', '0.95', '--rollout', '100', '--epochs', '1000', '--condition', 'change-point'],
                'description': 'Train single RNN model with specified parameters'
            }
        ]
    },

    'visualization': {
        'name': 'Visualization Pipeline',
        'description': 'Generate all analysis plots and figures',
        'stages': [
            {
                'name': 'Learning Rate Visualizations',
                'script': str(ANALYSIS_DIR / 'visualize_learning_rates.py'),
                'args': [],
                'description': 'Create learning rate analysis plots'
            },
            {
                'name': 'Nassar Figure Reproduction',
                'script': str(ANALYSIS_DIR / 'nassarfig6.py'),
                'args': [],
                'description': 'Reproduce Nassar et al. 2021 figures'
            }
        ]
    },

    'bayesian': {
        'name': 'Bayesian Fitting Pipeline',
        'description': 'Fit Bayesian models to behavioral data',
        'stages': [
            {
                'name': 'PyEM Framework Fitting',
                'script': str(FITTING_DIR / 'fit_bayesian_pyem.py'),
                'args': ['--n_iter', '1000'],
                'description': 'Fit models using PyEM framework'
            },
            {
                'name': 'PyMC Bayesian Fitting',
                'script': str(FITTING_DIR / 'fit_bayesian_pymc.py'),
                'args': ['--method', 'mle'],
                'description': 'Fit models using PyMC (MLE method)'
            }
        ]
    },

    'hyperparams': {
        'name': 'Hyperparameter Analysis',
        'description': 'Analyze effects of individual hyperparameters',
        'stages': [
            {
                'name': 'Gamma Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_hyperparams_unified.py'),
                'args': ['--param', 'gamma'],
                'description': 'Analyze discount factor effects'
            },
            {
                'name': 'Rollout Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_hyperparams_unified.py'),
                'args': ['--param', 'rollout'],
                'description': 'Analyze rollout size effects'
            },
            {
                'name': 'Preset Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_hyperparams_unified.py'),
                'args': ['--param', 'preset'],
                'description': 'Analyze memory preset effects'
            },
            {
                'name': 'Scale Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_hyperparams_unified.py'),
                'args': ['--param', 'scale'],
                'description': 'Analyze TD scale effects'
            }
        ]
    },

    'advanced': {
        'name': 'Advanced Neural Analysis',
        'description': 'Fixed point analysis and neural dynamics',
        'stages': [
            {
                'name': 'Fixed Point Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_fixed_points.py'),
                'args': [],
                'description': 'Analyze RNN fixed points and attractors'
            },
            {
                'name': 'RNN Analysis',
                'script': str(ANALYSIS_DIR / 'analyze_rnn_refactored.py'),
                'args': [],
                'description': 'Complete RNN behavioral analysis'
            }
        ]
    },

    'validation': {
        'name': 'Validation Pipeline',
        'description': 'Run validation tests for reorganization',
        'stages': [
            {
                'name': 'Import Tests',
                'script': 'pytest',
                'args': [str(PROJECT_ROOT / 'validation' / 'test_reorganization.py'), '-v', '-k', 'TestImportStructure'],
                'description': 'Test package import structure'
            },
            {
                'name': 'Configuration Tests',
                'script': 'pytest',
                'args': [str(PROJECT_ROOT / 'validation' / 'test_reorganization.py'), '-v', '-k', 'TestConfigurationSystem'],
                'description': 'Test configuration system'
            },
            {
                'name': 'Integration Tests',
                'script': 'pytest',
                'args': [str(PROJECT_ROOT / 'validation' / 'test_reorganization.py'), '-v', '-k', 'TestIntegration'],
                'description': 'Test end-to-end integration'
            },
            {
                'name': 'Full Test Suite',
                'script': 'pytest',
                'args': [str(PROJECT_ROOT / 'validation' / 'test_reorganization.py'), '-v'],
                'description': 'Run complete validation suite'
            }
        ]
    }
}

# Quick presets for common workflows
WORKFLOW_PRESETS = {
    'quick': ['data', 'visualization'],
    'full': ['data', 'visualization', 'hyperparams'],
    'training_analysis': ['training', 'data', 'visualization'],
    'bayesian_full': ['data', 'bayesian', 'visualization'],
    'validate': ['validation'],
    'complete': ['validation', 'data', 'visualization', 'hyperparams', 'bayesian', 'advanced']
}

# =============================================================================
# Pipeline Runner Functions
# =============================================================================

class PipelineRunner:
    """Manages execution of analysis pipelines"""

    def __init__(self, dry_run: bool = False, continue_on_error: bool = False):
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.results = {}
        self.start_time = datetime.datetime.now()

    def run_command(self, command: str, args: List[str], cwd: str = None) -> bool:
        """Execute a single command with arguments"""
        full_command = [command] + args if command != 'pytest' else [sys.executable, '-m', command] + args

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(map(str, full_command))}")
            return True

        try:
            logger.info(f"Executing: {' '.join(map(str, full_command))}")
            result = subprocess.run(
                full_command,
                cwd=cwd or str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr[:500]}")
                return False

            logger.info("Command completed successfully")
            return True

        except FileNotFoundError:
            logger.error(f"Command not found: {command}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False

    def run_stage(self, stage: Dict) -> bool:
        """Execute a single pipeline stage"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage: {stage['name']}")
        logger.info(f"Description: {stage['description']}")
        logger.info(f"{'='*60}")

        # Determine if this is a Python script or external command
        script = stage['script']
        args = stage.get('args', [])

        if script.endswith('.py'):
            # Python script
            success = self.run_command(sys.executable, [script] + args)
        else:
            # External command (e.g., pytest)
            success = self.run_command(script, args)

        self.results[stage['name']] = 'SUCCESS' if success else 'FAILED'

        if not success and not self.continue_on_error:
            logger.error("Stage failed. Stopping pipeline.")
            return False

        return True

    def run_pipeline(self, pipeline_name: str) -> bool:
        """Execute a complete pipeline"""
        if pipeline_name not in ANALYSIS_PIPELINES:
            logger.error(f"Unknown pipeline: {pipeline_name}")
            return False

        pipeline = ANALYSIS_PIPELINES[pipeline_name]

        logger.info(f"\n{'#'*80}")
        logger.info(f"# {pipeline['name']}")
        logger.info(f"# {pipeline['description']}")
        logger.info(f"{'#'*80}")

        for stage in pipeline['stages']:
            if not self.run_stage(stage):
                return False

        return True

    def run_workflow(self, workflow: List[str]) -> bool:
        """Execute multiple pipelines in sequence"""
        logger.info(f"\nStarting workflow with {len(workflow)} pipelines")

        for pipeline_name in workflow:
            if not self.run_pipeline(pipeline_name):
                if not self.continue_on_error:
                    return False

        return True

    def print_summary(self):
        """Print execution summary"""
        duration = datetime.datetime.now() - self.start_time

        logger.info(f"\n{'='*80}")
        logger.info("EXECUTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total duration: {duration}")
        logger.info("\nStage Results:")

        for stage_name, result in self.results.items():
            status_symbol = '✓' if result == 'SUCCESS' else '✗'
            logger.info(f"  {status_symbol} {stage_name}: {result}")

        success_count = sum(1 for r in self.results.values() if r == 'SUCCESS')
        total_count = len(self.results)

        logger.info(f"\nTotal: {success_count}/{total_count} stages completed successfully")

        if success_count == total_count:
            logger.info("\n🎉 All stages completed successfully!")
        else:
            logger.warning(f"\n⚠️ {total_count - success_count} stage(s) failed")

# =============================================================================
# CLI Interface
# =============================================================================

def list_pipelines():
    """Display all available pipelines and presets"""
    print("\n" + "="*80)
    print("AVAILABLE ANALYSIS PIPELINES")
    print("="*80)

    for key, pipeline in ANALYSIS_PIPELINES.items():
        print(f"\n[{key}] {pipeline['name']}")
        print(f"  {pipeline['description']}")
        print(f"  Stages: {len(pipeline['stages'])}")
        for i, stage in enumerate(pipeline['stages'], 1):
            print(f"    {i}. {stage['name']}")

    print("\n" + "="*80)
    print("WORKFLOW PRESETS")
    print("="*80)

    for preset_name, pipelines in WORKFLOW_PRESETS.items():
        print(f"\n[{preset_name}]: {', '.join(pipelines)}")

def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="NN4Psych Complete Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available pipelines
  python run_complete_analysis.py --list

  # Run data processing pipeline
  python run_complete_analysis.py --pipeline data

  # Run multiple pipelines
  python run_complete_analysis.py --pipeline data visualization hyperparams

  # Use a workflow preset
  python run_complete_analysis.py --preset full

  # Run validation tests
  python run_complete_analysis.py --preset validate

  # Dry run to see what would be executed
  python run_complete_analysis.py --preset full --dry-run

  # Continue on errors
  python run_complete_analysis.py --preset complete --continue-on-error
        """
    )

    parser.add_argument(
        '--pipeline', '-p',
        nargs='+',
        choices=list(ANALYSIS_PIPELINES.keys()),
        help='Run specific pipeline(s)'
    )

    parser.add_argument(
        '--preset',
        choices=list(WORKFLOW_PRESETS.keys()),
        help='Run a predefined workflow preset'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available pipelines and presets'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue execution even if a stage fails'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'output',
        help='Output directory for results (default: output/)'
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_pipelines()
        return 0

    # Determine what to run
    if args.preset:
        pipelines_to_run = WORKFLOW_PRESETS[args.preset]
        logger.info(f"Using preset '{args.preset}': {pipelines_to_run}")
    elif args.pipeline:
        pipelines_to_run = args.pipeline
    else:
        parser.print_help()
        return 1

    # Create runner and execute
    runner = PipelineRunner(
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error
    )

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run the workflow
    success = runner.run_workflow(pipelines_to_run)

    # Print summary
    runner.print_summary()

    # Save results to JSON
    if not args.dry_run:
        results_file = args.output_dir / f"analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': runner.start_time.isoformat(),
                'duration': str(datetime.datetime.now() - runner.start_time),
                'pipelines': pipelines_to_run,
                'results': runner.results,
                'success': success
            }, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
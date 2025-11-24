#!/usr/bin/env python3
"""
Stage 07: Dynamical Systems Analysis

Analyzes the dynamical systems properties of trained RNN models:
- Finding fixed points (equilibrium states)
- Stability analysis via eigenvalue decomposition
- Line attractor detection via null space analysis
- Trajectory visualization in PCA space

Input: Trained model files (.pth)
Output: Figures in figures/dynamical_systems/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from sklearn.decomposition import PCA
import torch

from config import (
    TRAINED_MODELS_DIR, CHECKPOINTS_DIR, MODEL_PARAMS,
    DYNAMICAL_SYSTEMS_FIGURES_DIR
)
from nn4psych.models import ActorCritic


class FixedPointAnalyzer:
    """Analyze fixed points and dynamics of RNN models."""

    def __init__(self, model: ActorCritic):
        """
        Initialize analyzer with a trained model.

        Parameters
        ----------
        model : ActorCritic
            Trained RNN model.
        """
        self.model = model
        self.W = model.state_dict()['rnn.weight_hh_l0'].numpy()
        self.hidden_dim = self.W.shape[0]

    @staticmethod
    def rnn_dynamics(h: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute RNN dynamics: h' = tanh(W @ h).

        Parameters
        ----------
        h : np.ndarray
            Hidden state.
        W : np.ndarray
            Recurrent weight matrix.

        Returns
        -------
        np.ndarray
            Next hidden state.
        """
        return np.tanh(np.dot(W, h))

    def find_fixed_points(
        self,
        num_iterations: int = 1000,
        tolerance: float = 1e-6,
        max_steps: int = 100,
    ) -> np.ndarray:
        """
        Find fixed points of the RNN dynamics.

        Parameters
        ----------
        num_iterations : int
            Number of random initializations.
        tolerance : float
            Convergence tolerance.
        max_steps : int
            Maximum steps per initialization.

        Returns
        -------
        np.ndarray
            Array of unique fixed points.
        """
        fixed_points = []

        for _ in range(num_iterations):
            h = np.random.randn(self.hidden_dim)

            for _ in range(max_steps):
                h_new = self.rnn_dynamics(h, self.W)

                if np.linalg.norm(h_new - h) < tolerance:
                    fixed_points.append(h_new)
                    break

                h = h_new

        # Remove duplicates
        if fixed_points:
            unique_fps = np.unique(
                np.round(fixed_points, decimals=6),
                axis=0,
            )
            return unique_fps
        return np.array([])

    def compute_jacobian(self, h: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian at a fixed point.

        Parameters
        ----------
        h : np.ndarray
            Fixed point location.

        Returns
        -------
        np.ndarray
            Jacobian matrix.
        """
        derivative = 1 - np.tanh(np.dot(self.W, h)) ** 2
        J = self.W * derivative[:, np.newaxis]
        return J

    def analyze_stability(self, fixed_points: np.ndarray) -> list:
        """
        Analyze stability of fixed points.

        Parameters
        ----------
        fixed_points : np.ndarray
            Array of fixed points.

        Returns
        -------
        list
            List of dicts with stability information.
        """
        results = []

        for i, h in enumerate(fixed_points):
            J = self.compute_jacobian(h)
            eigenvalues = np.linalg.eigvals(J)
            max_eig = np.max(np.abs(eigenvalues))

            results.append({
                'index': i,
                'fixed_point': h,
                'max_eigenvalue': max_eig,
                'stable': max_eig < 1,
                'eigenvalues': eigenvalues,
            })

        return results

    def check_line_attractor(self, tolerance: float = 1e-10) -> tuple:
        """
        Check for line attractors via null space analysis.

        Parameters
        ----------
        tolerance : float
            Tolerance for null space computation.

        Returns
        -------
        tuple
            (has_line_attractor, null_space_basis)
        """
        ns = null_space(self.W, rcond=tolerance)

        if ns.shape[1] > 0:
            return True, ns
        else:
            return False, None

    def simulate_trajectory(
        self,
        h0: np.ndarray,
        num_steps: int = 100,
    ) -> np.ndarray:
        """
        Simulate RNN trajectory from initial state.

        Parameters
        ----------
        h0 : np.ndarray
            Initial hidden state.
        num_steps : int
            Number of simulation steps.

        Returns
        -------
        np.ndarray
            Trajectory array (num_steps + 1, hidden_dim).
        """
        h = h0
        trajectory = [h]

        for _ in range(num_steps):
            h = self.rnn_dynamics(h, self.W)
            trajectory.append(h)

        return np.array(trajectory)

    def visualize_dynamics(
        self,
        fixed_points: np.ndarray,
        trajectory: np.ndarray,
        save_path: Path = None,
    ):
        """
        Visualize RNN dynamics in 2D PCA space.

        Parameters
        ----------
        fixed_points : np.ndarray
            Fixed points to plot.
        trajectory : np.ndarray
            Trajectory to plot.
        save_path : Path, optional
            Path to save figure.
        """
        # PCA projection
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory
        ax.plot(
            trajectory_2d[:, 0],
            trajectory_2d[:, 1],
            'b-',
            alpha=0.6,
            linewidth=2,
            label='RNN Trajectory',
        )
        ax.plot(
            trajectory_2d[0, 0],
            trajectory_2d[0, 1],
            'go',
            markersize=12,
            label='Start',
            zorder=4,
        )
        ax.plot(
            trajectory_2d[-1, 0],
            trajectory_2d[-1, 1],
            'rs',
            markersize=12,
            label='End',
            zorder=4,
        )

        # Plot fixed points
        if len(fixed_points) > 0:
            fixed_points_2d = pca.transform(fixed_points)
            ax.scatter(
                fixed_points_2d[:, 0],
                fixed_points_2d[:, 1],
                c='red',
                s=200,
                marker='*',
                edgecolors='black',
                linewidths=2,
                label=f'Fixed Points (n={len(fixed_points)})',
                zorder=5,
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title('RNN Dynamics in 2D PCA Space', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.svg'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()


def find_best_model(model_dir):
    """Find best performing model in directory."""
    model_dir = Path(model_dir)
    files = list(model_dir.glob("*_V3_0.95g_0.0rm_100bz_*_1.0tds_*_64n_50000e_*.pth"))

    if not files:
        files = list(model_dir.glob("*.pth"))

    if not files:
        return None

    # Sort by loss (first number in filename)
    try:
        sorted_files = sorted(files, key=lambda x: float(x.name.split('_')[0]))
        return sorted_files[0]
    except:
        return files[0]


def main():
    parser = argparse.ArgumentParser(description="Stage 07: Dynamical Systems Analysis")
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to specific model (auto-selects best if not provided)')
    parser.add_argument('--model_dir', type=str,
                       default=str(CHECKPOINTS_DIR / 'model_params_101000'),
                       help='Directory containing models')
    parser.add_argument('--output_dir', type=str,
                       default=str(DYNAMICAL_SYSTEMS_FIGURES_DIR),
                       help='Output directory for figures')
    parser.add_argument('--num_fps', type=int, default=1000,
                       help='Number of random initializations for finding fixed points')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of simulation steps for trajectory')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance for fixed point finding')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 07: DYNAMICAL SYSTEMS ANALYSIS")
    print("=" * 60)

    # Find or load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        print("\n1. Finding best model...")
        model_path = find_best_model(args.model_dir)

    if model_path is None:
        print("ERROR: No model found!")
        return 1

    print(f"   Using: {model_path.name}")

    # Load model
    print("\n2. Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    params = MODEL_PARAMS['actor_critic']
    hidden_dim = params['hidden_dim']

    model = ActorCritic(params['input_dim'], hidden_dim, params['action_dim'])
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"   Loaded model with hidden_dim={hidden_dim}")

    # Create analyzer
    analyzer = FixedPointAnalyzer(model)

    # Find fixed points
    print(f"\n3. Finding fixed points ({args.num_fps} random initializations)...")
    fixed_points = analyzer.find_fixed_points(
        num_iterations=args.num_fps,
        tolerance=args.tolerance,
    )
    print(f"   Found {len(fixed_points)} unique fixed points")

    # Analyze stability
    if len(fixed_points) > 0:
        print("\n4. Stability analysis...")
        stability_results = analyzer.analyze_stability(fixed_points)

        stable_count = sum(1 for r in stability_results if r['stable'])
        unstable_count = len(stability_results) - stable_count

        print(f"   Stable fixed points: {stable_count}")
        print(f"   Unstable fixed points: {unstable_count}")

        for result in stability_results[:5]:  # Show first 5
            stable_str = "STABLE" if result['stable'] else "UNSTABLE"
            print(f"     FP {result['index'] + 1}: {stable_str} "
                  f"(max |λ| = {result['max_eigenvalue']:.4f})")

        if len(stability_results) > 5:
            print(f"     ... ({len(stability_results) - 5} more)")

    else:
        print("\n4. No fixed points found for stability analysis")

    # Check for line attractors
    print("\n5. Line attractor analysis...")
    has_line, null_basis = analyzer.check_line_attractor()
    if has_line:
        print(f"   Line attractor detected! Null space dimension: {null_basis.shape[1]}")
    else:
        print("   No line attractor detected")

    # Simulate trajectory
    print(f"\n6. Simulating trajectory ({args.num_steps} steps)...")
    h0 = np.random.randn(hidden_dim) * 0.5
    trajectory = analyzer.simulate_trajectory(h0, num_steps=args.num_steps)
    print(f"   Trajectory shape: {trajectory.shape}")

    # Visualize
    print("\n7. Generating visualization...")
    output_path = output_dir / f"fixed_points_{model_path.stem}.png"
    analyzer.visualize_dynamics(fixed_points, trajectory, save_path=output_path)
    print(f"   Saved: {output_path}")
    print(f"   Saved: {output_path.with_suffix('.svg')}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

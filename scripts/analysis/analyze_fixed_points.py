#!/usr/bin/env python3
"""
Fixed Point Analysis for RNN Models

Analyzes the dynamical systems properties of trained RNN models, including:
- Finding fixed points
- Stability analysis
- Line attractor detection
- Trajectory visualization

Refactored to use nn4psych package structure.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from sklearn.decomposition import PCA
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nn4psych.models import ActorCritic
from nn4psych.utils.io import load_model
from config import BEHAVIORAL_FIGURES_DIR


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
    ) -> np.ndarray:
        """
        Find fixed points of the RNN dynamics.

        Parameters
        ----------
        num_iterations : int
            Number of random initializations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        np.ndarray
            Array of unique fixed points.
        """
        fixed_points = []

        for _ in range(num_iterations):
            h = np.random.randn(self.hidden_dim)

            for _ in range(100):
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

    def check_line_attractor(self) -> tuple:
        """
        Check for line attractors via null space analysis.

        Returns
        -------
        tuple
            (has_line_attractor, null_space_basis)
        """
        ns = null_space(self.W)

        if ns.shape[1] > 0:
            print(f"Line attractor detected! Null space dimension: {ns.shape[1]}")
            return True, ns
        else:
            print("No line attractor detected.")
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
            label='RNN Trajectory',
        )
        ax.plot(
            trajectory_2d[0, 0],
            trajectory_2d[0, 1],
            'go',
            markersize=10,
            label='Start',
        )
        ax.plot(
            trajectory_2d[-1, 0],
            trajectory_2d[-1, 1],
            'rs',
            markersize=10,
            label='End',
        )

        # Plot fixed points
        if len(fixed_points) > 0:
            fixed_points_2d = pca.transform(fixed_points)
            ax.scatter(
                fixed_points_2d[:, 0],
                fixed_points_2d[:, 1],
                c='red',
                s=100,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                label='Fixed Points',
                zorder=5,
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('RNN Dynamics (2D PCA Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fixed point analysis of RNN models")
    parser.add_argument('model_path', type=str, help='Path to model weights')
    parser.add_argument('--num-fps', type=int, default=1000,
                       help='Number of random inits for finding FPs')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of simulation steps')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # Infer model dimensions from checkpoint
    weight_hh = checkpoint['rnn.weight_hh_l0']
    hidden_dim = weight_hh.shape[0]

    model = ActorCritic(input_dim=9, hidden_dim=hidden_dim, action_dim=3)
    model.load_state_dict(checkpoint)

    print(f"Model loaded with hidden_dim={hidden_dim}")

    # Create analyzer
    analyzer = FixedPointAnalyzer(model)

    # Find fixed points
    print(f"\nFinding fixed points ({args.num_fps} random initializations)...")
    fixed_points = analyzer.find_fixed_points(num_iterations=args.num_fps)
    print(f"Found {len(fixed_points)} unique fixed points")

    # Analyze stability
    if len(fixed_points) > 0:
        print("\nStability Analysis:")
        stability_results = analyzer.analyze_stability(fixed_points)

        for result in stability_results:
            stable_str = "STABLE" if result['stable'] else "UNSTABLE"
            print(f"  FP {result['index'] + 1}: {stable_str} "
                  f"(max |λ| = {result['max_eigenvalue']:.4f})")

    # Check for line attractors
    print("\nLine Attractor Analysis:")
    has_line, null_basis = analyzer.check_line_attractor()

    # Simulate trajectory
    print(f"\nSimulating trajectory ({args.num_steps} steps)...")
    h0 = np.random.randn(hidden_dim)
    trajectory = analyzer.simulate_trajectory(h0, num_steps=args.num_steps)

    # Visualize
    output_dir = Path(args.output_dir) if args.output_dir else BEHAVIORAL_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fixed_points_{Path(args.model_path).stem}.png"

    analyzer.visualize_dynamics(fixed_points, trajectory, save_path=output_path)


if __name__ == "__main__":
    main()

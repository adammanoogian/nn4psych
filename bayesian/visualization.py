"""
Visualization Tools for Bayesian Normative Models

This module provides publication-ready plotting functions for visualizing
Bayesian model fits, parameter estimates, and model diagnostics.

Reference:
    Loosen et al. (2023) - https://link.springer.com/article/10.3758/s13428-024-02427-y
    McGuire et al. (2014) - Functionally Dissociable Influences on Learning Rate
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import seaborn as sns


# Set publication-ready style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_model_fit_comprehensive(
    results: Dict,
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    helicopter_positions: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive 6-panel visualization of model fit.

    Parameters
    ----------
    results : dict
        Model output from fit(..., output='all')
    bucket_positions : np.ndarray
        Observed bucket positions
    bag_positions : np.ndarray
        Observed bag positions
    helicopter_positions : np.ndarray, optional
        True helicopter positions (if available)
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    n_trials = len(bucket_positions)
    trials = np.arange(n_trials)

    # Extract model outputs
    context = results['context']
    params = results['params']

    # Panel 1: Task positions and predictions
    ax1 = fig.add_subplot(gs[0, 0])
    if helicopter_positions is not None:
        ax1.plot(trials, helicopter_positions, 'g-', linewidth=2,
                label='Helicopter', alpha=0.6)
    ax1.scatter(trials, bag_positions, c='red', s=30, alpha=0.5,
               label='Bag', edgecolors='darkred', linewidth=0.5)
    ax1.plot(trials, bucket_positions, 'o-', color='orange', linewidth=2,
            markersize=4, label='Bucket (obs)', alpha=0.8)
    ax1.plot(trials, results['pred_bucket_placement'], '--', color='blue',
            linewidth=2, label='Bucket (pred)', alpha=0.7)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Position')
    ax1.set_title('Task Behavior & Model Predictions')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Learning Rate Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(trials, results['learning_rate'], color='steelblue', linewidth=2)
    ax2.fill_between(trials, 0, results['learning_rate'], alpha=0.3, color='steelblue')
    ax2.axhline(y=results['learning_rate'].mean(), color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f"Mean = {results['learning_rate'].mean():.3f}")
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Learning Rate (α)')
    ax2.set_title('Model Learning Rate Trajectory')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel 3: Prediction Error
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['red' if abs(pe) > 60 else 'gray' for pe in results['pred_error']]
    ax3.scatter(trials, results['pred_error'], c=colors, s=30, alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Large error (>2σ)')
    ax3.axhline(y=-60, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Prediction Error (δ)')
    ax3.set_title('Prediction Errors (Bag - Bucket)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Panel 4: Changepoint Probability (Omega)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(trials, results['omega'], color='purple', linewidth=2)
    ax4.fill_between(trials, 0, results['omega'], alpha=0.3, color='purple')
    ax4.axhline(y=params[0], color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f"H (prior) = {params[0]:.3f}")
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Changepoint Probability (Ω)')
    ax4.set_title('Changepoint Detection')
    ax4.set_ylim(0, 1)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Panel 5: Relative Uncertainty (Tau)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(trials, results['tau'][:-1], color='teal', linewidth=2)  # tau has n+1 elements
    ax5.fill_between(trials, 0, results['tau'][:-1], alpha=0.3, color='teal')
    ax5.set_xlabel('Trial')
    ax5.set_ylabel('Relative Uncertainty (τ)')
    ax5.set_title('Uncertainty Tracking')
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Panel 6: Update Comparison (Observed vs Normative)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(results['normative_update'], results['bucket_update'],
               c=results['learning_rate'], cmap='viridis', s=40, alpha=0.6,
               edgecolors='black', linewidth=0.5)

    # Add identity line
    min_val = min(results['normative_update'].min(), results['bucket_update'].min())
    max_val = max(results['normative_update'].max(), results['bucket_update'].max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5,
            alpha=0.5, label='Perfect match')

    # Calculate R²
    corr = np.corrcoef(results['normative_update'], results['bucket_update'])[0, 1]
    r_squared = corr ** 2

    ax6.set_xlabel('Normative Update')
    ax6.set_ylabel('Observed Update')
    ax6.set_title(f'Update Comparison (R² = {r_squared:.3f})')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax6)
    cbar.set_label('Learning Rate', rotation=270, labelpad=20)

    # Main title with parameters
    title = (f"Bayesian Normative Model: {context.capitalize()}\n"
            f"H={params[0]:.3f}, LW={params[1]:.3f}, UU={params[2]:.3f}, "
            f"σ_motor={params[3]:.3f}, σ_LR={params[4]:.3f} | "
            f"NegLL={results['negll']:.1f}, BIC={results['BIC']:.1f}")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_parameter_distributions(
    param_estimates: Dict[str, np.ndarray],
    true_values: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distributions of parameter estimates across multiple fits.

    Useful for parameter recovery studies or batch fitting results.

    Parameters
    ----------
    param_estimates : dict
        Dictionary with parameter names as keys and arrays of estimates as values
        e.g., {'H': array([...]), 'LW': array([...]), ...}
    true_values : dict, optional
        True parameter values (for recovery studies)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_params = len(param_estimates)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)

    if n_params == 1:
        axes = [axes]

    param_names = ['H', 'LW', 'UU', 'σ_motor', 'σ_LR']
    param_labels = ['Hazard Rate', 'Likelihood Weight', 'Uncertainty Underest.',
                   'Motor Variance', 'LR Variance Slope']

    for i, (param, label) in enumerate(zip(param_names[:n_params], param_labels[:n_params])):
        if param not in param_estimates:
            continue

        ax = axes[i]
        values = param_estimates[param]

        # Plot histogram
        ax.hist(values, bins=20, alpha=0.6, color='steelblue', edgecolor='black')

        # Plot mean
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_val:.3f}')

        # Plot true value if provided
        if true_values and param in true_values:
            ax.axvline(true_values[param], color='red', linestyle='-',
                      linewidth=2, label=f'True = {true_values[param]:.3f}')

        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label}\n({param})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_learning_rate_by_prediction_error(
    results: Dict,
    bins: int = 10,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot learning rate as a function of prediction error magnitude.

    Reproduces key analysis from McGuire et al. (2014).

    Parameters
    ----------
    results : dict
        Model output from fit(..., output='all')
    bins : int
        Number of bins for prediction error
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    pred_error = results['pred_error']
    learning_rate = results['learning_rate']
    abs_pred_error = np.abs(pred_error)

    # Panel 1: Binned relationship
    bin_edges = np.percentile(abs_pred_error, np.linspace(0, 100, bins + 1))
    bin_centers = []
    lr_means = []
    lr_sems = []

    for i in range(bins):
        mask = (abs_pred_error >= bin_edges[i]) & (abs_pred_error < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(abs_pred_error[mask].mean())
            lr_means.append(learning_rate[mask].mean())
            lr_sems.append(learning_rate[mask].std() / np.sqrt(mask.sum()))

    ax1.errorbar(bin_centers, lr_means, yerr=lr_sems, marker='o',
                linestyle='-', linewidth=2, markersize=8, capsize=5,
                color='steelblue', ecolor='gray')
    ax1.set_xlabel('|Prediction Error|')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate vs Prediction Error Magnitude')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter plot with marginal distributions
    ax2.scatter(abs_pred_error, learning_rate, alpha=0.4, s=30,
               c='steelblue', edgecolors='black', linewidth=0.3)
    ax2.set_xlabel('|Prediction Error|')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Trial-by-Trial Learning Rate')
    ax2.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(abs_pred_error, learning_rate)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))

    context = results['context']
    fig.suptitle(f'Learning Rate Analysis: {context.capitalize()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_model_comparison(
    comparison: Dict[str, float],
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize model comparison results.

    Parameters
    ----------
    comparison : dict
        Output from bayesian.model_comparison.compare_contexts()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    models = ['Changepoint', 'Oddball']
    negll_values = [comparison['cp_negll'], comparison['ob_negll']]
    bic_values = [comparison['cp_bic'], comparison['ob_bic']]
    aic_values = [comparison['cp_aic'], comparison['ob_aic']]

    x = np.arange(len(models))
    width = 0.35

    # Panel 1: Negative Log-Likelihood
    bars1 = ax1.bar(x, negll_values, width, color=['steelblue', 'coral'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Negative Log-Likelihood')
    ax1.set_title('Model Likelihood')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, negll_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom')

    # Panel 2: Information Criteria
    bars_bic = ax2.bar(x - width/2, bic_values, width, label='BIC',
                      color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars_aic = ax2.bar(x + width/2, aic_values, width, label='AIC',
                      color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)

    ax2.set_ylabel('Information Criterion')
    ax2.set_title('Model Selection Criteria (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight best model
    best = comparison['best_model']
    best_idx = 0 if best == 'changepoint' else 1
    ax2.patches[best_idx].set_edgecolor('green')
    ax2.patches[best_idx].set_linewidth(3)

    fig.suptitle(f'Model Comparison: Best = {best.capitalize()} '
                f'(ΔB IC = {abs(comparison["bic_difference"]):.1f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_residuals(
    results: Dict,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot diagnostic residual plots for model fit quality.

    Parameters
    ----------
    results : dict
        Model output from fit(..., output='all')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    residuals = results['bucket_update'] - results['normative_update']
    trials = np.arange(len(residuals))

    # Panel 1: Residuals over time
    ax1.scatter(trials, residuals, alpha=0.5, s=30, c='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals Over Time')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residual histogram
    ax2.hist(residuals, bins=20, alpha=0.7, color='steelblue',
            edgecolor='black', density=True)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Residual Distribution\n(Mean={residuals.mean():.3f}, SD={residuals.std():.3f})')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig

# =============================================================================
# NumPyro-Specific Visualization Functions
# =============================================================================


def plot_posterior_comparison(
    mcmc_cp,
    mcmc_ob,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare posterior distributions for changepoint vs oddball models.

    Parameters
    ----------
    mcmc_cp : MCMC
        Fitted MCMC object for changepoint model
    mcmc_ob : MCMC
        Fitted MCMC object for oddball model
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    
    samples_cp = {k: np.array(v) for k, v in mcmc_cp.get_samples().items()}
    samples_ob = {k: np.array(v) for k, v in mcmc_ob.get_samples().items()}

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    param_names = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']
    param_labels = ['Hazard Rate', 'Likelihood Weight', 'Uncertainty Underest.',
                   'Motor Variance', 'LR Variance Slope']

    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]

        # Plot both distributions
        ax.hist(samples_cp[param], bins=30, alpha=0.5, color='blue',
               label='Changepoint', density=True, edgecolor='black')
        ax.hist(samples_ob[param], bins=30, alpha=0.5, color='orange',
               label='Oddball', density=True, edgecolor='black')

        # Add means
        ax.axvline(samples_cp[param].mean(), color='blue', linestyle='--',
                  linewidth=2, alpha=0.7)
        ax.axvline(samples_ob[param].mean(), color='orange', linestyle='--',
                  linewidth=2, alpha=0.7)

        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label} Posterior')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    # Hide extra subplot
    axes[5].axis('off')

    plt.suptitle('Posterior Comparison: Changepoint vs Oddball',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_posterior_pairs(
    mcmc,
    params: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot pairwise posterior distributions (corner plot).

    Useful for detecting parameter correlations and multimodality.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    params : list, optional
        Parameters to include (default: all 5 main parameters)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import corner
    except ImportError:
        print("Warning: 'corner' package not installed. Install with: pip install corner")
        return None

    import numpy as np
    
    samples = mcmc.get_samples()

    if params is None:
        params = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']

    # Extract samples for selected parameters
    samples_array = np.column_stack([np.array(samples[p]) for p in params])

    # Create corner plot
    fig = corner.corner(
        samples_array,
        labels=params,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        smooth=1.0,
        color='steelblue',
        hist_kwargs={'density': True, 'color': 'steelblue'},
        plot_datapoints=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_forest_plot(
    mcmc_dict: Dict[str, any],
    param: str,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create forest plot comparing parameter estimates across models/conditions.

    Parameters
    ----------
    mcmc_dict : dict
        Dictionary mapping model names to MCMC objects
        e.g., {'Model 1': mcmc1, 'Model 2': mcmc2}
    param : str
        Parameter name to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    from numpyro.diagnostics import hpdi
    
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(mcmc_dict.keys())
    n_models = len(model_names)

    means = []
    hpdis = []

    for model_name in model_names:
        mcmc = mcmc_dict[model_name]
        samples = mcmc.get_samples()
        
        param_samples = samples[param]
        mean = np.array(param_samples).mean()
        hpdi_interval = hpdi(param_samples, prob=0.89)

        means.append(mean)
        hpdis.append(hpdi_interval)

    # Plot
    y_pos = np.arange(n_models)

    for i, (mean, (low, high)) in enumerate(zip(means, hpdis)):
        ax.plot([low, high], [i, i], 'o-', linewidth=2, markersize=8,
               color='steelblue')
        ax.plot(mean, i, 'o', markersize=10, color='darkblue')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel(f'{param} Value')
    ax.set_title(f'Forest Plot: {param} (89% HPDI)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_energy_diagnostic(
    mcmc,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MCMC energy diagnostic.

    Divergences between energy distributions indicate sampling problems.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import arviz as az
    except ImportError:
        print("Warning: 'arviz' package not installed. Install with: pip install arviz")
        return None

    idata = az.from_numpyro(mcmc)

    fig, ax = plt.subplots(figsize=figsize)
    az.plot_energy(idata, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig

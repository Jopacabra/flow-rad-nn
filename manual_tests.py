"""
diagnose_model.py

Diagnostic plots to understand model performance.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from train_radiation_nn import RadiationEmulatorInference


def check_training_data(data_file="radiation_training_data.h5"):
    """
    Check training data for problems.

    Some considerations:
    Fraction with I_err > |I|: Integration did not converge at these points. These points are noise.
    Fraction with I < 0: Negative values. Should see a good deal of these, since this is modification to vac. spectrum
    Fraction with |I| < 1e-10: Very small values for which log transformation will cause problems.

    """
    with h5py.File(data_file, 'r') as f:
        I = f['I'][:]
        I_err = f['I_err'][:]

        print(f"Total samples: {len(I)}")
        print(f"NaN values: {np.isnan(I).sum()}")
        print(f"Inf values: {np.isinf(I).sum()}")
        print(f"I range: [{I.min():.4e}, {I.max():.4e}]")
        print(f"I_err range: [{I_err.min():.4e}, {I_err.max():.4e}]")
        print(f"Fraction with I_err > |I|: {(I_err > np.abs(I)).mean():.1%}")  # Noisy data!
        print(f"Fraction with I < 0: {(I < 0).mean():.1%}")
        print(f"Fraction with |I| < 1e-10: {(np.abs(I) < 1e-10).mean():.1%}")


def diagnose_model(
        data_file: str = "radiation_training_data.h5",
        model_file: str = "radiation_emulator.pt",
        normalization_file: str = "radiation_normalization.json",
        n_samples: int = 2000,
):
    """
    Compare model predictions to ground truth data.

    Example considerations:
    Predictions clustered around mean -- Model learned to predict the average -- Architecture too small, or log-transform issue
    Residuals correlated with a feature -- Model missing that dependency -- Need more data varying that feature, or feature engineering
    High residuals for large |I| -- Log-transform not working properly -- Check sign handling in transform/inverse
    Predictions have wrong sign -- Sign handling bug in inverse transform -- Fix the predict() method
    Scattered but unbiased residuals -- Random noise, need more data -- Generate more training data
    """

    # Load ground truth data
    with h5py.File(data_file, 'r') as f:
        X = {
            'x': f['x'][:],
            'kx': f['kx'][:],
            'ky': f['ky'][:],
            'E': f['E'][:],
            'z0': f['z0'][:],
            'zf': f['zf'][:],
            'u_perp': f['u_perp'][:],
            'T': f['T'][:],
            'g': f['g'][:],
        }
        y_true = f['I'][:]
        y_err = f['I_err'][:]

    # Subsample for plotting
    n = min(n_samples, len(y_true))
    idx = np.random.choice(len(y_true), n, replace=False)

    X_sub = {k: v[idx] for k, v in X.items()}
    y_true_sub = y_true[idx]
    y_err_sub = y_err[idx]

    # Load model and predict
    emulator = RadiationEmulatorInference(model_file, normalization_file, device="cpu")
    y_pred = emulator.predict_dict(X_sub)

    # === Plot 1: Predicted vs True ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Linear scale
    ax = axes[0, 0]
    ax.scatter(y_true_sub, y_pred, alpha=0.3, s=5)
    lims = [min(y_true_sub.min(), y_pred.min()), max(y_true_sub.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('True I')
    ax.set_ylabel('Predicted I')
    ax.set_title('Prediction vs Truth (linear)')
    ax.legend()

    # Log scale (for positive values)
    ax = axes[0, 1]
    pos_mask = (y_true_sub > 0) & (y_pred > 0)
    if pos_mask.sum() > 10:
        ax.scatter(y_true_sub[pos_mask], y_pred[pos_mask], alpha=0.3, s=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        lims = [min(y_true_sub[pos_mask].min(), y_pred[pos_mask].min()),
                max(y_true_sub[pos_mask].max(), y_pred[pos_mask].max())]
        ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('True I')
    ax.set_ylabel('Predicted I')
    ax.set_title('Prediction vs Truth (log scale, positive only)')

    # Residuals vs True
    ax = axes[1, 0]
    residuals = y_pred - y_true_sub
    ax.scatter(y_true_sub, residuals, alpha=0.3, s=5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('True I')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residuals vs Truth')

    # Residual histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (std={residuals.std():.2e})')

    plt.tight_layout()
    plt.savefig('model_diagnostics.png', dpi=150)
    plt.show()

    # === Plot 2: Residuals vs Each Input Feature ===
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    feature_names = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g']

    for i, name in enumerate(feature_names):
        ax = axes[i // 3, i % 3]
        ax.scatter(X_sub[name], residuals, alpha=0.3, s=5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel(name)
        ax.set_ylabel('Residual')
        ax.set_title(f'Residual vs {name}')

    plt.tight_layout()
    plt.savefig('residuals_vs_features.png', dpi=150)
    plt.show()

    # === Summary Statistics ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Number of samples: {n}")
    print(f"True I range: [{y_true_sub.min():.4e}, {y_true_sub.max():.4e}]")
    print(f"Pred I range: [{y_pred.min():.4e}, {y_pred.max():.4e}]")
    print(f"Mean residual: {residuals.mean():.4e}")
    print(f"Std residual: {residuals.std():.4e}")
    print(f"Mean absolute error: {np.abs(residuals).mean():.4e}")
    print(f"Mean relative error: {(np.abs(residuals) / (np.abs(y_true_sub) + 1e-10)).mean():.2%}")

    # Check for systematic biases
    print("\n--- Systematic Bias Check ---")
    for name in feature_names:
        corr = np.corrcoef(X_sub[name], residuals)[0, 1]
        print(f"  Correlation(residual, {name}): {corr:+.3f}")

    return y_true_sub, y_pred, residuals

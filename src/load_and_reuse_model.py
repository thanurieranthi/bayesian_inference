"""Example of how to load and reuse the saved BHM model."""

from pathlib import Path
from model_saver import load_model, load_summary
import numpy as np

# Example 1: Load the saved model samples
models_dir = Path(__file__).resolve().parent.parent / "models"
model_samples = load_model(str(models_dir / "bhm_model.pkl"))

print("Available parameters:", list(model_samples.keys()))
print(f"Shape of beta_0 samples: {model_samples['beta_0'].shape}")

# Example 2: Load the posterior summary
posterior_summary = load_summary(str(models_dir / "posterior_summary.csv"))
print("\nPosterior Summary:")
print(posterior_summary.head())

# Example 3: Use the samples for predictions or analysis
# Access specific parameter samples
beta_0_samples = model_samples['beta_0']  # Shape: (num_chains * num_samples,)

# Calculate statistics
mean_beta_0 = np.mean(beta_0_samples)
std_beta_0 = np.std(beta_0_samples)
print(f"\nβ₀: mean={mean_beta_0:.4f}, std={std_beta_0:.4f}")

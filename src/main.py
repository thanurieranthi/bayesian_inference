"""Simple Bayesian ground-motion model example."""

import os
from pathlib import Path
from jax import random

from ground_motion_model import (
    load_ground_motion_data,
    run_inference,
    summarise_posterior,
    generate_ppc,
    sample_prior,
)
from model_saver import save_model
from plot_prior_posterior import plot_prior_posterior_pairs
from plot_residuals import plot_predictive_check

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    """Run Bayesian inference on ground motion data."""
    # Paths
    data_path = Path(__file__).resolve().parent.parent / "data" / "updated_metadata_vel.csv"
    models_dir = Path(__file__).resolve().parent.parent / "models"
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_ground_motion_data(
        csv_path=str(data_path),
        intensity_col="pga_rotd50",
        min_magnitude=4,
        max_magnitude=5
    )
    print(f"  Loaded {data['n_obs']} observations")
    print(f"  Events: {data['num_events']}, Stations: {data['num_stations']}, Site classes: {data['num_site_classes']}\n")
    
    # Run inference
    print("Running MCMC inference...")
    rng_key, ppc_key, prior_key = random.split(random.PRNGKey(42), 3)
    
    mcmc = run_inference(
        data=data,
        rng_key=rng_key,
        num_warmup=2000,
        num_samples=2000,
        num_chains=4,
        progress_bar=True,
    )
    
    # Summarize posterior
    print("\nPosterior Summary:")
    posterior = summarise_posterior(mcmc)
    print(posterior[["mean", "std"]])
    
    # Save results
    posterior.to_csv(models_dir / "posterior_summary.csv")
    print(f"\nSaved posterior summary to {models_dir / 'posterior_summary.csv'}")
    
    # Save full model samples (needed for advanced plotting like PPC density)
    save_model(mcmc, models_dir / "bhm_model.pkl")
    
    # Generate posterior predictive
    print("\nGenerating posterior predictive samples...")
    ppc = generate_ppc(mcmc, data, ppc_key)
    print(f"  Mean prediction (first 5): {ppc['y_pred_mean'][:5]}")
    
    # Plot residuals
    print("Plotting residuals...")
    plot_predictive_check(
        y_obs=ppc["y_obs"],
        y_pred_mean=ppc["y_pred_mean"],
        output_dir=plots_dir
    )
    
    # Generate prior
    print("\nSampling from prior...")
    prior_samples = sample_prior(data, prior_key, num_samples=100)
    posterior_samples = mcmc.get_samples()
    
    # Plot comparisons
    print("Creating plots...")
    plot_prior_posterior_pairs(
        prior_samples,
        posterior_samples,
        plots_dir,
    )
    print(f"Plots saved to {plots_dir}\n")


if __name__ == "__main__":
    main()

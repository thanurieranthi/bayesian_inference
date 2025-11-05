"""Example driver for the Bayesian ground-motion model."""

import numpy as np
from jax import random
import os
from pathlib import Path

from plot_prior_posterior import plot_prior_posterior_pairs
from ground_motion_model import (
    generate_ppc,
    load_ground_motion_data,
    run_inference,
    sample_prior,
    summarise_posterior,
)

# Force CPU execution unless the user explicitly opts in to GPU support.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / \
        "data" / "metadata.csv"

    data = load_ground_motion_data(
        csv_path=str(data_path),
        min_magnitude=3.0,
        max_records=1000,
    )

    rng_key, ppc_key, prior_key = random.split(random.PRNGKey(123), 3)

    mcmc = run_inference(
        data=data,
        rng_key=rng_key,
        num_warmup=10,
        num_samples=100,
        num_chains=4,
        progress_bar=True,
        platform="cpu",
        chain_method="sequential"
    )

    posterior = summarise_posterior(mcmc)
    print(posterior.loc[["beta_0", "beta_magnitude",
          "beta_distance", "beta_depth"]])

    ppc = generate_ppc(mcmc, data, ppc_key)
    print("Posterior predictive mean (first five):", ppc["mean"][:5])

    prior_samples = sample_prior(data, prior_key, num_samples=500)
    posterior_samples = {
        name: np.asarray(values)
        for name, values in mcmc.get_samples().items()
    }

    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plot_prior_posterior_pairs(
        prior_samples,
        posterior_samples,
        plots_dir,
    )


if __name__ == "__main__":
    main()

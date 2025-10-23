"""Example driver for the Bayesian ground-motion model."""

from pathlib import Path

import numpy as np
from jax import random

from ground_motion_model import (
	generate_ppc,
	load_ground_motion_data,
	run_inference,
	sample_prior,
	summarise_posterior,
)
from plot_prior_posterior import plot_prior_posterior_pairs


def main() -> None:
	data_path = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"

	data = load_ground_motion_data(
		csv_path=str(data_path),
		min_magnitude=3.0,
		max_records=100000,
	)

	rng_key, ppc_key, prior_key = random.split(random.PRNGKey(123), 3)

	mcmc = run_inference(
		data=data,
		rng_key=rng_key,
		num_warmup=1500,
		num_samples=3000,
		num_chains=1,
		progress_bar=True,
		platform="cpu",
		chain_method="sequential"
	)

	posterior = summarise_posterior(mcmc)
	print(posterior.loc[["beta_0", "beta_magnitude", "beta_distance", "beta_depth"]])

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

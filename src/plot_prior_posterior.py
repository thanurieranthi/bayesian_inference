"""Utilities for plotting prior and posterior distributions side by side."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt


def _flatten(array: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    flat = np.ravel(array)
    flat = flat[np.isfinite(flat)]
    if flat.size <= max_points:
        return flat
    step = max(flat.size // max_points, 1)
    return flat[::step][:max_points]


def plot_prior_posterior_pairs(
    prior_samples: Dict[str, np.ndarray],
    posterior_samples: Dict[str, np.ndarray],
    output_dir: Path,
    bins: int = 60,
    skip_sites: Iterable[str] = ("y",),
) -> None:
    """Write prior vs posterior histograms for every common site."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_sites = sorted(set(prior_samples) & set(posterior_samples))
    for name in shared_sites:
        if name in skip_sites:
            continue

        prior = np.asarray(prior_samples[name])
        posterior = np.asarray(posterior_samples[name])

        if prior.ndim == 0:
            prior = prior[None]
        if posterior.ndim == 0:
            posterior = posterior[None]

        prior_flat = _flatten(prior)
        posterior_flat = _flatten(posterior)

        if prior_flat.size == 0 or posterior_flat.size == 0:
            # Skip degenerate plots.
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            prior_flat,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Prior",
            color="#1f77b4",
        )
        ax.hist(
            posterior_flat,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Posterior",
            color="#ff7f0e",
        )
        ax.set_title(f"Prior vs Posterior — {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        safe_name = name.replace("/", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=200)
        plt.close(fig)


__all__ = ["plot_prior_posterior_pairs"]

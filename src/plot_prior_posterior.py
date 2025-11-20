"""Utilities for plotting prior and posterior distributions side by side."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _flatten(array: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    flat = np.ravel(array)
    flat = flat[np.isfinite(flat)]
    if flat.size <= max_points:
        return flat
    step = max(flat.size // max_points, 1)
    return flat[::step][:max_points]


def _plot_kde_envelope(ax, data: np.ndarray, x_range, color: str, label: str, linewidth: float = 2.0) -> None:
    """Plot a KDE envelope curve over the data."""
    if data.size < 2:
        return

    # Create KDE
    kde = stats.gaussian_kde(data)

    # Generate smooth x values for the envelope
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    y_vals = kde(x_vals)

    # Plot the envelope curve
    ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, label=label)


def plot_prior_posterior_pairs(
    prior_samples: Dict[str, np.ndarray],
    posterior_samples: Dict[str, np.ndarray],
    output_dir: Path,
    bins: int = 60,
    skip_sites: Iterable[str] = ("y",),
) -> None:
    """Write prior vs posterior histograms with envelope curves for every common site."""
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

        # Plot histograms
        ax.hist(
            prior_flat,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Prior (histogram)",
            color="#1f77b4",
        )
        ax.hist(
            posterior_flat,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Posterior (histogram)",
            color="#ff7f0e",
        )

        # Determine x-range for envelope curves
        all_data = np.concatenate([prior_flat, posterior_flat])
        x_min, x_max = np.min(all_data), np.max(all_data)
        x_range = (x_min, x_max)

        # Plot KDE envelope curves
        _plot_kde_envelope(ax, prior_flat, x_range, color="#1f77b4",
                           label="Prior (envelope)", linewidth=2.5)
        _plot_kde_envelope(ax, posterior_flat, x_range, color="#ff7f0e",
                           label="Posterior (envelope)", linewidth=2.5)

        ax.set_title(f"Prior vs Posterior — {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        safe_name = name.replace("/", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=200)
        plt.close(fig)


__all__ = ["plot_prior_posterior_pairs"]

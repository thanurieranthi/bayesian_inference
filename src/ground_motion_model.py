"""Bayesian hierarchical ground-motion model using NumPyro.

This module builds and fits a multilevel Student-t model for S-wave intensity
observations recorded across New Zealand. It exposes helpers for preparing the
metadata, running MCMC with NUTS, summarising posterior draws, and producing
posterior predictive checks.
"""
from __future__ import annotations
from numpyro.diagnostics import summary as diagnostics_summary
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import Exponential, HalfNormal, Normal, StudentT
from numpyro import plate, sample
import numpyro
from jax import random
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Literal

# Default to CPU execution to avoid platform-specific GPU crashes.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


ArrayDict = Dict[str, jnp.ndarray]
ScalerDict = Dict[str, Tuple[float, float]]


def _zscore(values: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Return mean-centred and standardised values with their (mean, std)."""
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma == 0.0 or np.isnan(sigma):
        sigma = 1.0
    return (values - mu) / sigma, (mu, sigma)


@dataclass
class GroundMotionData:
    """Container for arrays dispatched into the NumPyro model."""

    y: jnp.ndarray
    magnitude: jnp.ndarray
    distance: jnp.ndarray
    depth: jnp.ndarray
    event_id: jnp.ndarray
    station_id: jnp.ndarray
    site_class_id: jnp.ndarray
    event_station_id: jnp.ndarray
    num_events: int
    num_stations: int
    num_site_classes: int
    num_event_station_pairs: int
    scalers: ScalerDict
    lookups: Dict[str, Iterable[str]]

    def to_dict(self) -> ArrayDict:
        return {
            "y": self.y,
            "magnitude": self.magnitude,
            "distance": self.distance,
            "depth": self.depth,
            "event_id": self.event_id,
            "station_id": self.station_id,
            "site_class_id": self.site_class_id,
            "event_station_id": self.event_station_id,
            "num_events": int(self.num_events),
            "num_stations": int(self.num_stations),
            "num_site_classes": int(self.num_site_classes),
            "num_event_station_pairs": int(self.num_event_station_pairs),
        }


def load_ground_motion_data(
    csv_path: str,
    intensity_col: str = "pga_rotd50",
    site_class_col: str = "site_class",
    min_magnitude: Optional[float] = None,
    years: Optional[list[int]] = None,
) -> GroundMotionData:
    """Load metadata and construct arrays for the hierarchical model.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing ground motion data.
    intensity_col : str
        Name of the intensity column (default: "pga_rotd50").
    site_class_col : str
        Name of the site class column (default: "site_class").
    min_magnitude : Optional[float]
        Minimum magnitude threshold for filtering events.
    years : Optional[list[int]]
        List of years to include in the analysis (e.g., [2013, 2014, 2015]).
        If None, all years are included.

    Returns
    -------
    GroundMotionData
        Container with processed arrays for the hierarchical model.
    """
    df = pd.read_csv(csv_path)
    df = df[df["event_type"].str.lower() == "earthquake"].copy()

    if min_magnitude is not None:
        df = df[df["magnitude"] >= min_magnitude]

    # Extract year from trace_name (first 4 digits)
    df["year"] = df["trace_name"].str[:4].astype(int)

    # Filter by years if provided
    if years is not None:
        df = df[df["year"].isin(years)].copy()

    required_cols = [
        intensity_col,
        "magnitude",
        "epicentral_distance",
        "event_depth",
        "station",
        site_class_col,
    ]
    df = df.dropna(subset=required_cols).copy()

    intensity = df[intensity_col].astype(float)
    # Guard logarithm by clipping minuscule accelerations.
    intensity = intensity.clip(lower=1e-6)
    df["log_intensity"] = np.log(intensity)

    df["event_code"] = df["earthquake_id"].astype(str)
    df["station_code"] = df["station"].astype(str)

    # Site class is already in the data
    df["site_class"] = df[site_class_col].astype(str)

    magnitude_scaled, mag_stats = _zscore(df["magnitude"].to_numpy())
    distance_scaled, dist_stats = _zscore(df["epicentral_distance"].to_numpy())
    depth_scaled, depth_stats = _zscore(df["event_depth"].to_numpy())

    event_codes, event_categories = pd.factorize(df["event_code"], sort=True)
    station_codes, station_categories = pd.factorize(
        df["station_code"], sort=True)
    site_codes, site_categories = pd.factorize(df["site_class"], sort=True)

    multi_pairs = pd.MultiIndex.from_arrays([event_codes, station_codes])
    event_station_ids, pair_categories = pd.factorize(multi_pairs, sort=True)

    data = GroundMotionData(
        y=jnp.array(df["log_intensity"].to_numpy(), dtype=jnp.float32),
        magnitude=jnp.array(magnitude_scaled, dtype=jnp.float32),
        distance=jnp.array(distance_scaled, dtype=jnp.float32),
        depth=jnp.array(depth_scaled, dtype=jnp.float32),
        event_id=jnp.array(event_codes, dtype=jnp.int32),
        station_id=jnp.array(station_codes, dtype=jnp.int32),
        site_class_id=jnp.array(site_codes, dtype=jnp.int32),
        event_station_id=jnp.array(event_station_ids, dtype=jnp.int32),
        num_events=len(event_categories),
        num_stations=len(station_categories),
        num_site_classes=len(site_categories),
        num_event_station_pairs=len(pair_categories),
        scalers={
            "magnitude": mag_stats,
            "distance": dist_stats,
            "depth": depth_stats,
        },
        lookups={
            "event_index": event_categories,
            "station_index": station_categories,
            "site_class_index": site_categories,
        },
    )

    return data


def summarise_data(
    data: GroundMotionData,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """Print comprehensive summary statistics of the ground motion data.

    Parameters
    ----------
    data : GroundMotionData
        The processed ground motion data container.
    df : Optional[pd.DataFrame]
        The original dataframe (before processing). If provided, shows additional
        original value statistics.
    """
    print("\n" + "="*70)
    print("BAYESIAN HIERARCHICAL MODEL - DATA SUMMARY")
    print("="*70)

    # Dataset structure
    print("\n📊 DATASET STRUCTURE")
    print("-" * 70)
    print(f"  Total observations:        {len(data.y):,}")
    print(f"  Number of events:          {data.num_events:,}")
    print(f"  Number of stations:        {data.num_stations:,}")
    print(f"  Number of site classes:    {data.num_site_classes}")
    print(f"  Event-station pairs:       {data.num_event_station_pairs:,}")

    # Response variable (log intensity)
    print("\n📈 RESPONSE VARIABLE (Log Intensity)")
    print("-" * 70)
    y_values = np.asarray(data.y)
    print(f"  Mean:                      {y_values.mean():.4f}")
    print(f"  Std Dev:                   {y_values.std():.4f}")
    print(f"  Min:                       {y_values.min():.4f}")
    print(f"  Max:                       {y_values.max():.4f}")
    print(f"  Median:                    {np.median(y_values):.4f}")
    print(f"  Q1 (25%):                  {np.percentile(y_values, 25):.4f}")
    print(f"  Q3 (75%):                  {np.percentile(y_values, 75):.4f}")

    # Predictors (standardized values)
    print("\n📍 PREDICTORS (Standardized)")
    print("-" * 70)

    magnitude = np.asarray(data.magnitude)
    distance = np.asarray(data.distance)
    depth = np.asarray(data.depth)

    print("  MAGNITUDE:")
    print(f"    Mean:                    {magnitude.mean():.4f}")
    print(f"    Std Dev:                 {magnitude.std():.4f}")
    print(
        f"    Range:                   [{magnitude.min():.4f}, {magnitude.max():.4f}]")

    print("  EPICENTRAL DISTANCE:")
    print(f"    Mean:                    {distance.mean():.4f}")
    print(f"    Std Dev:                 {distance.std():.4f}")
    print(
        f"    Range:                   [{distance.min():.4f}, {distance.max():.4f}]")

    print("  EVENT DEPTH:")
    print(f"    Mean:                    {depth.mean():.4f}")
    print(f"    Std Dev:                 {depth.std():.4f}")
    print(
        f"    Range:                   [{depth.min():.4f}, {depth.max():.4f}]")

    # Scaling factors
    print("\n🔧 SCALING FACTORS (Original → Standardized)")
    print("-" * 70)
    for name, (mu, sigma) in data.scalers.items():
        print(f"  {name:20s}: μ={mu:.4f}, σ={sigma:.4f}")

    # Hierarchical levels
    print("\n🏗️  HIERARCHICAL LEVELS")
    print("-" * 70)
    event_cat = data.lookups["event_index"]
    station_cat = data.lookups["station_index"]
    site_cat = data.lookups["site_class_index"]

    print(f"  Events:                    {len(event_cat)}")
    print(f"  Stations:                  {len(station_cat)}")
    print(f"  Site classes:              {len(site_cat)}")
    if len(site_cat) <= 10:
        print(f"    - Classes:               {list(site_cat)}")

    # Observations per hierarchical level
    event_counts = np.bincount(np.asarray(data.event_id))
    station_counts = np.bincount(np.asarray(data.station_id))

    print("\n  OBSERVATIONS PER EVENT:")
    print(f"    Mean:                    {event_counts.mean():.1f}")
    print(f"    Std Dev:                 {event_counts.std():.1f}")
    print(
        f"    Min - Max:               {event_counts.min()} - {event_counts.max()}")

    print("\n  OBSERVATIONS PER STATION:")
    print(f"    Mean:                    {station_counts.mean():.1f}")
    print(f"    Std Dev:                 {station_counts.std():.1f}")
    print(
        f"    Min - Max:               {station_counts.min()} - {station_counts.max()}")

    print("\n" + "="*70 + "\n")


def ground_motion_model(
    y: Optional[jnp.ndarray],
    magnitude: jnp.ndarray,
    distance: jnp.ndarray,
    depth: jnp.ndarray,
    event_id: jnp.ndarray,
    station_id: jnp.ndarray,
    site_class_id: jnp.ndarray,
    event_station_id: jnp.ndarray,
    num_events: int,
    num_stations: int,
    num_site_classes: int,
    num_event_station_pairs: int,
) -> None:
    """Hierarchical Student-t regression for S-wave ground-motion intensity."""
    beta_0 = sample("beta_0", Normal(0.0, 5.0))
    beta_mag = sample("beta_magnitude", Normal(3, ))  # Normal(0.0, 5.0)
    beta_dist = sample("beta_distance", Normal(100, 32))  # Normal(0.0, 5.0)
    beta_depth = sample("beta_depth", Normal(60, 5.0))  # Normal(0.0, 5.0)

    sigma_event = sample("sigma_event", HalfNormal(1.0))
    nu_event = sample("nu_event", Exponential(1.0 / 30.0))
    with plate("events", num_events):
        event_effect = sample(
            "eta_event",
            StudentT(df=nu_event + 1e-3, loc=0.0, scale=sigma_event),
        )

    sigma_station = sample("sigma_station", HalfNormal(1.0))
    nu_station = sample("nu_station", Exponential(1.0 / 30.0))
    with plate("stations", num_stations):
        station_effect = sample(
            "delta_station",
            StudentT(df=nu_station + 1e-3, loc=0.0, scale=sigma_station),
        )

    sigma_site = sample("sigma_site_class", HalfNormal(1.0))
    nu_site = sample("nu_site_class", Exponential(1.0 / 30.0))
    with plate("site_classes", num_site_classes):
        site_class_effect = sample(
            "gamma_site_class",
            StudentT(df=nu_site + 1e-3, loc=0.0, scale=sigma_site),
        )

    sigma_interaction = sample("sigma_interaction", HalfNormal(1.0))
    nu_interaction = sample("nu_interaction", Exponential(1.0 / 30.0))
    with plate("event_station_pairs", num_event_station_pairs):
        interaction_effect = sample(
            "xi_event_station",
            StudentT(df=nu_interaction + 1e-3,
                     loc=0.0, scale=sigma_interaction),
        )

    mu = (
        beta_0
        + beta_mag * magnitude
        + beta_dist * distance
        + beta_depth * depth
        + event_effect[event_id]
        + station_effect[station_id]
        + site_class_effect[site_class_id]
        + interaction_effect[event_station_id]
    )

    sigma_obs = sample("sigma_obs", HalfNormal(1.0))
    nu_obs = sample("nu_obs", Exponential(1.0 / 30.0))
    sample("y", StudentT(df=nu_obs + 1e-3, loc=mu, scale=sigma_obs), obs=y)


def run_inference(
    data: GroundMotionData,
    rng_key: random.KeyArray,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    progress_bar: bool = True,
    platform: Literal["auto", "cpu", "gpu"] = "auto",
    chain_method: Optional[Literal["parallel", "sequential"]] = None,
) -> MCMC:
    """Execute NUTS sampling for the hierarchical model."""
    resolved_platform = platform
    if platform == "auto":
        try:
            from jax.lib import xla_client  # type: ignore  # noqa: F401

            if jax.local_device_count() == 0:
                resolved_platform = "cpu"
            else:
                resolved_platform = jax.default_backend()
        except Exception:
            resolved_platform = "cpu"

    numpyro.set_platform(resolved_platform)

    resolved_chain_method = chain_method or (
        "parallel"
        if num_chains <= max(1, jax.local_device_count())
        else "sequential"
    )

    nuts = NUTS(ground_motion_model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        chain_method=resolved_chain_method,
    )
    mcmc.run(rng_key, **data.to_dict())
    return mcmc


def summarise_posterior(mcmc: MCMC, credible_interval: float = 0.94) -> pd.DataFrame:
    """Return posterior means and central credible intervals as a DataFrame."""
    stats = diagnostics_summary(
        mcmc.get_samples(),
        prob=credible_interval,
        group_by_chain=False,
    )
    frame = pd.DataFrame(stats).T
    frame.index.name = "parameter"
    return frame


def generate_ppc(
    mcmc: MCMC,
    data: GroundMotionData,
    rng_key: random.KeyArray,
) -> Dict[str, np.ndarray]:
    """Draw posterior predictive samples for observed design points."""
    predictive = Predictive(
        ground_motion_model,
        posterior_samples=mcmc.get_samples(),
    )
    draws = predictive(rng_key, **{**data.to_dict(), "y": None})
    y_draws = np.asarray(draws["y"])
    return {
        "posterior_predictive": y_draws,
        "mean": y_draws.mean(axis=0),
        "lower": np.percentile(y_draws, 2.5, axis=0),
        "upper": np.percentile(y_draws, 97.5, axis=0),
        "observed": np.asarray(data.y),
    }


def sample_prior(
    data: GroundMotionData,
    rng_key: random.KeyArray,
    num_samples: int = 2000,
    return_sites: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Draw samples from the model prior conditioned on design matrices."""
    predictive = Predictive(
        ground_motion_model,
        num_samples=num_samples,
        return_sites=return_sites,
    )
    draws = predictive(
        rng_key,
        **{**data.to_dict(), "y": None},
    )
    return {name: np.asarray(value) for name, value in draws.items()}


def make_rng_keys(seed: int = 0) -> Tuple[random.KeyArray, random.KeyArray]:
    """Produce a pair of reproducible PRNG keys for inference utilities."""
    root = random.PRNGKey(seed)
    return random.split(root, 2)


__all__ = [
    "GroundMotionData",
    "generate_ppc",
    "ground_motion_model",
    "load_ground_motion_data",
    "make_rng_keys",
    "sample_prior",
    "run_inference",
    "summarise_data",
    "summarise_posterior",
]

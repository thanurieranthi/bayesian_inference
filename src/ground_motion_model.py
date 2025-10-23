"""Bayesian hierarchical ground-motion model using NumPyro.

This module builds and fits a multilevel Student-t model for S-wave intensity
observations recorded across New Zealand. It exposes helpers for preparing the
metadata, running MCMC with NUTS, summarising posterior draws, and producing
posterior predictive checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

import numpyro
from numpyro import plate, sample
from numpyro.distributions import Exponential, HalfNormal, Normal, StudentT
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary as diagnostics_summary

ArrayDict = Dict[str, jnp.ndarray]
ScalerDict = Dict[str, Tuple[float, float]]


def _zscore(values: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Return mean-centred and standardised values with their (mean, std)."""
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma == 0.0 or np.isnan(sigma):
        sigma = 1.0
    return (values - mu) / sigma, (mu, sigma)


def _ensure_site_class(
    stations: pd.Series,
    site_class: Optional[pd.Series] = None,
    site_class_mapping: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """Provide a site-class series, falling back to station-based heuristics."""
    if site_class is not None:
        filled = site_class.fillna("Unknown")
        if filled.eq("Unknown").all() and site_class_mapping:
            mapped = stations.map(site_class_mapping)
            filled = mapped.fillna("Unknown")
        return filled

    if site_class_mapping:
        mapped = stations.map(site_class_mapping)
        return mapped.fillna("Unknown")

    # Fallback: coarse proxy built from the first letter of the station code.
    return stations.str[:1].fillna("U")


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
    max_records: Optional[int] = None,
    site_class_mapping: Optional[Dict[str, str]] = None,
) -> GroundMotionData:
    """Load metadata and construct arrays for the hierarchical model."""
    df = pd.read_csv(csv_path)
    df = df[df["event_type"].str.lower() == "earthquake"].copy()

    if min_magnitude is not None:
        df = df[df["magnitude"] >= min_magnitude]

    if max_records is not None:
        df = df.head(max_records).copy()

    required_cols = [
        intensity_col,
        "magnitude",
        "epicentral_distance",
        "event_depth",
        "station",
    ]
    df = df.dropna(subset=required_cols).copy()

    intensity = df[intensity_col].astype(float)
    # Guard logarithm by clipping minuscule accelerations.
    intensity = intensity.clip(lower=1e-6)
    df["log_intensity"] = np.log(intensity)

    df["event_code"] = df["trace_name"].str.split("_").str[0]
    df["station_code"] = df["station"].astype(str)

    site_series = None
    if site_class_col in df.columns:
        site_series = df[site_class_col].astype(str)

    df["site_class"] = _ensure_site_class(
        stations=df["station_code"],
        site_class=site_series,
        site_class_mapping=site_class_mapping,
    )

    magnitude_scaled, mag_stats = _zscore(df["magnitude"].to_numpy())
    distance_scaled, dist_stats = _zscore(df["epicentral_distance"].to_numpy())
    depth_scaled, depth_stats = _zscore(df["event_depth"].to_numpy())

    event_codes, event_categories = pd.factorize(df["event_code"], sort=True)
    station_codes, station_categories = pd.factorize(df["station_code"], sort=True)
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
    beta_mag = sample("beta_magnitude", Normal(0.0, 5.0))
    beta_dist = sample("beta_distance", Normal(0.0, 5.0))
    beta_depth = sample("beta_depth", Normal(0.0, 5.0))

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
            StudentT(df=nu_interaction + 1e-3, loc=0.0, scale=sigma_interaction),
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
    "summarise_posterior",
]

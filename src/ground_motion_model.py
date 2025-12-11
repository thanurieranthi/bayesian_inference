"""Simple Bayesian ground-motion model using NumPyro."""
from __future__ import annotations

from numpyro.diagnostics import summary as diagnostics_summary
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import HalfNormal, Normal, StudentT
from numpyro import sample, plate
import numpyro
from jax import random
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

import os
from typing import Dict, Optional, Tuple, Literal

# Default to CPU execution
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _standardize(values: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Standardize values and return (standardized, (mean, std))."""
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma == 0.0 or np.isnan(sigma):
        sigma = 1.0
    return (values - mu) / sigma, (mu, sigma)


def load_ground_motion_data(
    csv_path: str,
    intensity_col: str = "pga_rotd50",
    min_magnitude: Optional[float] = None,
    max_magnitude: Optional[float] = None
) -> Dict:

    df = pd.read_csv(csv_path)
    
    # Basic filtering
    if min_magnitude is not None:
        df = df[df["magnitude"] >= min_magnitude].copy()
    if max_magnitude is not None:
        df = df[df["magnitude"] <= max_magnitude].copy()
    
    required_cols = [intensity_col, "magnitude", "epicentral_distance", "event_depth", 
                     "earthquake_id", "station", "site_class"]
    df = df.dropna(subset=required_cols).copy()
    
    # Log-transform intensity (PGA)
    intensity = df[intensity_col].astype(float).clip(lower=1e-6)
    y = np.log(intensity)
    
    # Standardize predictors
    #magnitude, mag_stats = _standardize(df["magnitude"].values)
    magnitude, mag_stats = df["magnitude"].values, []
    
    # 1. Log-transform distance first (add small epsilon if distance can be 0)
    #log_dist = np.log10(df["epicentral_distance"].values + 1.0)
    distance, dist_stats = df["epicentral_distance"].values, []

    # 2. Then standardize the log-distance
    #distance, dist_stats = _standardize(log_dist)
    
    depth, depth_stats = df["event_depth"].values, []
    
    # Factorize hierarchical indices (for between-event, between-station, between-site-class effects)
    event_id, event_categories = pd.factorize(df["earthquake_id"], sort=True)
    station_id, station_categories = pd.factorize(df["station"], sort=True)
    site_class_id, site_categories = pd.factorize(df["site_class"], sort=True)
    
    return {
        "y": jnp.array(y, dtype=jnp.float32),
        "magnitude": jnp.array(magnitude, dtype=jnp.float32),
        "distance": jnp.array(distance, dtype=jnp.float32),
        "depth": jnp.array(depth, dtype=jnp.float32),
        "event_id": jnp.array(event_id, dtype=jnp.int32),
        "station_id": jnp.array(station_id, dtype=jnp.int32),
        "site_class_id": jnp.array(site_class_id, dtype=jnp.int32),
        "num_events": len(event_categories),
        "num_stations": len(station_categories),
        "num_site_classes": len(site_categories),
        "scalers": {
            "magnitude": mag_stats,
            "distance": dist_stats,
            "depth": depth_stats,
        },
        "n_obs": len(y),
    }



def ground_motion_model(
    y: Optional[jnp.ndarray],
    magnitude: jnp.ndarray,
    distance: jnp.ndarray,
    depth: jnp.ndarray,
    event_id: jnp.ndarray,
    station_id: jnp.ndarray,
    site_class_id: jnp.ndarray,
    num_events: int,
    num_stations: int,
    num_site_classes: int,
) -> None:
    """Hierarchical Bayesian model for ground motion intensity.
    
    Model structure:
    Y_es ~ StudentT(df=4, loc=μ_total, scale=σ_phi)
    
    where:
    μ_total = μ_es + δB_e + δS_s + δC_c
    μ_es = β₁ + β₂*M_e + β₃*log₁₀(R_es) + β₄*D_e
    
    δB_e: Between-event effect (source physics variability)
    δS_s: Between-station effect (station response)
    δC_c: Between-site-class effect (crustal site classification)
    σ_phi: Within-event aleatory variability (combines path effects, record-to-record variability, and measurement error)
    """
    
    # ===== EPISTEMIC CORE: Median prediction parameters =====
    # Intercept (β₁)
    beta_1 = sample("beta_1", Normal(0.0, 5.0))
    
    # Magnitude coefficient (β₂)
    beta_2 = sample("beta_2_magnitude", Normal(0.0, 2.0))
    
    # Distance coefficient (β₃) - log₁₀(R_es)
    beta_3 = sample("beta_3_distance", Normal(0.0, 2.0))
    
    # Depth coefficient (β₄)
    beta_4 = sample("beta_4_depth", Normal(0.0, 2.0))
    
    # ===== ALEATORY HIERARCHY =====
    
    # Between-event effect (δB_e): Variability in source physics
    sigma_B = sample("sigma_B", HalfNormal(1.0))
    with plate("events", num_events):
        delta_B = sample("delta_B_event", StudentT(df=4.0, loc=0.0, scale=sigma_B))
    
    # Between-station effect (δS_s): Station response deviations
    sigma_S = sample("sigma_S", HalfNormal(1.0))
    with plate("stations", num_stations):
        delta_S = sample("delta_S_station", StudentT(df=4.0, loc=0.0, scale=sigma_S))
    
    # Between-site-class effect (δC_c): Site classification response
    sigma_C = sample("sigma_C", HalfNormal(1.0))
    with plate("site_classes", num_site_classes):
        delta_C = sample("delta_C_siteclass", StudentT(df=4.0, loc=0.0, scale=sigma_C))
    
    # Within-event effect (δWS_es): Record-to-record variability (Aleatory Uncertainty)
    # We use this as the scale for the likelihood, replacing the redundant delta_WS + sigma_R structure.
    sigma_phi = sample("sigma_phi", HalfNormal(1.0))
    
    # ===== CONSTRUCT MEAN =====
    mu = (
        beta_1 
        + beta_2 * magnitude 
        + beta_3 * distance 
        + beta_4 * depth
        + delta_B[event_id]
        + delta_S[station_id]
        + delta_C[site_class_id]
    )
    
    # ===== LIKELIHOOD =====
    # Using StudentT likelihood to capture heavy tails and outliers directly
    sample("y", StudentT(df=4.0, loc=mu, scale=sigma_phi), obs=y)


def run_inference(
    data: Dict,
    rng_key: random.KeyArray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    progress_bar: bool = True,
) -> MCMC:
    """Run MCMC inference for the model."""
    numpyro.set_platform("cpu")
    
    nuts = NUTS(ground_motion_model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        chain_method="sequential",
    )
    
    # Extract model inputs (exclude scalers and n_obs metadata)
    model_inputs = {
        "y": data["y"],
        "magnitude": data["magnitude"],
        "distance": data["distance"],
        "depth": data["depth"],
        "event_id": data["event_id"],
        "station_id": data["station_id"],
        "site_class_id": data["site_class_id"],
        "num_events": data["num_events"],
        "num_stations": data["num_stations"],
        "num_site_classes": data["num_site_classes"],
    }
    mcmc.run(rng_key, **model_inputs)
    return mcmc


def summarise_posterior(mcmc: MCMC) -> pd.DataFrame:
    """Return posterior statistics as a DataFrame."""
    stats = diagnostics_summary(mcmc.get_samples(), group_by_chain=False)
    frame = pd.DataFrame(stats).T
    frame.index.name = "parameter"
    return frame


def generate_ppc(
    mcmc: MCMC,
    data: Dict,
    rng_key: random.KeyArray,
) -> Dict[str, np.ndarray]:
    """Generate posterior predictive samples."""
    predictive = Predictive(
        ground_motion_model,
        posterior_samples=mcmc.get_samples(),
    )
    
    model_inputs = {
        "y": None,
        "magnitude": data["magnitude"],
        "distance": data["distance"],
        "depth": data["depth"],
        "event_id": data["event_id"],
        "station_id": data["station_id"],
        "site_class_id": data["site_class_id"],
        "num_events": data["num_events"],
        "num_stations": data["num_stations"],
        "num_site_classes": data["num_site_classes"],
    }
    
    draws = predictive(rng_key, **model_inputs)
    y_pred = np.asarray(draws["y"])
    
    return {
        "y_pred": y_pred,
        "y_pred_mean": y_pred.mean(axis=0),
        "y_obs": np.asarray(data["y"]),
    }


def sample_prior(
    data: Dict,
    rng_key: random.KeyArray,
    num_samples: int = 1000,
) -> Dict[str, np.ndarray]:
    """Draw samples from the model prior."""
    predictive = Predictive(
        ground_motion_model,
        num_samples=num_samples,
    )
    
    model_inputs = {
        "y": None,
        "magnitude": data["magnitude"],
        "distance": data["distance"],
        "depth": data["depth"],
        "event_id": data["event_id"],
        "station_id": data["station_id"],
        "site_class_id": data["site_class_id"],
        "num_events": data["num_events"],
        "num_stations": data["num_stations"],
        "num_site_classes": data["num_site_classes"],
    }
    
    draws = predictive(rng_key, **model_inputs)
    return {name: np.asarray(value) for name, value in draws.items()}


def make_rng_keys(seed: int = 0) -> Tuple[random.KeyArray, random.KeyArray]:
    """Produce a pair of reproducible PRNG keys for inference utilities."""
    root = random.PRNGKey(seed)
    return random.split(root, 2)


__all__ = [
    "load_ground_motion_data",
    "ground_motion_model",
    "run_inference",
    "summarise_posterior",
    "generate_ppc",
    "sample_prior",
    "make_rng_keys",
]

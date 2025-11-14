"""Simple utilities to save and load Bayesian Hierarchical Models."""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from numpyro.infer import MCMC


def save_model(mcmc: MCMC, model_path: str) -> None:
    """Save the MCMC model and its samples to a file.

    Args:
        mcmc: Fitted MCMC object from NumPyro
        model_path: Path where the model will be saved
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Extract samples and convert to numpy arrays for serialization
    samples = mcmc.get_samples()
    samples_numpy = {
        name: np.asarray(values)
        for name, values in samples.items()
    }

    # Save using pickle
    with open(model_path, "wb") as f:
        pickle.dump(samples_numpy, f)

    print(f"✓ Model saved to: {model_path}")


def load_model(model_path: str) -> Dict[str, np.ndarray]:
    """Load the saved MCMC model samples.

    Args:
        model_path: Path to the saved model file

    Returns:
        Dictionary of posterior samples
    """
    with open(model_path, "rb") as f:
        samples = pickle.load(f)

    print(f"✓ Model loaded from: {model_path}")
    return samples


def save_summary(posterior_summary: Any, summary_path: str) -> None:
    """Save the posterior summary statistics.

    Args:
        posterior_summary: Summary dataframe from numpyro
        summary_path: Path where the summary will be saved
    """
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    posterior_summary.to_csv(summary_path)
    print(f"✓ Summary saved to: {summary_path}")


def load_summary(summary_path: str) -> Any:
    """Load the posterior summary statistics.

    Args:
        summary_path: Path to the saved summary file

    Returns:
        Summary dataframe
    """
    import pandas as pd
    summary = pd.read_csv(summary_path, index_col=0)
    print(f"✓ Summary loaded from: {summary_path}")
    return summary

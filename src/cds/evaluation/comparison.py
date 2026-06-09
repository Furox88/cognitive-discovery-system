"""
Rigorous model comparison engine.

Implements the exact statistical machinery used for systematic
comparison of RBA against dark matter profiles (Einasto, NFW, Burkert, etc.)
on real galaxy data.

Core ideas (non-weak):
- Per-dataset (per-galaxy) model selection via BIC
- Winner determination
- ΔBIC strength classification (standard scale for evidence)
- Aggregate statistics (counts, fractions, strength distribution)

This turns "which model is better?" into a reproducible, quantitative workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def compute_delta_bic_strength(deltas: list[float] | np.ndarray) -> pd.Series:
    """
    Bucket ΔBIC values into standard evidence strength categories.

    Buckets (common interpretation):
        < 2     : inconclusive
        2-6     : positive evidence for the best model
        6-10    : strong evidence
        >=10    : very strong evidence
    """
    def bucket(x: float) -> str:
        if pd.isna(x):
            return "NA"
        if x < 2:
            return "<2"
        if x < 6:
            return "2-6"
        if x < 10:
            return "6-10"
        return ">=10"

    return pd.Series(deltas).apply(bucket)


@dataclass
class ModelComparison:
    """
    Compare a set of models across many independent datasets.

    Expected input DataFrame columns (at minimum):
        galaxy | model | BIC | AIC | chi2 | lnL | N
    """

    results: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"galaxy", "model", "BIC"}
        missing = required - set(self.results.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = self.results.copy()
        for col in ["BIC", "AIC", "chi2", "lnL", "N"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def per_dataset_winners(self, criterion: str = "BIC") -> pd.DataFrame:
        """Return the best model per galaxy according to the chosen criterion."""
        if criterion not in self.df.columns:
            raise ValueError(f"Criterion '{criterion}' not in data")

        winners = (
            self.df.loc[self.df.groupby("galaxy")[criterion].idxmin()]
            .copy()
            .rename(columns={"model": f"best_model_{criterion}", criterion: f"best_{criterion}"})
        )
        return winners

    def compute_deltas(self, criterion: str = "BIC") -> pd.DataFrame:
        """Add Δ(criterion) = criterion - min(criterion) per galaxy."""
        min_val = self.df.groupby("galaxy")[criterion].transform("min")
        self.df[f"d{criterion}"] = self.df[criterion] - min_val
        return self.df

    def strength_distribution(self, criterion: str = "BIC") -> pd.DataFrame:
        """
        Return the distribution of evidence strength for the winning model.
        """
        if f"d{criterion}" not in self.df.columns:
            self.compute_deltas(criterion)

        d2 = (
            self.df.sort_values(["galaxy", criterion])
            .groupby("galaxy")[criterion]
            .apply(lambda x: list(x)[1] - list(x)[0] if len(x) >= 2 else np.nan)
        )

        buckets = compute_delta_bic_strength(d2.values)
        counts = buckets.value_counts().rename_axis("bucket").reset_index(name="count")
        counts["fraction"] = counts["count"] / counts["count"].sum()
        return counts

    def summary(self, criterion: str = "BIC") -> pd.DataFrame:
        """Overall win counts per model."""
        winners = self.per_dataset_winners(criterion)
        best_col = f"best_model_{criterion}"
        counts = (
            winners[best_col]
            .value_counts()
            .rename_axis("model")
            .reset_index(name="count")
        )
        counts["fraction"] = counts["count"] / counts["count"].sum()
        return counts

    def full_report(self) -> dict[str, Any]:
        """Convenience bundle of the most useful outputs."""
        winners = self.per_dataset_winners("BIC")
        strength = self.strength_distribution("BIC")
        summary = self.summary("BIC")

        return {
            "winners": winners,
            "strength_distribution": strength,
            "model_counts": summary,
            "n_datasets": winners.shape[0],
        }

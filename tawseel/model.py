"""Failure-risk baseline for Tawseel COD parcels.

Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


TARGET = "failed"

# Only information available when the parcel is created.
# Exclude row identifiers and information recorded after the delivery attempt.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time customer history without modifying the input."""
    required = {"customer_id", "created_at", "failed"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(next(iter(sorted(missing))))

    out = df.copy()

    prior_counts = pd.Series(0, index=out.index, dtype="int64")
    prior_rates = pd.Series(np.nan, index=out.index, dtype="float64")

    # Work on a copy and preserve the original row order and index.
    work = out[["customer_id", "created_at", "failed"]].copy()
    work["_original_index"] = out.index

    work = work.sort_values(
        "created_at",
        kind="mergesort",
        na_position="last",
    )

    # Process each customer independently.
    for _, customer_rows in work.groupby(
        "customer_id",
        sort=False,
        dropna=False,
    ):
        history_count = 0
        history_failures = 0

        # Process timestamps in chronological order.
        # Parcels created at the same time must not count as prior parcels.
        for _, timestamp_rows in customer_rows.groupby(
            "created_at",
            sort=False,
            dropna=False,
        ):
            indices = timestamp_rows["_original_index"].tolist()

            for idx in indices:
                prior_counts.loc[idx] = history_count

                if history_count > 0:
                    prior_rates.loc[idx] = (
                        history_failures / history_count
                    )

            # Every earlier parcel contributes to the parcel count.
            # Only labels equal to 1 contribute to the failure count.
            history_count += len(timestamp_rows)
            history_failures += int(
                (timestamp_rows["failed"] == 1).sum()
            )

    out["customer_prior_parcels"] = (
        prior_counts.reindex(out.index).astype("int64")
    )
    out["customer_prior_fail_rate"] = prior_rates.reindex(out.index)

    return out


def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically, keeping all rows at the boundary in test."""
    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be strictly between 0 and 1"
        )

    if time_col not in df.columns:
        raise KeyError(time_col)

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = ceil(len(df) * test_fraction)

    ordered = df.sort_values(
        time_col,
        kind="mergesort",
    )

    boundary = ordered.iloc[len(df) - n_test][time_col]

    test_mask = df[time_col] >= boundary
    train_mask = df[time_col] < boundary

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    return train, test


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an unfitted preprocessing/model pipeline."""
    numeric_features = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    categorical_features = [
        "customer_id",
        "merchant_id",
        "area",
        "promised_slot",
        "created_at",
    ]

    def address_text_values(X):
        """Convert the address column into documents for TF-IDF."""
        return X["address_text"].fillna("").astype(str).to_numpy()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    address_pipeline = Pipeline(
        steps=[
            (
                "to_text",
                FunctionTransformer(
                    address_text_values,
                    validate=False,
                ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=300,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "address",
                address_pipeline,
                ["address_text"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate predicted failure probabilities on labelled test data."""
    y_true = test_df[TARGET].to_numpy()
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(test_df)),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compare the pipeline with a prior-only dummy classifier."""
    history_columns = [
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    model = build_pipeline(seed)
    model.fit(train_df, train_df[TARGET])

    dummy = DummyClassifier(strategy="prior")

    dummy.fit(
        train_df[FEATURE_COLUMNS + history_columns],
        train_df[TARGET],
    )

    model_metrics = evaluate(model, test_df)

    dummy_input = test_df[FEATURE_COLUMNS + history_columns]
    dummy_probabilities = dummy.predict_proba(dummy_input)[:, 1]
    y_true = test_df[TARGET].to_numpy()

    dummy_metrics = {
        "roc_auc": float(
            roc_auc_score(y_true, dummy_probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, dummy_probabilities)
        ),
        "brier": float(
            brier_score_loss(y_true, dummy_probabilities)
        ),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(test_df)),
    }

    return {
        "model": model_metrics,
        "dummy": dummy_metrics,
        "lift_pr_auc": float(
            model_metrics["pr_auc"] - dummy_metrics["pr_auc"]
        ),
    }
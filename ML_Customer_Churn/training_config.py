from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    data: Path
    model_out: Path
    test_size: float
    random_state: int
    preview_rows: int
    preprocessed_out: Path
    categorical_encoding: Literal["onehot", "ordinal"]
    numeric_scaler: Literal["standard", "bad"]


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for customer churn prediction."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the churn CSV file.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("logistic_regression_churn.pkl"),
        help="Path where the trained model pipeline will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="How many rows of preprocessed training data to print.",
    )
    parser.add_argument(
        "--preprocessed-out",
        type=Path,
        default=Path("preprocessed_training_data.csv"),
        help="Path where the preprocessed training data will be saved as CSV.",
    )
    parser.add_argument(
        "--categorical-encoding",
        choices=["onehot", "ordinal"],
        default="onehot",
        help=(
            "How to encode categorical features. 'ordinal' is useful as a teaching "
            "example of an arbitrary and often-wrong numeric encoding."
        ),
    )
    parser.add_argument(
        "--numeric-scaler",
        choices=["standard", "bad"],
        default="standard",
        help=(
            "How to scale numeric features. 'bad' deliberately pushes different "
            "features onto wildly different magnitudes as a teaching example."
        ),
    )
    args = parser.parse_args()

    return TrainingConfig(
        data=args.data,
        model_out=args.model_out,
        test_size=args.test_size,
        random_state=args.random_state,
        preview_rows=args.preview_rows,
        preprocessed_out=args.preprocessed_out,
        categorical_encoding=args.categorical_encoding,
        numeric_scaler=args.numeric_scaler,
    )

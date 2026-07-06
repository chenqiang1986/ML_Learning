from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def transformed_to_dataframe(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def select_teaching_preview_columns(
    preprocessed_df: pd.DataFrame,
    categorical_encoding: str,
) -> pd.DataFrame:
    preview_columns = ["num__MonthlyCharges"]
    if categorical_encoding == "onehot":
        preview_columns.extend(
            [
                "cat__Contract_Month-to-month",
                "cat__Contract_One year",
                "cat__Contract_Two year",
            ]
        )
    else:
        preview_columns.append("cat__Contract")

    return preprocessed_df[preview_columns]


def print_preprocessor_summary(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    preprocessor = model.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["num"]
    categorical_pipeline = preprocessor.named_transformers_["cat"]

    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["encoder"]

    scaler_summary_dict = {"feature": numeric_features}
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    scale_factors = getattr(scaler, "scale_factors_", None)

    if mean is not None:
        scaler_summary_dict["mean"] = mean
    if scale is not None:
        scaler_summary_dict["scale"] = scale
    if scale_factors is not None:
        scaler_summary_dict["multiplier"] = scale_factors

    scaler_summary = pd.DataFrame(scaler_summary_dict)
    print(f"\n{scaler.__class__.__name__} summary:")
    print(scaler_summary.to_string(index=False))

    print(f"\n{encoder.__class__.__name__} categories:")
    for feature_name, categories in zip(categorical_features, encoder.categories_):
        print(f"{feature_name}: {list(categories)}")


def show_preprocessing_preview(
    model: Pipeline,
    X_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_encoding: str,
    numeric_scaler: str,
    preview_rows: int,
    preprocessed_out: Path,
) -> None:
    print_preprocessor_summary(model, numeric_features, categorical_features)

    preprocessor = model.named_steps["preprocessor"]
    preprocessed_train_df = transformed_to_dataframe(preprocessor, X_train)
    preprocessed_preview = select_teaching_preview_columns(
        preprocessed_train_df,
        categorical_encoding,
    ).head(preview_rows)

    monthly_charges_label = (
        "standard-scaled"
        if numeric_scaler == "standard"
        else "badly scaled"
    )

    if categorical_encoding == "onehot":
        print(
            f"\nPreprocessed training data preview for {monthly_charges_label} "
            f"'MonthlyCharges' and 'Contract' "
            f"(first {len(preprocessed_preview)} rows):"
        )
    else:
        print(
            f"\nPreprocessed training data preview for {monthly_charges_label} "
            f"'MonthlyCharges' and "
            f"ordinal-encoded 'Contract' (first {len(preprocessed_preview)} rows):"
        )
        contract_feature_index = categorical_features.index("Contract")
        contract_categories = (
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .categories_[contract_feature_index]
        )
        contract_mapping = {
            category: index for index, category in enumerate(contract_categories)
        }
        print(f"Ordinal mapping used for Contract: {contract_mapping}")

    print(preprocessed_preview.to_string())
    preprocessed_train_df.to_csv(preprocessed_out, index=False)
    print(f"Saved preprocessed training data to: {preprocessed_out}")

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from preprocess_preview import show_preprocessing_preview
from sklearn.callback import ProgressBar, ScoringMonitor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from teaching_scalers import BadMagnitudeScaler
from training_config import parse_args


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    print("Raw Data:\n", data)

    # The customer id is an identifier, not a predictive signal.
    data = data.drop(columns=["customerID"])

    # Some TotalCharges values are stored as text and may contain blanks.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    
    print("Data Numericalize Churn:\n", data)

    features = data.drop(columns=["Churn"])
    target = data["Churn"]
    return features, target

def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_encoding: str,
    numeric_scaler: str,
) -> Pipeline:
    # Imputer defines, what value the machine should treat it by default, if missing
    # Scaler defines how to rescale the data before training.
    if numeric_scaler == "standard":
        scaler = StandardScaler()
    else:
        scaler = BadMagnitudeScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    if categorical_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        # this is actually a wrong way, we just demonstrate for teaching.
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # f(w, X) = \sum_{i} w[i] * X[i], where w is TBD coefficients.
    # p(w, X) = exp(f(w,X)) / (exp(f(w,X)) + 1) 
    #
    # loss(w) = - y ln p(w,X) - (1-y) ln (1-p(w,X)) 
    #
    # Training process essential: Find a w to minimize loss function
    classifier = LogisticRegression(max_iter=10, solver="lbfgs", verbose=1)
    scoring_monitor = None
    monitoring_mode = "verbose-only"

    try:
        scoring_monitor = ScoringMonitor(scoring={"accuracy": "accuracy"})
        callbacks = [scoring_monitor]

        try:
            callbacks.insert(0, ProgressBar())
            monitoring_mode = "progress-bar-and-scoring"
        except Exception:
            monitoring_mode = "scoring-only"

        classifier.set_callbacks(*callbacks)
    except Exception:
        scoring_monitor = None

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model._scoring_monitor = scoring_monitor
    model._fit_monitoring_mode = monitoring_mode
    model._categorical_encoding = categorical_encoding
    model._numeric_scaler = numeric_scaler
    return model


def main() -> None:
    args = parse_args()

    X, y = load_data(args.data)
    
    print("X:\n", X)
    print("y:\n", y)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    print("numeric_features:", numeric_features)
    print("categorical_features:", categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(
        numeric_features,
        categorical_features,
        args.categorical_encoding,
        args.numeric_scaler,
    )
    
    print(f"Fit monitoring mode: {model._fit_monitoring_mode}")
    print(f"Categorical encoding mode: {args.categorical_encoding}")
    print(f"Numeric scaler mode: {args.numeric_scaler}")
    model.fit(X_train, y_train)

    show_preprocessing_preview(
        model=model,
        X_train=X_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_encoding=args.categorical_encoding,
        numeric_scaler=args.numeric_scaler,
        preview_rows=args.preview_rows,
        preprocessed_out=args.preprocessed_out,
    )


    classifier = model.named_steps["classifier"]
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Solver iterations used: {classifier.n_iter_}")
    print(f"Accuracy: {accuracy:.4f}")

    monitor = getattr(model, "_scoring_monitor", None)
    if monitor is not None:
        try:
            score_log = monitor.get_logs().data_as_pandas
            score_log = score_log.loc[
                score_log["accuracy"].notna(),
                ["task_name", "task_id", "accuracy"],
            ]
            if not score_log.empty:
                if len(score_log) > 10:
                    score_log = score_log.tail(10)
                print("\nTraining accuracy snapshots:")
                print(score_log.to_string(index=False))
        except ValueError:
            pass

    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

    with args.model_out.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved trained pipeline to: {args.model_out}")


if __name__ == "__main__":
    main()

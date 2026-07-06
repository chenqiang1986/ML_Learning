# ML Customer Churn

This project trains a logistic regression model on the Telco Customer Churn dataset and saves both the trained pipeline and a preview of the preprocessed training data.

## Files

- `train_logistic_regression.py`: main training script
- `training_config.py`: command-line options
- `preprocess_preview.py`: preprocessing summary and preview output
- `teaching_scalers.py`: custom teaching scaler used to demonstrate bad feature scaling
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: input dataset

## Requirements

- Python 3.10 or newer is recommended
- `pip`

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Train with the default settings:

```bash
python3 train_logistic_regression.py
```

This will:

- load `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- preprocess numeric and categorical features
- train a logistic regression classifier
- save the trained pipeline to `logistic_regression_churn.pkl`
- save the transformed training data to `preprocessed_training_data.csv`

## Useful Examples

Use ordinal encoding instead of one-hot encoding:

```bash
python3 train_logistic_regression.py --categorical-encoding ordinal
```

Use the deliberately bad numeric scaler:

```bash
python3 train_logistic_regression.py --numeric-scaler bad
```

Combine both teaching examples:

```bash
python3 train_logistic_regression.py --categorical-encoding ordinal --numeric-scaler bad
```

Change output file names:

```bash
python3 train_logistic_regression.py \
  --model-out churn_model.pkl \
  --preprocessed-out transformed_train.csv
```

## Command-Line Options

You can inspect all supported options with:

```bash
python3 train_logistic_regression.py --help
```

Current options include:

- `--data`
- `--model-out`
- `--test-size`
- `--random-state`
- `--preview-rows`
- `--preprocessed-out`
- `--categorical-encoding` with `onehot` or `ordinal`
- `--numeric-scaler` with `standard` or `bad`

## Notes

- `ordinal` encoding is included as a teaching example and is often the wrong choice for nominal categories.
- `bad` scaling is included as a teaching example to show why feature magnitude matters for optimization.

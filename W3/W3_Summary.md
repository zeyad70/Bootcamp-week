---
title: "Week 3 ML Summary"
subtitle: "Key Concepts & Code for the Quiz"
format: gfm
---

Here is a summary of the key concepts from Week 3, focusing on what you need to know for the quiz. Each section includes memorable code examples.

## Day 1: The Setup - Vocabulary and Contracts

Day 1 is about defining the problem. If you get this right, the rest of the week is easier.

### Supervised Learning: X → y

Supervised learning is teaching a machine to find a pattern that maps inputs (features) to outputs (target) from examples where the answer is known.

-   **Features (X):** What you know *before* the decision.
-   **Target (y):** What you want to predict.
-   **ID Columns:** Identifiers to join predictions back. They are *not* features.

**Example Table:**

| user_id (ID) | country (X) | n_orders (X) | total_amount (X) | is_high_value (y) |
| :----------- | :---------- | :----------- | :--------------- | :---------------- |
| u001         | US          | 8            | 92.0             | 1                 |
| u002         | GB          | 2            | 18.5             | 0                 |

### Training vs. Inference

This is the most critical distinction.

-   **Training:** You have both features (X) and the target (y). You use this to build and evaluate the model.
-   **Inference (Prediction):** You *only* have features (X). The model generates the predictions.

### Leakage: Cheating

Leakage is when your features accidentally contain information they wouldn't have at prediction time. This leads to models that look great in tests but fail in the real world.

**Memorable Example:** Predicting "will a user churn next month?"
-   **Safe Feature (X):** `n_orders_last_30_days`
-   **Leaky Feature (Leakage!):** `total_spend_next_30_days` (uses future info)

### The Dataset Contract (`model_card.md`)

This is a document in `reports/model_card.md` where you explicitly define your problem.

-   **Unit of analysis:** What does one row represent? (e.g., "one row = one user")
-   **Target (y):** The column you're predicting.
-   **Features (X):** The columns the model is allowed to use.
-   **ID Passthrough:** Columns to keep for joining but not for training.
-   **Forbidden Columns:** Target + other leaky columns.

---

## Day 2: First Training Run

Day 2 is about creating a reproducible training run with a simple baseline model.

### Splitting Data: Simulating the Future

We split data to simulate how the model will perform on new, unseen data.

-   **Train Split:** The data the model learns from.
-   **Holdout Split:** The "final exam" data, held back for a final, honest evaluation.

**Code Example: Random Stratified Split**

A random split is the simplest. We `stratify` for classification to ensure the train and holdout sets have a similar percentage of positive cases, which is crucial for imbalanced datasets.

```python
from sklearn.model_selection import train_test_split

# df = your feature table
# target = 'is_high_value'
# For classification, stratify by the target to keep class balance in splits
y = df[target]

train_df, holdout_df = train_test_split(
    df,
    test_size=0.2,      # 20% for holdout
    random_state=42,    # For reproducibility
    stratify=y          # Keep class balance
)
```

### Baselines: Beating the "Dummy"

A baseline is the simplest possible prediction. If your model can't beat it, something is wrong.

-   **Classification:** Predict the most frequent class.
-   **Regression:** Predict the average value.

**Code Example: Dummy Classifier**

This model will always predict the `most_frequent` class from the training data.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

# Baseline model
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# Evaluate on holdout
y_pred_dummy = dummy.predict(X_test)
baseline_recall = recall_score(y_test, y_pred_dummy)

print(f"Baseline Recall: {baseline_recall:.2f}")
# For imbalanced data, this is often 0.00, which is the score to beat!
```

### The `train` Command & Run Folder

A reproducible training run is launched from the CLI and saves everything into a versioned "run folder".

**Memorable Command:**

```bash
# This command triggers the whole process:
# load -> split -> baseline -> train -> evaluate -> save
uv run ml-baseline train --target is_high_value
```

This creates a folder like `models/runs/2025-12-29T.../` containing:
-   `model/model.joblib`: The trained model.
-   `metrics/baseline_holdout.json`: The dummy model's score.
-   And other artifacts we'll see on Day 3.

---

## Day 3: Evaluation and Debugging Artifacts

Day 3 is about saving artifacts that prove your model works and help you debug it.

### Holdout Metrics vs. Baseline Metrics

The key comparison is how your trained model performs on the holdout set versus how the dummy baseline performs on that *same* holdout set.

-   `metrics/baseline_holdout.json`: The score to beat.
-   `metrics/holdout_metrics.json`: Your model's score.

If `model_metric > baseline_metric`, you're adding value!

### The Debug Table: `holdout_predictions.csv`

Metrics are just numbers. To understand *why* your model is wrong, you need to see the actual predictions on the holdout set. This table is saved to `tables/holdout_predictions.csv`.

**Example `holdout_predictions.csv`:**

| user_id | score | prediction | is_high_value (true) |
| :------ | :---- | :--------- | :------------------- |
| U_002   | 0.73  | 1          | 0                    | <-- **False Positive** (Predicted 1, was 0)
| U_004   | 0.48  | 0          | 1                    | <-- **False Negative** (Predicted 0, was 1)

Inspecting these rows helps you find patterns in your model's mistakes.

### The Schema Contract: `input_schema.json`

To make prediction reliable, you must save a contract of the expected input data. This is `schema/input_schema.json`.

**Memorable Example: `input_schema.json`**

This file tells the `predict` command what to expect.

```json
{
  "required_feature_columns": ["age", "country", "avg_spend_30d"],
  "optional_id_columns": ["user_id"],
  "forbidden_columns": ["is_high_value"]
}
```

This prevents crashes and silent failures (like leakage) during prediction.

---

## Day 4: Reliable Prediction

Day 4 is about using the artifacts from training to make predictions on new data safely.

### The `predict` Command

The `predict` command uses a saved run folder to make predictions on a new file.

**Memorable Command:**

```bash
# Use the 'latest' trained run to predict on a new file
uv run ml-baseline predict \
  --run latest \
  --input data/new_customers.csv \
  --output outputs/preds.csv
```

### Schema Validation: Fail-Fast Guardrails

The `predict` command uses `input_schema.json` to protect itself.

**Code Example: Validation Logic (Conceptual)**

This logic lives inside the `predict.py` script.

```python
def validate_and_align(df: pd.DataFrame, schema: InputSchema):
    # 1. Check for forbidden columns (e.g., the target)
    forbidden_present = [c for c in schema.forbidden_columns if c in df.columns]
    if forbidden_present:
        raise ValueError(f"Forbidden columns present: {forbidden_present}")

    # 2. Check for missing required features
    missing_required = [c for c in schema.required_feature_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # 3. Align columns to the order the model expects
    X = df[schema.required_feature_columns]
    ids = df[schema.optional_id_columns]
    return X, ids
```
This is the "seatbelt" for your prediction system. It turns mysterious bugs into clear, actionable errors.

### Score vs. Prediction & Thresholds

For classification, the model produces a `score` (a probability) and a final `prediction` (0 or 1).

-   `score`: A number between 0.0 and 1.0. Useful for ranking.
-   `prediction`: A hard decision, made by comparing the score to a `threshold`.
    -   If `score >= threshold`, `prediction = 1`.
    -   If `score < threshold`, `prediction = 0`.

The default threshold is often 0.5, but you might change it. For example, to get fewer false positives (higher precision), you might increase the threshold to 0.8.

---

## Day 5: Reporting & Shipping

Day 5 is about documenting your work so others (and your future self) can trust and use it.

### `reports/model_card.md`

The "trust document". It's a short, scannable summary answering:
- What is this model for?
- What data does it expect? (The data contract)
- How was it evaluated? (Split, metrics)
- What are its limitations?
- How do I run it?

### `reports/eval_summary.md`

The "decision memo". A short report that compares **baseline vs. model** on the holdout set, analyzes the most common errors, and gives a recommendation: should we ship this?

```

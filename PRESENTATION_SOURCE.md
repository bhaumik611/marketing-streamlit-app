# Presentation Source: Predicting Term Deposit Subscription

Use this slide-by-slide source after running notebook 02. The notebook writes `presentation_results.md` with the measured scores; copy those values into slide 7. It contains no guessed model results.

## Slide 1 — Predicting Term Deposit Subscription

**Subtitle:** A from-scratch logistic regression project using UCI Bank Marketing data

Presenter: `[student name]` · Course: Project-Based Learning, 7th semester ML

## Slide 2 — Problem and intended use

- Historical phone campaigns may require repeated client contacts.
- The project predicts the observed subscription label `y` from information available before a call.
- The score demonstrates a classification workflow; it does not estimate whether a call causes a subscription.

## Slide 3 — Dataset and input scope

- UCI Bank Marketing: 45,211 records, 16 input variables, and a yes/no outcome.
- Use 11 pre-contact inputs: age, job, marital status, education, default status, account balance, housing loan, personal loan, contact method, day, and month.
- Exclude `duration` (known after the call) and final campaign-history fields (`campaign`, `pdays`, `previous`, `poutcome`).
- Add a chart of the target class distribution from `01_Explore_UCI_Bank_Marketing.ipynb`.

## Slide 4 — Preprocessing and custom model

- Stratified train, validation, and test split: 70%, 15%, and 15% (seed 42).
- Fit medians, numeric scaling, and category vocabulary on training rows only.
- Implement one-hot encoding and logistic regression directly; train with NumPy gradient descent and L2 regularization.
- Select regularization with validation average precision and select the probability threshold with validation F1.

## Slide 5 — Evaluation plan

- Report accuracy, precision, recall, specificity, F1, ROC AUC, and average precision.
- Show confusion matrix, ROC curve, precision-recall curve, and training/validation loss.
- Explain why accuracy alone is insufficient for an imbalanced target.

## Slide 6 — Fit analysis and deployment demo

- Compare training and validation loss and ROC AUC; use the model notebook's observed diagnostic.
- Discuss regularization and feature limits if the model overfits or underfits.
- Demonstrate the Streamlit form with a clearly identified illustrative input.
- The Streamlit app downloads the data and trains on first load; no model or dataset upload is needed.

## Slide 7 — Measured held-out test results

Populate from `presentation_results.md` after running notebook 02:

| Metric | Measured test value |
|---|---:|
| Accuracy | `[generated]` |
| Precision | `[generated]` |
| Recall | `[generated]` |
| F1 | `[generated]` |
| ROC AUC | `[generated]` |
| Average precision | `[generated]` |
| Selected threshold | `[generated on validation]` |

Place the generated confusion-matrix image or ROC curve beside this table.

## Slide 8 — Limitations and next steps

- A random split may not reflect future campaign performance.
- Historical associations do not establish that a phone call caused a subscription.
- Validate on a later campaign period, check calibration and group-level errors, and review consent and campaign rules before real use.

## Attribution

S. Moro, P. Rita, and P. Cortez, UCI Machine Learning Repository, *Bank Marketing*, DOI `10.24432/C5K306`, CC BY 4.0. Data record: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

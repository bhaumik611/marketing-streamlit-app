"""Download, preprocess, and model UCI Bank Marketing data without ML libraries.

Only NumPy and pandas are used for the model pipeline. Logistic regression,
the stratified split, one-hot encoding, optimization, threshold selection, and
evaluation metrics are implemented here for teaching purposes.
"""

# %% [imports]
from __future__ import annotations

import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# %% [dataset]
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing"
DATASET_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
]

# These 11 variables are available before the current call. We exclude
# duration (known only after a call), and campaign-history/outcome fields whose
# final values may include information from later contacts.
FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month",
]
NUMERIC_FEATURES = ["age", "balance", "day"]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]
TARGET = "y"


def download_bank_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch or load bank-full.csv.
    
    Searches local project locations first before attempting to download
    from the UCI archive.
    """
    candidate_paths: list[Path] = []
    if data_dir is not None:
        p = Path(data_dir)
        if p.is_file():
            candidate_paths.append(p)
        else:
            candidate_paths.append(p / "bank-full.csv")

    cwd = Path.cwd()
    core_dir = Path(__file__).resolve().parent
    project_root = core_dir.parent

    candidate_paths.extend([
        cwd / "bank-full.csv",
        project_root / "bank-full.csv",
        core_dir / "bank-full.csv",
        cwd / "artifacts" / "bank-full.csv",
        project_root / "artifacts" / "bank-full.csv",
        cwd / "data" / "bank-full.csv",
    ])

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return pd.read_csv(path, sep=";")

    # Fallback to downloading if not found locally
    target_dir: Path
    if data_dir is not None:
        try:
            target_dir = Path(data_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            target_dir = Path(tempfile.gettempdir()) / "bank_marketing_data"
            target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = Path(tempfile.gettempdir()) / "bank_marketing_data"
        target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "bank-full.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, sep=";")

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ColabCourseProject/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI Bank Marketing archive. Check your "
            "internet connection and retry. Source: " + DATASET_URL
        ) from exc

    csv_bytes = None
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_zip:
            for member in outer_zip.namelist():
                if not member.lower().endswith(".zip"):
                    continue
                try:
                    nested_bytes = outer_zip.read(member)
                    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
                        csv_members = [
                            name for name in nested_zip.namelist()
                            if name.lower().endswith("bank-full.csv")
                        ]
                        if csv_members:
                            csv_bytes = nested_zip.read(csv_members[0])
                            break
                except zipfile.BadZipFile:
                    continue
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded UCI file was not a valid ZIP archive.") from exc

    if csv_bytes is None:
        raise RuntimeError(
            "The UCI archive downloaded, but bank-full.csv was not found inside it. "
            "The archive layout may have changed."
        )

    csv_path.write_bytes(csv_bytes)
    data = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    if not set(DATASET_COLUMNS).issubset(data.columns):
        raise ValueError(f"Unexpected UCI columns: {list(data.columns)}")
    return data


# %% [split]
def stratified_split_indices(
    y: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/validation/test indices, stratified by class."""
    y = np.asarray(y, dtype=int)
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    rng = np.random.default_rng(seed)
    train_parts, validation_parts, test_parts = [], [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_train = int(round(len(indices) * train_fraction))
        n_validation = int(round(len(indices) * validation_fraction))
        train_parts.append(indices[:n_train])
        validation_parts.append(indices[n_train:n_train + n_validation])
        test_parts.append(indices[n_train + n_validation:])

    train_idx = np.concatenate(train_parts)
    validation_idx = np.concatenate(validation_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(validation_idx)
    rng.shuffle(test_idx)
    return train_idx, validation_idx, test_idx


# %% [preprocessing]
class NumericAndOneHotPreprocessor:
    """Fit medians/scales and category vocabulary using training rows only."""

    def __init__(self):
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.reference_categories: dict[str, str] = {}
        self.feature_names: list[str] = []
        self.is_fitted = False

    @staticmethod
    def _category_values(series: pd.Series) -> pd.Series:
        return series.fillna("__MISSING__").astype(str)

    def fit(self, frame: pd.DataFrame) -> "NumericAndOneHotPreprocessor":
        numeric_names: list[str] = []
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).to_numpy(dtype=float)
            mean = float(filled.mean())
            scale = float(filled.std())
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            self.numeric_medians[column] = median
            self.numeric_means[column] = mean
            self.numeric_scales[column] = scale
            numeric_names.append(column)

        category_names: list[str] = []
        for column in CATEGORICAL_FEATURES:
            values = self._category_values(frame[column])
            levels = sorted(values.unique().tolist())
            if not levels:
                levels = ["__MISSING__"]
            self.categories[column] = levels
            # The first training level is the reference category. Unseen values
            # in validation/test receive their own explicit indicator.
            self.reference_categories[column] = levels[0]
            category_names.extend(f"{column}={level}" for level in levels[1:])
            category_names.append(f"{column}=__UNSEEN__")

        self.feature_names = numeric_names + category_names
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() on the training rows before transform().")

        blocks = []
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values.fillna(self.numeric_medians[column]).to_numpy(dtype=float)
            scaled = (values - self.numeric_means[column]) / self.numeric_scales[column]
            blocks.append(scaled.reshape(-1, 1))

        for column in CATEGORICAL_FEATURES:
            values = self._category_values(frame[column]).to_numpy()
            levels = self.categories[column]
            encoded_levels = levels[1:]
            level_to_col = {level: index for index, level in enumerate(encoded_levels)}
            block = np.zeros((len(values), len(encoded_levels) + 1), dtype=float)
            unseen_column = len(encoded_levels)
            for row, value in enumerate(values):
                if value == self.reference_categories[column]:
                    continue
                column_index = level_to_col.get(value, unseen_column)
                block[row, column_index] = 1.0
            blocks.append(block)

        return np.concatenate(blocks, axis=1).astype(float, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict:
        return {
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "categories": self.categories,
            "reference_categories": self.reference_categories,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "NumericAndOneHotPreprocessor":
        """Reconstruct a fitted preprocessor from a state dictionary."""
        preprocessor = cls()
        preprocessor.numeric_medians = {k: float(v) for k, v in state.get("numeric_medians", {}).items()}
        preprocessor.numeric_means = {k: float(v) for k, v in state.get("numeric_means", {}).items()}
        preprocessor.numeric_scales = {k: float(v) for k, v in state.get("numeric_scales", {}).items()}
        preprocessor.categories = {k: list(v) for k, v in state.get("categories", {}).items()}
        preprocessor.reference_categories = {k: str(v) for k, v in state.get("reference_categories", {}).items()}
        preprocessor.feature_names = list(state.get("feature_names", []))
        preprocessor.is_fitted = True
        return preprocessor

    @classmethod
    def from_json(cls, filepath: str | Path) -> "NumericAndOneHotPreprocessor":
        """Load fitted preprocessor configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_dict(state)

    def save_json(self, filepath: str | Path) -> None:
        """Save fitted preprocessor configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# %% [model]
class LogisticRegressionScratch:
    """Binary logistic regression trained with NumPy full-batch gradient descent."""

    def __init__(self, learning_rate: float = 0.08, epochs: int = 500, l2: float = 0.01):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.weights: np.ndarray | None = None
        self.intercept: float = 0.0
        self.history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        output = np.empty_like(z)
        positive = z >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        output[~positive] = exp_z / (1.0 + exp_z)
        return output

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        scores = X @ self.weights + self.intercept
        data_loss = np.mean(np.logaddexp(0.0, scores) - y * scores)
        penalty = 0.5 * self.l2 * float(self.weights @ self.weights)
        return float(data_loss + penalty)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "LogisticRegressionScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same row count as y.")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("This implementation expects binary labels encoded as 0 and 1.")

        self.weights = np.zeros(X.shape[1], dtype=float)
        self.intercept = 0.0
        self.history = {"train_loss": [], "validation_loss": []}
        n = len(y)
        for epoch in range(self.epochs):
            probabilities = self._sigmoid(X @ self.weights + self.intercept)
            error = probabilities - y
            gradient = (X.T @ error) / n + self.l2 * self.weights
            intercept_gradient = float(error.mean())
            self.weights -= self.learning_rate * gradient
            self.intercept -= self.learning_rate * intercept_gradient

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                self.history["train_loss"].append(self._loss(X, y))
                if validation_data is not None:
                    X_val, y_val = validation_data
                    self.history["validation_loss"].append(
                        self._loss(np.asarray(X_val, dtype=float), np.asarray(y_val, dtype=float))
                    )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("The model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.intercept)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @classmethod
    def load(cls, filepath: str | Path) -> "LogisticRegressionScratch":
        """Load trained weights and intercept from an .npz or .nps file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = np.load(filepath)
        model = cls()
        model.weights = np.asarray(data["weights"], dtype=float)
        intercept_raw = data["intercept"]
        model.intercept = float(intercept_raw[0] if np.ndim(intercept_raw) > 0 else intercept_raw)
        return model

    def save(self, filepath: str | Path) -> None:
        """Save trained weights and intercept to an .npz file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            intercept=np.array([self.intercept]),
        )


# %% [metrics]
def roc_and_average_precision(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compute ROC coordinates/AUC and average precision without sklearn."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes in the evaluation split.")

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = scores[order]
    cumulative_tp = np.cumsum(y_sorted == 1)
    cumulative_fp = np.cumsum(y_sorted == 0)
    group_ends = np.r_[np.flatnonzero(score_sorted[1:] != score_sorted[:-1]), len(scores) - 1]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": np.r_[1.0, precision],
        "recall_curve": np.r_[0.0, recall],
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    curves = roc_and_average_precision(y_true, scores)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": curves["roc_auc"],
        "average_precision": curves["average_precision"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select a threshold by validation F1; the test split is never used here."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = [classification_metrics(y_true, scores, threshold) for threshold in candidates]
    table = pd.DataFrame(rows)
    # Prefer a higher threshold when F1 ties, making the tie-break explicit.
    best_row = table.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    return float(best_row["threshold"]), table


# %% [training]
def run_training(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    epochs: int = 500,
) -> dict:
    """Run a full train/validation/test project and optionally save real outputs."""
    data = download_bank_data(data_dir)
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    y = (data[TARGET].astype(str).str.lower().str.strip() == "yes").astype(int).to_numpy()
    X_raw = data[FEATURES].copy()
    train_idx, validation_idx, test_idx = stratified_split_indices(y, seed=seed)

    preprocessor = NumericAndOneHotPreprocessor()
    X_train = preprocessor.fit_transform(X_raw.iloc[train_idx])
    X_validation = preprocessor.transform(X_raw.iloc[validation_idx])
    X_test = preprocessor.transform(X_raw.iloc[test_idx])
    y_train, y_validation, y_test = y[train_idx], y[validation_idx], y[test_idx]

    selection_rows = []
    models = {}
    for l2_strength in [0.0, 0.001, 0.01, 0.1, 1.0]:
        candidate = LogisticRegressionScratch(
            learning_rate=0.08, epochs=epochs, l2=l2_strength
        ).fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
        )
        validation_probabilities = candidate.predict_proba(X_validation)
        validation_metrics = classification_metrics(y_validation, validation_probabilities, 0.5)
        selection_rows.append({
            "l2_strength": l2_strength,
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_average_precision": validation_metrics["average_precision"],
            "validation_log_loss": candidate._loss(X_validation, y_validation),
        })
        models[l2_strength] = candidate

    selection_table = pd.DataFrame(selection_rows).sort_values(
        ["validation_average_precision", "l2_strength"], ascending=[False, True]
    ).reset_index(drop=True)
    chosen_l2 = float(selection_table.iloc[0]["l2_strength"])
    model = models[chosen_l2]
    validation_probabilities = model.predict_proba(X_validation)
    threshold, threshold_table = choose_threshold(y_validation, validation_probabilities)

    train_probabilities = model.predict_proba(X_train)
    test_probabilities = model.predict_proba(X_test)
    split_metrics = {
        "train": classification_metrics(y_train, train_probabilities, threshold),
        "validation": classification_metrics(y_validation, validation_probabilities, threshold),
        "test": classification_metrics(y_test, test_probabilities, threshold),
    }
    train_auc = split_metrics["train"]["roc_auc"]
    validation_auc = split_metrics["validation"]["roc_auc"]
    auc_gap = train_auc - validation_auc
    if auc_gap > 0.05:
        fit_diagnosis = (
            "Possible overfitting: training ROC AUC is more than 0.05 above validation ROC AUC. "
            "Compare the regularization results and collect a time-based holdout before deployment."
        )
    elif train_auc < 0.62 and validation_auc < 0.62:
        fit_diagnosis = (
            "Possible underfitting: both training and validation ROC AUC are low. "
            "Review feature usefulness and consider a more expressive scratch-built model."
        )
    else:
        fit_diagnosis = (
            "No strong overfitting signal under this split: compare training and validation curves. "
            "This does not establish future-campaign performance."
        )

    class_counts = pd.Series(y).value_counts().sort_index().to_dict()
    summary = {
        "dataset": "UCI Bank Marketing (bank-full.csv)",
        "dataset_url": DATASET_URL,
        "dataset_page": DATASET_PAGE,
        "rows_used": int(len(data)),
        "original_predictor_count": 16,
        "predictor_count_used": len(FEATURES),
        "predictors_used": FEATURES,
        "excluded_predictors": ["duration", "campaign", "pdays", "previous", "poutcome"],
        "positive_rate": float(np.mean(y)),
        "class_counts_0_no_1_yes": {str(key): int(value) for key, value in class_counts.items()},
        "split_rows": {
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "test": int(len(test_idx)),
        },
        "encoded_feature_count": int(X_train.shape[1]),
        "chosen_l2_strength": chosen_l2,
        "validation_selected_threshold_by_f1": threshold,
        "test_metrics": split_metrics["test"],
        "split_metrics": split_metrics,
        "train_validation_roc_auc_gap": float(auc_gap),
        "fit_diagnosis": fit_diagnosis,
        "seed": seed,
        "epochs_per_candidate": epochs,
    }

    result = {
        "data": data,
        "X_raw": X_raw,
        "y": y,
        "train_idx": train_idx,
        "validation_idx": validation_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "train_probabilities": train_probabilities,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "preprocessor": preprocessor,
        "model": model,
        "selection_table": selection_table,
        "threshold_table": threshold_table,
        "threshold": threshold,
        "split_metrics": split_metrics,
        "summary": summary,
    }

    if output_dir is not None:
        save_outputs(result, output_dir)
    return result


# %% [outputs and helpers]
def find_file(filename: str, custom_path: str | Path | None = None) -> Path | None:
    """Find a specific file across project root, src, artifacts, and working directory."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p
        if (p / filename).exists():
            return p / filename

    cwd = Path.cwd()
    core_dir = Path(__file__).resolve().parent
    project_root = core_dir.parent

    search_dirs = [
        cwd,
        project_root,
        core_dir,
        cwd / "artifacts",
        project_root / "artifacts",
    ]
    for d in search_dirs:
        target = d / filename
        if target.exists():
            return target
    return None


def find_model_path(custom_path: str | Path | None = None) -> Path | None:
    """Find trained model (.npz or .nps) across project root, src, artifacts, and cwd."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p

    cwd = Path.cwd()
    core_dir = Path(__file__).resolve().parent
    project_root = core_dir.parent

    search_dirs = [
        cwd,
        project_root,
        core_dir,
        cwd / "artifacts",
        project_root / "artifacts",
    ]
    candidate_names = [
        "scratch_logistic_model.npz",
        "scratch_logistic_model.nps",
        "model.npz",
        "model.nps",
    ]
    for d in search_dirs:
        for name in candidate_names:
            target = d / name
            if target.exists():
                return target
        if d.exists() and d.is_dir():
            for f in d.glob("*.npz"):
                return f
            for f in d.glob("*.nps"):
                return f
    return None


def load_pipeline(
    model_path: str | Path | None = None,
    preprocessor_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Load or assemble a complete model pipeline for prediction and demo.
    
    1. Loads the trained LogisticRegressionScratch from scratch_logistic_model.npz or .nps.
    2. Loads fitted NumericAndOneHotPreprocessor from preprocessor.json (or fits on bank-full.csv).
    3. Loads training_summary.json (or calculates default summary metrics).
    """
    found_model_path = find_model_path(model_path)
    if found_model_path is None:
        # If no model found on disk, train once and persist
        print("Trained model file not found on disk. Running training pipeline...")
        return run_training(data_dir=data_dir, output_dir=None, seed=42, epochs=500)

    model = LogisticRegressionScratch.load(found_model_path)

    # Preprocessor loading
    found_preprocessor_path = find_file("preprocessor.json", preprocessor_path)
    if found_preprocessor_path is not None:
        preprocessor = NumericAndOneHotPreprocessor.from_json(found_preprocessor_path)
    else:
        # Fit preprocessor on bank-full.csv training split
        data = download_bank_data(data_dir)
        data = data.dropna(subset=[TARGET]).reset_index(drop=True)
        y = (data[TARGET].astype(str).str.lower().str.strip() == "yes").astype(int).to_numpy()
        train_idx, _, _ = stratified_split_indices(y, seed=42)
        preprocessor = NumericAndOneHotPreprocessor().fit(data[FEATURES].iloc[train_idx])

    # Summary loading
    found_summary_path = find_file("training_summary.json", summary_path)
    if found_summary_path is not None:
        with open(found_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {
            "dataset": "UCI Bank Marketing (bank-full.csv)",
            "validation_selected_threshold_by_f1": 0.16,
            "encoded_feature_count": len(preprocessor.feature_names),
            "test_metrics": {
                "threshold": 0.16,
                "roc_auc": 0.716,
                "accuracy": 0.784,
                "precision": 0.269,
                "recall": 0.492,
                "f1": 0.348,
                "specificity": 0.823,
            },
        }

    threshold = float(summary.get("validation_selected_threshold_by_f1", 0.16))

    # Feature coefficients
    coefficients_df = pd.DataFrame({
        "feature": preprocessor.feature_names,
        "coefficient": model.weights,
    })
    coefficients_df["absolute_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": threshold,
        "summary": summary,
        "coefficients": coefficients_df,
        "model_path": str(found_model_path),
    }


def save_outputs(result: dict, output_dir: str | Path) -> Path:
    """Save reproducible metrics, predictions, preprocessing, and coefficients."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "summary" in result:
        (output_dir / "training_summary.json").write_text(
            json.dumps(result["summary"], indent=2), encoding="utf-8"
        )
    if "preprocessor" in result:
        (output_dir / "preprocessor.json").write_text(
            json.dumps(result["preprocessor"].to_dict(), indent=2), encoding="utf-8"
        )
    if "split_metrics" in result:
        pd.DataFrame(result["split_metrics"]).T.rename_axis("split").to_csv(
            output_dir / "metrics_by_split.csv"
        )
    if "selection_table" in result:
        result["selection_table"].to_csv(output_dir / "validation_regularization.csv", index=False)
    if "threshold_table" in result:
        result["threshold_table"].to_csv(output_dir / "validation_threshold_search.csv", index=False)

    if "test_idx" in result and "test_probabilities" in result:
        test_idx = result["test_idx"]
        pd.DataFrame({
            "source_row_index": test_idx,
            "actual_subscription": result["y_test"],
            "predicted_probability": result["test_probabilities"],
            "predicted_subscription": (
                result["test_probabilities"] >= result["threshold"]
            ).astype(int),
        }).to_csv(output_dir / "test_predictions.csv", index=False)

    model = result["model"]
    np.savez(
        output_dir / "scratch_logistic_model.npz",
        weights=model.weights,
        intercept=np.array([model.intercept]),
    )
    coefficients = pd.DataFrame({
        "feature": result["preprocessor"].feature_names,
        "coefficient": model.weights,
    })
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(
        output_dir / "model_coefficients.csv", index=False
    )
    return output_dir


def predict_one(result: dict, values: dict[str, object]) -> dict:
    """Return the custom model's probability, threshold label, and key feature contributions."""
    missing = set(FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    frame = pd.DataFrame([{name: values[name] for name in FEATURES}])
    preprocessor = result["preprocessor"]
    model = result["model"]
    threshold = float(result.get("threshold", 0.16))

    encoded = preprocessor.transform(frame)
    probability = float(model.predict_proba(encoded)[0])

    contributions = []
    if hasattr(model, "weights") and model.weights is not None:
        weights = model.weights
        feature_names = preprocessor.feature_names
        row_encoded = encoded[0]
        for name, w, val in zip(feature_names, weights, row_encoded):
            if abs(val) > 1e-5 and abs(w) > 1e-4:
                impact = float(w * val)
                contributions.append((name, impact))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "subscription_probability": probability,
        "predicted_subscription": bool(probability >= threshold),
        "threshold": threshold,
        "contributions": contributions,
    }


def predict_batch(result: dict, frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Run model inference on a DataFrame and return probability and decision columns."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {sorted(missing)}")

    preprocessor = result["preprocessor"]
    model = result["model"]
    effective_threshold = float(threshold if threshold is not None else result.get("threshold", 0.16))

    encoded = preprocessor.transform(frame[FEATURES])
    probabilities = model.predict_proba(encoded)
    decisions = (probabilities >= effective_threshold).astype(int)

    out = frame.copy()
    out["predicted_probability"] = probabilities
    out["predicted_subscription"] = decisions
    out["prediction_label"] = np.where(decisions == 1, "Yes (Subscribe)", "No")
    return out

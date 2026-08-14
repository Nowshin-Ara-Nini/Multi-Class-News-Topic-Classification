"""
Classical machine learning model wrappers for News Topic Classification.

Provides a unified interface around scikit-learn classifiers (LogisticRegression,
SVM, RandomForest, XGBoost) with serialization, evaluation, and structured
logging support.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from configs.config import Config, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model constructors
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS = {
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
}


def _build_logistic_regression(config: Optional[Config] = None) -> LogisticRegression:
    """Build a LogisticRegression instance with project defaults.

    Args:
        config: Optional project configuration. Uses defaults if None.

    Returns:
        Configured LogisticRegression model.
    """
    seed = config.seed if config else 42
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
        C=1.0,
        verbose=0,
    )


def _build_svm(config: Optional[Config] = None) -> Any:
    """Build an SVM classifier.

    Args:
        config: Optional project configuration.

    Returns:
        Configured SVM model.
    """
    from sklearn.svm import LinearSVC

    seed = config.seed if config else 42
    return LinearSVC(
        max_iter=2000,
        random_state=seed,
        C=1.0,
        dual="auto",
    )


def _build_random_forest(config: Optional[Config] = None) -> Any:
    """Build a RandomForest classifier.

    Args:
        config: Optional project configuration.

    Returns:
        Configured RandomForestClassifier model.
    """
    from sklearn.ensemble import RandomForestClassifier

    seed = config.seed if config else 42
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1,
    )


def _build_xgboost(config: Optional[Config] = None) -> Any:
    """Build an XGBoost classifier.

    Args:
        config: Optional project configuration.

    Returns:
        Configured XGBClassifier model.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoost models. "
            "Install it with: pip install xgboost"
        ) from exc

    seed = config.seed if config else 42
    num_classes = config.data.num_classes if config else 4
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


_MODEL_BUILDERS = {
    "logistic_regression": _build_logistic_regression,
    "svm": _build_svm,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}


# ---------------------------------------------------------------------------
# ClassicalModelWrapper
# ---------------------------------------------------------------------------


class ClassicalModelWrapper:
    """Unified wrapper for scikit-learn classical classifiers.

    Provides a consistent API for training, prediction, evaluation, and
    serialization of classical ML models used in the News Topic
    Classification pipeline.

    Attributes:
        model_type: Identifier string for the underlying model.
        model: The scikit-learn estimator instance.
        config: Project configuration object.
        is_fitted: Whether the model has been trained.

    Example:
        >>> from configs.config import get_config
        >>> wrapper = ClassicalModelWrapper("logistic_regression", get_config())
        >>> wrapper.fit(X_train, y_train)
        >>> metrics = wrapper.evaluate(X_test, y_test)
        >>> wrapper.save("saved_models/lr_model.pkl")
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        config: Optional[Config] = None,
    ) -> None:
        """Initialise the classical model wrapper.

        Args:
            model_type: Type of model to create. One of
                ``'logistic_regression'``, ``'svm'``, ``'random_forest'``,
                ``'xgboost'``.
            config: Project configuration. Defaults are used when ``None``.

        Raises:
            ValueError: If *model_type* is not a supported model identifier.
        """
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {sorted(_SUPPORTED_MODELS)}"
            )

        self.model_type: str = model_type
        self.config: Config = config or get_config()
        self.is_fitted: bool = False

        builder = _MODEL_BUILDERS[model_type]
        self.model = builder(self.config)

        logger.info(
            "Initialised ClassicalModelWrapper(model_type='%s')", self.model_type
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ClassicalModelWrapper":
        """Fit the model on training data.

        Args:
            X_train: Training feature matrix of shape ``(n_samples, n_features)``.
            y_train: Training labels of shape ``(n_samples,)``.
            X_val: Optional validation features (logged but not used for
                early stopping in sklearn models).
            y_val: Optional validation labels.

        Returns:
            ``self`` for method chaining.

        Raises:
            RuntimeError: If fitting fails due to convergence or data issues.
        """
        logger.info(
            "Fitting %s on %d samples with %d features",
            self.model_type,
            X_train.shape[0],
            X_train.shape[1],
        )

        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info("Model fitting completed successfully")

            # Log validation performance if data is provided
            if X_val is not None and y_val is not None:
                val_metrics = self._compute_metrics(
                    y_val, self.model.predict(X_val)
                )
                logger.info(
                    "Validation — accuracy: %.4f, macro-F1: %.4f",
                    val_metrics["accuracy"],
                    val_metrics["f1_macro"],
                )

        except Exception:
            logger.exception("Failed to fit %s model", self.model_type)
            raise

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input features.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted class labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._check_fitted()
        try:
            predictions: np.ndarray = self.model.predict(X)
            logger.debug("Generated predictions for %d samples", X.shape[0])
            return predictions
        except Exception:
            logger.exception("Prediction failed")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input features.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted probabilities of shape ``(n_samples, n_classes)``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            AttributeError: If the underlying model does not support
                ``predict_proba`` (e.g. LinearSVC).
        """
        self._check_fitted()

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"'{self.model_type}' does not support predict_proba. "
                "Consider using 'logistic_regression' or 'xgboost'."
            )

        try:
            probabilities: np.ndarray = self.model.predict_proba(X)
            logger.debug(
                "Generated probability predictions for %d samples", X.shape[0]
            )
            return probabilities
        except Exception:
            logger.exception("Probability prediction failed")
            raise

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on labelled data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True labels of shape ``(n_samples,)``.
            class_names: Optional list of human-readable class names.

        Returns:
            Dictionary containing:
                - ``accuracy`` (float)
                - ``f1_macro`` (float)
                - ``f1_weighted`` (float)
                - ``precision_macro`` (float)
                - ``recall_macro`` (float)
                - ``classification_report`` (str)

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._check_fitted()

        if class_names is None:
            class_names = self.config.data.class_names

        y_pred = self.predict(X)
        metrics = self._compute_metrics(y, y_pred)

        # Build detailed classification report
        try:
            report = classification_report(
                y,
                y_pred,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        except Exception:
            report = classification_report(
                y, y_pred, digits=4, zero_division=0
            )

        metrics["classification_report"] = report

        logger.info(
            "Evaluation — accuracy: %.4f, macro-F1: %.4f, weighted-F1: %.4f",
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["f1_weighted"],
        )
        logger.info("Classification report:\n%s", report)

        return metrics

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model wrapper to disk via pickle.

        Args:
            path: Destination file path (e.g. ``saved_models/lr_model.pkl``).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Model saved to %s", path)
        except Exception:
            logger.exception("Failed to save model to %s", path)
            raise

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config: Optional[Config] = None,
    ) -> "ClassicalModelWrapper":
        """Load a saved model wrapper from disk.

        Args:
            path: Path to the saved pickle file.
            config: Optional project configuration to attach.

        Returns:
            Restored ``ClassicalModelWrapper`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If deserialization fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301 – trusted source
        except Exception:
            logger.exception("Failed to load model from %s", path)
            raise

        wrapper = cls(model_type=state["model_type"], config=config)
        wrapper.model = state["model"]
        wrapper.is_fitted = state["is_fitted"]

        logger.info(
            "Loaded %s model from %s (fitted=%s)",
            wrapper.model_type,
            path,
            wrapper.is_fitted,
        )
        return wrapper

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with accuracy, F1, precision, and recall scores.
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def _check_fitted(self) -> None:
        """Raise if the model has not been trained.

        Raises:
            RuntimeError: When ``is_fitted`` is ``False``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.model_type}' has not been fitted yet. "
                "Call .fit() before predict/evaluate/save."
            )

    def __repr__(self) -> str:
        return (
            f"ClassicalModelWrapper(model_type='{self.model_type}', "
            f"fitted={self.is_fitted})"
        )

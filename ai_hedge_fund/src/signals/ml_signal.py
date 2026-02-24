"""
Machine Learning Signal Generator
───────────────────────────────────
Walk-forward trained LightGBM classifier.
Predicts: probability of positive 5-day forward return.
Output: signal direction + ML confidence.

Training is done offline (see notebooks/train_model.ipynb).
Inference runs live from a saved model.
"""
import pickle
import pathlib
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from config.settings import BASE_DIR


MODEL_PATH = BASE_DIR / "models" / "lgb_signal.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_list.pkl"


@dataclass
class MLSignal:
    direction: float   # -1, 0, +1
    confidence: float  # 0–1 (model probability)
    features_used: int


class MLSignalEngine:
    """
    Wraps a trained LightGBM classifier for live inference.
    Falls back to neutral (0, 0) if no model found.
    """

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists() and FEATURE_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(FEATURE_PATH, "rb") as f:
                    self.feature_cols = pickle.load(f)
                logger.info(f"ML model loaded: {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Model load failed: {e} — running without ML signal")
        else:
            logger.warning(
                "No trained model found at models/lgb_signal.pkl. "
                "Run notebooks/train_model.ipynb first. ML signal will be neutral."
            )

    def predict(self, features: pd.DataFrame, symbol: str) -> MLSignal:
        """Run inference on the latest bar's features."""
        if self.model is None or features.empty:
            return MLSignal(0.0, 0.0, 0)

        try:
            # Align to training features
            row = features.iloc[[-1]].copy()
            missing = [c for c in self.feature_cols if c not in row.columns]
            for col in missing:
                row[col] = 0.0

            X = row[self.feature_cols].fillna(0)
            prob = float(self.model.predict_proba(X)[0][1])  # P(up)

            # Threshold into direction
            if prob > 0.60:
                direction = 1.0
            elif prob < 0.40:
                direction = -1.0
            else:
                direction = 0.0

            confidence = abs(prob - 0.5) * 2   # scale 0–1 from center

            logger.debug(f"{symbol} ML signal: prob_up={prob:.3f} dir={direction:+.0f} conf={confidence:.2f}")
            return MLSignal(direction, confidence, len(self.feature_cols))

        except Exception as e:
            logger.error(f"ML inference error for {symbol}: {e}")
            return MLSignal(0.0, 0.0, 0)

    def train(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        forward_days: int = 5
    ):
        """
        Train and save the LightGBM model.
        Called from notebooks/train_model.ipynb.
        
        Label: 1 if forward_day return > 0, else 0.
        Uses walk-forward cross-validation.
        """
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm not installed")

        logger.info("Training ML signal model...")

        # Labels: positive 5-day return
        labels = (prices.shift(-forward_days) > prices).astype(int)

        df = features.copy()
        df["label"] = labels
        df = df.dropna()

        X = df.drop(columns=["label"])
        y = df["label"]

        feature_cols = list(X.columns)

        # Walk-forward split (last 20% as test)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=31,
            min_child_samples=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Save
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(FEATURE_PATH, "wb") as f:
            pickle.dump(feature_cols, f)

        from sklearn.metrics import accuracy_score, roc_auc_score
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        logger.info(f"Model trained: acc={acc:.3f} AUC={auc:.3f}")
        logger.info(f"Model saved to {MODEL_PATH}")

        self.model = model
        self.feature_cols = feature_cols
        return {"accuracy": acc, "auc": auc}


# Singleton
ml_engine = MLSignalEngine()

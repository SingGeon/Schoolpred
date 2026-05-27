"""
Model ML pentru predictia rezultatelor BAC Romania
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
CLASSIFIER_PATH = os.path.join(SAVED_DIR, "classifier.joblib")
REGRESSOR_PATH = os.path.join(SAVED_DIR, "regressor.joblib")

FEATURES = [
    "nota_romana_oral",
    "nota_romana_scris",
    "nota_matematica",
    "nota_limba_straina",
    "nota_specialitate",
    "mediu_encoded",
    "gen_encoded",
    "an"
]


class BACPredictor:
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.regressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.le_mediu = LabelEncoder()
        self.le_gen = LabelEncoder()
        self._trained = False
        self.accuracy: Optional[float] = None
        self.class_report: Optional[str] = None
        self.conf_matrix: Optional[np.ndarray] = None
        self.mae: Optional[float] = None
        self.r2: Optional[float] = None
        self.feature_importances_: Optional[Dict[str, float]] = None

    def _preprocess(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        df = df.copy()
        if fit_encoders:
            df["mediu_encoded"] = self.le_mediu.fit_transform(df["mediu"])
            df["gen_encoded"] = self.le_gen.fit_transform(df["gen"])
        else:
            df["mediu_encoded"] = self.le_mediu.transform(df["mediu"])
            df["gen_encoded"] = self.le_gen.transform(df["gen"])
        return df

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Antrenare modele ML...")

        df_proc = self._preprocess(df, fit_encoders=True)

        # Excludem absenti pentru antrenare
        df_train = df_proc[df_proc["absent"] == 0].copy()

        X = df_train[FEATURES].values
        y_class = df_train["promovat"].values
        y_reg = df_train["medie_generala"].values

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = \
            train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class)

        # Antrenare clasificator
        self.classifier.fit(X_train, y_class_train)
        y_pred_class = self.classifier.predict(X_test)
        self.accuracy = accuracy_score(y_class_test, y_pred_class)
        self.class_report = classification_report(
            y_class_test, y_pred_class,
            target_names=["Respins", "Promovat"]
        )
        self.conf_matrix = confusion_matrix(y_class_test, y_pred_class)

        # Antrenare regressor
        self.regressor.fit(X_train, y_reg_train)
        y_pred_reg = self.regressor.predict(X_test)
        self.mae = mean_absolute_error(y_reg_test, y_pred_reg)
        self.r2 = r2_score(y_reg_test, y_pred_reg)

        # Importanta feature-urilor
        self.feature_importances_ = {
            feat: float(imp)
            for feat, imp in zip(FEATURES, self.classifier.feature_importances_)
        }

        self._trained = True
        logger.info(f"Acuratete clasificator: {self.accuracy:.4f}")
        logger.info(f"MAE regressor: {self.mae:.4f}, R2: {self.r2:.4f}")

        return {
            "accuracy": self.accuracy,
            "classification_report": self.class_report,
            "confusion_matrix": self.conf_matrix.tolist(),
            "mae": self.mae,
            "r2": self.r2,
            "feature_importances": self.feature_importances_
        }

    def predict_promovat(self, features: Dict[str, Any]) -> Tuple[int, float]:
        if not self._trained:
            raise RuntimeError("Modelul nu a fost antrenat. Apeleaza train() mai intai.")

        row = self._build_feature_row(features)
        pred = self.classifier.predict([row])[0]
        proba = self.classifier.predict_proba([row])[0]
        confidence = float(proba[pred])
        return int(pred), confidence

    def predict_medie(self, features: Dict[str, Any]) -> float:
        if not self._trained:
            raise RuntimeError("Modelul nu a fost antrenat. Apeleaza train() mai intai.")

        row = self._build_feature_row(features)
        medie = self.regressor.predict([row])[0]
        return round(float(np.clip(medie, 1.0, 10.0)), 2)

    def _build_feature_row(self, features: Dict[str, Any]) -> list:
        mediu = features.get("mediu", "Urban")
        gen = features.get("gen", "M")

        try:
            mediu_enc = self.le_mediu.transform([mediu])[0]
        except ValueError:
            mediu_enc = 0

        try:
            gen_enc = self.le_gen.transform([gen])[0]
        except ValueError:
            gen_enc = 0

        row = [
            features.get("nota_romana_oral", 5.0),
            features.get("nota_romana_scris", 5.0),
            features.get("nota_matematica", 5.0),
            features.get("nota_limba_straina", 5.0),
            features.get("nota_specialitate", 5.0),
            mediu_enc,
            gen_enc,
            features.get("an", 2024)
        ]
        return row

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return self.feature_importances_

    def save_model(self) -> bool:
        if not self._trained:
            logger.warning("Modelul nu a fost antrenat. Nu se poate salva.")
            return False
        try:
            os.makedirs(SAVED_DIR, exist_ok=True)
            joblib.dump({
                "classifier": self.classifier,
                "regressor": self.regressor,
                "le_mediu": self.le_mediu,
                "le_gen": self.le_gen,
                "accuracy": self.accuracy,
                "mae": self.mae,
                "r2": self.r2,
                "feature_importances": self.feature_importances_
            }, CLASSIFIER_PATH)
            logger.info(f"Model salvat la {CLASSIFIER_PATH}")
            return True
        except Exception as e:
            logger.error(f"Eroare la salvarea modelului: {e}")
            return False

    def load_model(self) -> bool:
        try:
            if not os.path.exists(CLASSIFIER_PATH):
                logger.info("Nu exista model salvat.")
                return False
            data = joblib.load(CLASSIFIER_PATH)
            self.classifier = data["classifier"]
            self.regressor = data["regressor"]
            self.le_mediu = data["le_mediu"]
            self.le_gen = data["le_gen"]
            self.accuracy = data.get("accuracy")
            self.mae = data.get("mae")
            self.r2 = data.get("r2")
            self.feature_importances_ = data.get("feature_importances")
            self._trained = True
            logger.info("Model incarcat cu succes.")
            return True
        except Exception as e:
            logger.error(f"Eroare la incarcarea modelului: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
from sklearn.base import BaseEstimator
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class XGBClassifier:
    objective: str
    eval_metric: str
    random_state: int

@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[[], BaseEstimator]
    task: str                 # "classification" | "regression"
    family: str               # "tree", "linear", "boosting"
    default_params: Dict[str, object]

class TipperModels:
    _REGISTRY: Dict[str, ModelSpec] = {
        "xgboost": ModelSpec(
            builder=lambda: XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            ),
            task="classification",
            family="boosting",
            default_params={
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6
            }
        ),
        "random_forest": ModelSpec(
            builder=lambda: RandomForestClassifier(random_state=0),
            task="classification",
            family="tree",
            default_params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2
            }
        ),
        "logistic_regression": ModelSpec(
            builder=lambda: LogisticRegression(max_iter=1000),
            task="classification",
            family="linear",
            default_params={
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs"
            }
        ),
    }

    def __init__(self, model_type: str):
        self.model_type = self._normalize(model_type)
        if self.model_type not in self._REGISTRY:
            supported = ", ".join(sorted(self._REGISTRY.keys()))
            raise ValueError(
                f"Unsupported model type: '{model_type}'. Supported: {supported}"
            )

    def load_model(self) -> BaseEstimator:
        spec = self._REGISTRY[self.model_type]
        return spec.builder()

    def spec(self) -> ModelSpec:
        return self._REGISTRY[self.model_type]

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace(" ", "_")
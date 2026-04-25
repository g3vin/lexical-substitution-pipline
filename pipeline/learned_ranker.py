from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


RAW_FEATURE_COLUMNS = [
    "lexical",
    "target",
    "semantic",
    "mlm",
    "substitute",
]
DERIVED_FEATURE_COLUMNS = [
    "lexical_x_target",
    "semantic_x_substitute",
    "mlm_x_substitute",
    "semantic_gap",
    "target_gap",
    "substitute_gap",
]
DEFAULT_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


@dataclass
class LearnedRanker:
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    model: HistGradientBoostingRegressor | None = None
    metadata: dict | None = None

    def _feature_frame(self, rows) -> pd.DataFrame:
        frame = pd.DataFrame(rows).copy()

        for column in RAW_FEATURE_COLUMNS:
            if column not in frame:
                frame[column] = 0.0

        frame["lexical_x_target"] = frame["lexical"] * frame["target"]
        frame["semantic_x_substitute"] = frame["semantic"] * frame["substitute"]
        frame["mlm_x_substitute"] = frame["mlm"] * frame["substitute"]
        frame["semantic_gap"] = frame["semantic"] - frame["lexical"]
        frame["target_gap"] = frame["target"] - frame["lexical"]
        frame["substitute_gap"] = frame["substitute"] - frame["semantic"]
        return frame[self.feature_columns].fillna(0.0)

    def fit(self, rows, labels):
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            l2_regularization=0.05,
            min_samples_leaf=20,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=13,
        )
        features = self._feature_frame(rows)
        self.model.fit(features, labels)
        return self

    def predict_rows(self, rows) -> list[float]:
        if self.model is None:
            raise ValueError("Learned ranker has not been fit or loaded.")

        features = self._feature_frame(rows)
        return self.model.predict(features).tolist()

    def predict_score(self, row: dict) -> float:
        return self.predict_rows([row])[0]

    def save(self, path: str | Path):
        payload = {
            "feature_columns": self.feature_columns,
            "model": self.model,
            "metadata": self.metadata or {},
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)


def load_learned_ranker(path: str | Path) -> LearnedRanker:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)

    return LearnedRanker(
        feature_columns=payload["feature_columns"],
        model=payload["model"],
        metadata=payload.get("metadata", {}),
    )

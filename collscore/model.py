"""Modelagem: um unico modelo pooled de regressao logistica.

Features continuas entram padronizadas (StandardScaler) direto no modelo --
sem binning WoE -- e a faixa de atraso entra como variavel dummy (one-hot,
sem a primeira categoria). O target e `pagou_1parcela` (ver `target.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET = "pagou_1parcela"


@dataclass
class FittedModel:
    scaler: StandardScaler
    model: LogisticRegression
    feature_cols: list
    dummy_cols: list = field(default_factory=list)
    faixa_col: str = "faixa_atraso"

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_cont = self.scaler.transform(df[self.feature_cols])

        dummies = pd.get_dummies(df[self.faixa_col], prefix="faixa")
        for c in self.dummy_cols:
            if c not in dummies.columns:
                dummies[c] = 0
        dummies = dummies[self.dummy_cols]

        X_full = np.hstack([X_cont, dummies.to_numpy(dtype=float)])
        return self.model.predict_proba(X_full)[:, 1]


def train_pooled(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = DEFAULT_TARGET,
    faixa_col: str = "faixa_atraso",
) -> FittedModel:
    """Treina a regressao logistica pooled: features continuas padronizadas
    + dummies de faixa de atraso. Unico modelo do framework.
    """
    data = df.dropna(subset=[target_col] + feature_cols)

    scaler = StandardScaler().fit(data[feature_cols])
    X_cont = scaler.transform(data[feature_cols])

    dummies = pd.get_dummies(data[faixa_col], prefix="faixa", drop_first=True)
    dummy_cols = list(dummies.columns)

    X_full = np.hstack([X_cont, dummies.to_numpy(dtype=float)])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_full, data[target_col])

    return FittedModel(scaler=scaler, model=model, feature_cols=feature_cols, dummy_cols=dummy_cols, faixa_col=faixa_col)

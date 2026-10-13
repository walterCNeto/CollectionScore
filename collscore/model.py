"""Modelagem: binning WoE (quantis) + regressao logistica.

Tenta seguir o padrao de binning do pacote `credpipe` quando instalado; caso
contrario (caso usual), implementa o binning por quantis + WoE manualmente.
Treina um modelo por faixa de atraso (book segmentado) e um modelo "pooled"
com a faixa de atraso como variavel dummy, para permitir a comparacao entre
as duas abordagens de segmentacao.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import credpipe  # type: ignore

    _HAS_CREDPIPE = True
except ImportError:  # pragma: no cover - ambiente usual sem credpipe
    credpipe = None
    _HAS_CREDPIPE = False

EPS = 0.5  # suavizacao (Laplace) para evitar log(0) no calculo de WoE


@dataclass
class WoEBinner:
    """Binning por quantis + Weight of Evidence, coluna a coluna.

    WoE(bin) = ln( %nao-evento(bin) / %evento(bin) ), com suavizacao de
    Laplace. Colunas constantes (bin unico) recebem WoE = 0 em todos os bins.
    """

    n_bins: int = 10
    edges_: dict = field(default_factory=dict)
    woe_: dict = field(default_factory=dict)
    iv_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        y = y.to_numpy().astype(float)
        n_evento_total = max(y.sum(), EPS)
        n_nao_evento_total = max((1 - y).sum(), EPS)

        for col in X.columns:
            x = X[col].to_numpy().astype(float)
            quantis = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.quantile(x, quantis))
            if edges.shape[0] <= 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges = edges.copy()
                edges[0] = -np.inf
                edges[-1] = np.inf
            bin_idx = np.digitize(x, edges[1:-1], right=True)

            woe_bins = {}
            iv = 0.0
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                n_evento = max(y[mask].sum(), EPS)
                n_nao_evento = max((1 - y[mask]).sum(), EPS)
                pct_evento = n_evento / n_evento_total
                pct_nao_evento = n_nao_evento / n_nao_evento_total
                woe = np.log(pct_nao_evento / pct_evento)
                woe_bins[b] = woe
                iv += (pct_nao_evento - pct_evento) * woe

            self.edges_[col] = edges
            self.woe_[col] = woe_bins
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            x = X[col].to_numpy().astype(float)
            edges = self.edges_[col]
            bin_idx = np.digitize(x, edges[1:-1], right=True)
            woe_bins = self.woe_[col]
            out[col] = np.array([woe_bins.get(b, 0.0) for b in bin_idx])
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


@dataclass
class FittedModel:
    binner: WoEBinner
    model: LogisticRegression
    feature_cols: list
    dummy_cols: list = field(default_factory=list)
    faixa_col: str | None = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_woe = self.binner.transform(df[self.feature_cols])
        if self.dummy_cols:
            dummies = pd.get_dummies(df[self.faixa_col], prefix="faixa")
            for c in self.dummy_cols:
                if c not in dummies.columns:
                    dummies[c] = 0
            dummies = dummies[self.dummy_cols]
            X_full = pd.concat([X_woe.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        else:
            X_full = X_woe
        return self.model.predict_proba(X_full)[:, 1]


def _fit_logistic(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def train_woe_logistic(
    df: pd.DataFrame, feature_cols: list, target_col: str, n_bins: int = 10
) -> FittedModel:
    """Treina um modelo de regressao logistica com binning WoE sobre `feature_cols`."""
    data = df.dropna(subset=[target_col] + feature_cols)
    binner = WoEBinner(n_bins=n_bins)
    X_woe = binner.fit_transform(data[feature_cols], data[target_col])
    model = _fit_logistic(X_woe, data[target_col])
    return FittedModel(binner=binner, model=model, feature_cols=feature_cols)


def train_per_faixa(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    faixa_col: str = "faixa_atraso",
    n_bins: int = 10,
) -> dict:
    """Treina um modelo de regressao logistica independente para cada faixa
    de atraso presente em `df` (exclui faixas sem variancia no target).
    """
    modelos = {}
    for faixa, sub in df.groupby(faixa_col):
        sub_valid = sub.dropna(subset=[target_col])
        if sub_valid[target_col].nunique() < 2 or len(sub_valid) < 50:
            continue
        modelos[faixa] = train_woe_logistic(sub_valid, feature_cols, target_col, n_bins=n_bins)
    return modelos


def train_pooled(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    faixa_col: str = "faixa_atraso",
    n_bins: int = 10,
) -> FittedModel:
    """Treina um unico modelo sobre toda a populacao, com a faixa de atraso
    codificada como variaveis dummy (one-hot, sem a primeira categoria).
    """
    data = df.dropna(subset=[target_col] + feature_cols)
    binner = WoEBinner(n_bins=n_bins)
    X_woe = binner.fit_transform(data[feature_cols], data[target_col])

    dummies = pd.get_dummies(data[faixa_col], prefix="faixa", drop_first=True)
    dummy_cols = list(dummies.columns)

    X_full = pd.concat([X_woe.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    model = _fit_logistic(X_full, data[target_col].reset_index(drop=True))

    return FittedModel(
        binner=binner, model=model, feature_cols=feature_cols, dummy_cols=dummy_cols, faixa_col=faixa_col
    )

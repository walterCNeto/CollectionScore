"""Metricas de avaliacao de score de collection: deciles, AUC/KS/Gini por
faixa, PSI entre fotos mensais. Todas as metricas numericas sao arredondadas
a 3 casas decimais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROUND = 3


def _round(x):
    return float(np.round(x, ROUND))


def decile_table(
    score: np.ndarray,
    target: np.ndarray,
    saldo_atraso: np.ndarray,
    valor_recuperado: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Tabela por decil de score (10 = maior score) com cure rate em #
    contratos, cash cure rate (R$ recuperado / R$ em atraso elegivel), saldo
    medio e caixa gerado.

    Decil 10 = contratos com maior score (mais propensos ao evento).
    """
    df = pd.DataFrame(
        {
            "score": score,
            "target": target,
            "saldo_atraso": saldo_atraso,
        }
    )
    if valor_recuperado is None:
        df["valor_recuperado"] = df["target"] * df["saldo_atraso"]
    else:
        df["valor_recuperado"] = valor_recuperado

    df = df.dropna(subset=["score", "target"])
    ranks = df["score"].rank(method="first")
    df["decil"] = pd.qcut(ranks, n_deciles, labels=False) + 1

    grp = df.groupby("decil")
    out = pd.DataFrame(
        {
            "n_contratos": grp.size(),
            "cure_rate_contratos": grp["target"].mean(),
            "saldo_medio": grp["saldo_atraso"].mean(),
            "caixa_gerado": grp["valor_recuperado"].sum(),
            "saldo_atraso_total": grp["saldo_atraso"].sum(),
        }
    )
    out["cash_cure_rate"] = out["caixa_gerado"] / out["saldo_atraso_total"].replace(0, np.nan)
    out = out.reset_index().sort_values("decil", ascending=False).reset_index(drop=True)

    num_cols = ["cure_rate_contratos", "saldo_medio", "caixa_gerado", "saldo_atraso_total", "cash_cure_rate"]
    out[num_cols] = out[num_cols].round(ROUND)
    return out


def decile_table_cash(
    score: np.ndarray,
    target: np.ndarray,
    saldo_atraso: np.ndarray,
    valor_recuperado: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Tabela de decis "de caixa": os contratos sao ordenados por score
    (decrescente) e agrupados em `n_deciles` faixas com massa de saldo em
    atraso aproximadamente igual (em vez de numero de contratos igual, como
    em `decile_table`). Util para priorizacao por volume financeiro.
    """
    df = pd.DataFrame(
        {
            "score": score,
            "target": target,
            "saldo_atraso": saldo_atraso,
        }
    ).dropna(subset=["score", "target"])
    if valor_recuperado is None:
        df["valor_recuperado"] = df["target"] * df["saldo_atraso"]
    else:
        df["valor_recuperado"] = valor_recuperado

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    cum_saldo = df["saldo_atraso"].cumsum()
    total_saldo = cum_saldo.iloc[-1]
    bordas = np.linspace(0, total_saldo, n_deciles + 1)
    bordas[0] = -np.inf
    bordas[-1] = np.inf
    df["decil_caixa"] = pd.cut(cum_saldo, bordas, labels=False) + 1
    # decil 10 = maior score (primeira faixa de massa de saldo percorrida)
    df["decil_caixa"] = n_deciles + 1 - df["decil_caixa"]

    grp = df.groupby("decil_caixa")
    out = pd.DataFrame(
        {
            "n_contratos": grp.size(),
            "cure_rate_contratos": grp["target"].mean(),
            "saldo_medio": grp["saldo_atraso"].mean(),
            "caixa_gerado": grp["valor_recuperado"].sum(),
            "saldo_atraso_total": grp["saldo_atraso"].sum(),
        }
    )
    out["cash_cure_rate"] = out["caixa_gerado"] / out["saldo_atraso_total"].replace(0, np.nan)
    out = out.reset_index().rename(columns={"decil_caixa": "decil"}).sort_values(
        "decil", ascending=False
    ).reset_index(drop=True)

    num_cols = ["cure_rate_contratos", "saldo_medio", "caixa_gerado", "saldo_atraso_total", "cash_cure_rate"]
    out[num_cols] = out[num_cols].round(ROUND)
    return out


def ks_statistic(score: np.ndarray, target: np.ndarray) -> float:
    df = pd.DataFrame({"score": score, "target": target}).dropna()
    df = df.sort_values("score")
    n_pos = df["target"].sum()
    n_neg = len(df) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = (df["target"] == 1).cumsum() / n_pos
    cum_neg = (df["target"] == 0).cumsum() / n_neg
    ks = (cum_pos - cum_neg).abs().max()
    return _round(ks)


def gini_coefficient(score: np.ndarray, target: np.ndarray) -> float:
    auc = auc_score(score, target)
    if np.isnan(auc):
        return float("nan")
    return _round(2 * auc - 1)


def auc_score(score: np.ndarray, target: np.ndarray) -> float:
    df = pd.DataFrame({"score": score, "target": target}).dropna()
    if df["target"].nunique() < 2:
        return float("nan")
    return _round(roc_auc_score(df["target"], df["score"]))


def metrics_by_faixa(
    df: pd.DataFrame, score_col: str, target_col: str, faixa_col: str = "faixa_atraso"
) -> pd.DataFrame:
    """AUC, KS e Gini do score para prever `target_col`, calculados dentro de
    cada faixa de atraso.
    """
    rows = []
    for faixa, sub in df.groupby(faixa_col):
        sub_valid = sub.dropna(subset=[score_col, target_col])
        rows.append(
            {
                "faixa_atraso": faixa,
                "n": len(sub_valid),
                "auc": auc_score(sub_valid[score_col].to_numpy(), sub_valid[target_col].to_numpy()),
                "ks": ks_statistic(sub_valid[score_col].to_numpy(), sub_valid[target_col].to_numpy()),
                "gini": gini_coefficient(sub_valid[score_col].to_numpy(), sub_valid[target_col].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("faixa_atraso").reset_index(drop=True)


def psi(score_base: np.ndarray, score_atual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index entre duas distribuicoes de score (ex.:
    duas fotos mensais), usando bins de quantil da distribuicao base.
    """
    score_base = pd.Series(score_base).dropna()
    score_atual = pd.Series(score_atual).dropna()
    if score_base.empty or score_atual.empty:
        return float("nan")

    quantis = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(score_base, quantis))
    if edges.shape[0] <= 2:
        return 0.0
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts = pd.cut(score_base, edges).value_counts(sort=False)
    atual_counts = pd.cut(score_atual, edges).value_counts(sort=False)

    base_pct = np.maximum(base_counts / base_counts.sum(), 1e-4)
    atual_pct = np.maximum(atual_counts / atual_counts.sum(), 1e-4)

    psi_val = ((atual_pct - base_pct) * np.log(atual_pct / base_pct)).sum()
    return _round(psi_val)


def psi_between_snapshots(
    df: pd.DataFrame, score_col: str, snapshot_col: str = "snapshot", n_bins: int = 10
) -> pd.DataFrame:
    """PSI de cada snapshot contra o snapshot base (o primeiro disponivel)."""
    snapshots = sorted(df[snapshot_col].unique())
    base_snap = snapshots[0]
    base_scores = df.loc[df[snapshot_col] == base_snap, score_col].to_numpy()

    rows = []
    for snap in snapshots:
        atual_scores = df.loc[df[snapshot_col] == snap, score_col].to_numpy()
        rows.append({"snapshot": snap, "psi_vs_base": psi(base_scores, atual_scores, n_bins=n_bins)})
    return pd.DataFrame(rows)

"""Metricas de avaliacao centradas em regularizacao financeira.

Todas as metricas sao calculadas na janela M -> M+1 e, quando aplicavel, por
faixa de atraso. Numeros arredondados a 3 casas decimais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from collscore import FAIXAS

ROUND = 3


def _round(x):
    return float(np.round(x, ROUND))


def resumo_regularizacao(
    df: pd.DataFrame,
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado: np.ndarray | pd.Series | None = None,
    faixa_col: str = "faixa_atraso",
) -> pd.DataFrame:
    """Tabela de regularizacao financeira por faixa de atraso (+ linha TOTAL):
    saldo em atraso elegivel, caixa recuperado, taxa_reg_contratos
    (# regularizados / # elegiveis) e taxa_reg_caixa (R$ recuperado / R$ em
    atraso elegivel).

    `valor_recuperado` default = `target_col` * `parcela_valor` se a coluna
    `parcela_valor` existir em `df`; caso contrario, usa `target_col` * saldo
    em atraso (fallback).
    """
    data = df.dropna(subset=[target_col]).copy()
    if valor_recuperado is None:
        if "parcela_valor" in data.columns:
            valor_recuperado = data[target_col] * data["parcela_valor"]
        else:
            valor_recuperado = data[target_col] * data[saldo_col]
    data["_valor_recuperado"] = np.asarray(valor_recuperado)

    def _linha(sub: pd.DataFrame) -> dict:
        saldo_total = sub[saldo_col].sum()
        caixa = sub["_valor_recuperado"].sum()
        return {
            "n_elegivel": len(sub),
            "saldo_atraso_total": _round(saldo_total),
            "caixa_recuperado": _round(caixa),
            "taxa_reg_contratos": _round(sub[target_col].mean()),
            "taxa_reg_caixa": _round(caixa / saldo_total) if saldo_total > 0 else float("nan"),
        }

    linhas = []
    for faixa, sub in data.groupby(faixa_col):
        linhas.append({faixa_col: faixa, **_linha(sub)})
    linhas.append({faixa_col: "TOTAL", **_linha(data)})

    ordem = {f: i for i, f in enumerate(FAIXAS)}
    out = pd.DataFrame(linhas)
    out["_ordem"] = out[faixa_col].map(lambda f: ordem.get(f, len(FAIXAS)))
    return out.sort_values("_ordem").drop(columns="_ordem").reset_index(drop=True)


def roll_rate_saldo(
    panel: pd.DataFrame,
    faixa_col: str = "faixa_atraso",
    saldo_col: str = "saldo_devedor",
) -> pd.DataFrame:
    """Matriz de transicao de faixas M -> M+1 ponderada por saldo (R$), e nao
    por numero de contratos: cada celula [origem, destino] e a fracao do
    saldo que estava na faixa `origem` em M e foi para a faixa `destino` em
    M+1. Cada linha (faixa de origem) soma 1.000.
    """
    nxt = panel[["contrato_id", "snapshot", faixa_col]].copy()
    nxt["snapshot"] = nxt["snapshot"] - 1
    nxt = nxt.rename(columns={faixa_col: f"{faixa_col}_next"})

    merged = panel.merge(nxt, on=["contrato_id", "snapshot"], how="left")
    merged = merged.dropna(subset=[f"{faixa_col}_next"])

    pivot = (
        merged.groupby([faixa_col, f"{faixa_col}_next"])[saldo_col]
        .sum()
        .unstack(fill_value=0.0)
    )
    pivot = pivot.reindex(index=FAIXAS, columns=FAIXAS, fill_value=0.0)

    row_totais = pivot.sum(axis=1)
    matriz = pivot.div(row_totais.replace(0, np.nan), axis=0)
    matriz = matriz.loc[row_totais > 0]
    return matriz.round(ROUND)


def decile_table(
    score: np.ndarray,
    target: np.ndarray,
    saldo_atraso: np.ndarray,
    valor_recuperado: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Tabela por decil de score (10 = maior score) com taxa de regularizacao
    em # contratos, caixa recuperado e taxa de regularizacao em R$.

    Decil 10 = contratos com maior score (mais propensos ao evento).
    """
    df = pd.DataFrame({"score": score, "target": target, "saldo_atraso": saldo_atraso})
    if valor_recuperado is None:
        df["valor_recuperado"] = df["target"] * df["saldo_atraso"]
    else:
        df["valor_recuperado"] = np.asarray(valor_recuperado)

    df = df.dropna(subset=["score", "target"])
    ranks = df["score"].rank(method="first")
    df["decil"] = pd.qcut(ranks, n_deciles, labels=False) + 1

    grp = df.groupby("decil")
    out = pd.DataFrame(
        {
            "n_contratos": grp.size(),
            "taxa_reg_contratos": grp["target"].mean(),
            "saldo_medio": grp["saldo_atraso"].mean(),
            "caixa_recuperado": grp["valor_recuperado"].sum(),
            "saldo_atraso_total": grp["saldo_atraso"].sum(),
        }
    )
    out["taxa_reg_caixa"] = out["caixa_recuperado"] / out["saldo_atraso_total"].replace(0, np.nan)
    out = out.reset_index().sort_values("decil", ascending=False).reset_index(drop=True)

    num_cols = ["taxa_reg_contratos", "saldo_medio", "caixa_recuperado", "saldo_atraso_total", "taxa_reg_caixa"]
    out[num_cols] = out[num_cols].round(ROUND)
    return out


def _ks_statistic(score: np.ndarray, target: np.ndarray) -> float:
    df = pd.DataFrame({"score": score, "target": target}).dropna().sort_values("score")
    n_pos = df["target"].sum()
    n_neg = len(df) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = (df["target"] == 1).cumsum() / n_pos
    cum_neg = (df["target"] == 0).cumsum() / n_neg
    return float((cum_pos - cum_neg).abs().max())


def _auc_score(score: np.ndarray, target: np.ndarray) -> float:
    df = pd.DataFrame({"score": score, "target": target}).dropna()
    if df["target"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df["target"], df["score"]))


def model_ks_gini(score: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    """KS e Gini globais do score (uma linha), unica avaliacao reportada do
    modelo pooled.
    """
    auc = _auc_score(score, target)
    gini = 2 * auc - 1 if not np.isnan(auc) else float("nan")
    ks = _ks_statistic(score, target)
    n = int(pd.Series(target).notna().sum())
    return pd.DataFrame([{"n": n, "ks": _round(ks), "gini": _round(gini)}])

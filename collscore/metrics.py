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


def taxas_evento_por_faixa(
    df: pd.DataFrame,
    target_cols: list[str],
    faixa_col: str = "faixa_atraso",
) -> pd.DataFrame:
    """Taxa media de cada coluna de `target_cols` por faixa de atraso (+
    linha TOTAL). Cada coluna e tratada com seu proprio dropna (os targets
    podem ter padroes de nulos distintos). Util para comparar o target
    escolhido com alternativas (ex.: `pagou_1parcela` vs `cura_total` vs
    `melhora_faixa`).
    """
    ordem = {f: i for i, f in enumerate(FAIXAS)}
    faixas_presentes = sorted(
        df[faixa_col].dropna().unique(), key=lambda f: ordem.get(f, len(FAIXAS))
    )

    linhas = []
    for faixa in faixas_presentes:
        sub = df[df[faixa_col] == faixa]
        linha = {faixa_col: faixa}
        for col in target_cols:
            validos = sub[col].dropna()
            linha[col] = _round(validos.mean()) if len(validos) > 0 else float("nan")
        linhas.append(linha)

    linha_total = {faixa_col: "TOTAL"}
    for col in target_cols:
        validos = df[col].dropna()
        linha_total[col] = _round(validos.mean()) if len(validos) > 0 else float("nan")
    linhas.append(linha_total)

    return pd.DataFrame(linhas)


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

    Atencao: o KS/Gini GLOBAL mistura contratos de faixas de atraso muito
    diferentes, cuja taxa de evento de base ja varia bastante entre si; essa
    separacao inter-faixa INFLA o KS/Gini global em relacao ao poder
    discriminante do score DENTRO de cada faixa (ver `ks_gini_por_faixa`).
    """
    auc = _auc_score(score, target)
    gini = 2 * auc - 1 if not np.isnan(auc) else float("nan")
    ks = _ks_statistic(score, target)
    n = int(pd.Series(target).notna().sum())
    return pd.DataFrame([{"n": n, "ks": _round(ks), "gini": _round(gini)}])


def ks_gini_por_faixa(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    faixa_col: str = "faixa_atraso",
) -> pd.DataFrame:
    """KS e Gini do MESMO score pooled, calculados separadamente DENTRO de
    cada faixa de atraso (sem treinar nenhum modelo novo por faixa). Uma
    linha por faixa presente em `df`.

    O KS/Gini global (`model_ks_gini`) e tipicamente maior que qualquer um
    destes, porque parte do seu poder discriminante vem simplesmente de
    separar faixas com taxas de evento muito diferentes -- o que nao reflete
    a capacidade do score de ordenar contratos DENTRO de uma mesma faixa.
    """
    linhas = []
    for faixa, sub in df.groupby(faixa_col):
        sub_valid = sub.dropna(subset=[score_col, target_col])
        auc = _auc_score(sub_valid[score_col].to_numpy(), sub_valid[target_col].to_numpy())
        gini = 2 * auc - 1 if not np.isnan(auc) else float("nan")
        ks = _ks_statistic(sub_valid[score_col].to_numpy(), sub_valid[target_col].to_numpy())
        linhas.append(
            {
                faixa_col: faixa,
                "n": len(sub_valid),
                "ks": _round(ks) if not np.isnan(ks) else float("nan"),
                "gini": _round(gini) if not np.isnan(gini) else float("nan"),
            }
        )

    ordem = {f: i for i, f in enumerate(FAIXAS)}
    out = pd.DataFrame(linhas)
    out["_ordem"] = out[faixa_col].map(lambda f: ordem.get(f, len(FAIXAS)))
    return out.sort_values("_ordem").drop(columns="_ordem").reset_index(drop=True)

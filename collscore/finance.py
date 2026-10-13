"""Avaliacao financeira da estrategia de cobranca.

Dado um book escorado (score de cura por contrato), simula o resultado de
acionar os top decis de cada faixa de atraso atraves de um canal de custo
crescente (sms -> humano -> juridico), e deriva a curva de lucro
L(corte) = caixa incremental - custo de acionamento, o corte otimo por faixa
e o ROI da estrategia.

"Caixa incremental" e o caixa recuperado acima de uma taxa de cura baseline
que ocorreria sem qualquer acionamento (self-cure). Por padrao essa baseline
e estimada como uma fracao (`baseline_self_cure_pct`) da taxa de cura
observada, aplicada apenas as faixas iniciais de atraso (<=30 dias) -- em
linha com o componente de self-cure gerado em `simulate.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANAL_CUSTOS = {"sms": 0.15, "humano": 8.00, "juridico": 150.00}

CANAL_POR_FAIXA = {
    "em_dia": None,
    "1-15": "sms",
    "16-30": "sms",
    "31-60": "humano",
    "61-90": "humano",
    "91-180": "juridico",
    "180+": "juridico",
}

HAIRCUT_FAIXA = {
    "em_dia": 1.00,
    "1-15": 0.95,
    "16-30": 0.90,
    "31-60": 0.80,
    "61-90": 0.65,
    "91-180": 0.50,
    "180+": 0.35,
}

FAIXAS_SELF_CURE = ("1-15", "16-30")
BASELINE_SELF_CURE_PCT = 0.30


def profit_curve(
    df: pd.DataFrame,
    faixa: str,
    score_col: str = "score",
    target_col: str = "cura_total",
    saldo_col: str = "saldo_atraso",
    faixa_col: str = "faixa_atraso",
    canal_custos: dict | None = None,
    canal_por_faixa: dict | None = None,
    haircut_faixa: dict | None = None,
    baseline_self_cure_pct: float = BASELINE_SELF_CURE_PCT,
    faixas_self_cure: tuple = FAIXAS_SELF_CURE,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Curva de lucro L(corte) para uma faixa de atraso, acionando os top
    `k` decis de score (k = 1..n_deciles, do mais ao menos propenso a cura).

    Retorna uma linha por numero de decis acionados, com custo, caixa
    incremental, lucro e ROI acumulados.
    """
    canal_custos = canal_custos or CANAL_CUSTOS
    canal_por_faixa = canal_por_faixa or CANAL_POR_FAIXA
    haircut_faixa = haircut_faixa or HAIRCUT_FAIXA

    sub = df.loc[df[faixa_col] == faixa, [score_col, target_col, saldo_col]].dropna()
    if sub.empty:
        return pd.DataFrame()

    canal = canal_por_faixa.get(faixa)
    custo_unitario = canal_custos.get(canal, 0.0) if canal else 0.0
    haircut = haircut_faixa.get(faixa, 1.0)

    cure_rate_total = sub[target_col].mean()
    baseline_rate = cure_rate_total * baseline_self_cure_pct if faixa in faixas_self_cure else 0.0
    incremental_rate_pop = max(cure_rate_total - baseline_rate, 0.0)
    # fracao da taxa de cura total que e incremental (atribuivel ao acionamento)
    fator_incremental = incremental_rate_pop / cure_rate_total if cure_rate_total > 0 else 0.0

    ranks = sub[score_col].rank(method="first")
    sub = sub.assign(decil=pd.qcut(ranks, n_deciles, labels=False) + 1)

    linhas = []
    for k in range(1, n_deciles + 1):
        acionados = sub[sub["decil"] > n_deciles - k]
        n_contratos = len(acionados)
        custo_total = n_contratos * custo_unitario

        caixa_bruto = (acionados[target_col] * acionados[saldo_col] * haircut).sum()
        caixa_incremental = caixa_bruto * fator_incremental

        lucro = caixa_incremental - custo_total
        roi = caixa_incremental / custo_total if custo_total > 0 else np.nan

        linhas.append(
            {
                "faixa_atraso": faixa,
                "canal": canal,
                "decis_acionados": k,
                "n_contratos": n_contratos,
                "custo_total": round(custo_total, 2),
                "caixa_incremental": round(caixa_incremental, 2),
                "lucro": round(lucro, 2),
                "roi": round(roi, 3) if not np.isnan(roi) else np.nan,
            }
        )
    return pd.DataFrame(linhas)


def optimal_cutoff(curve: pd.DataFrame) -> dict:
    """Dado o resultado de `profit_curve`, retorna o corte (numero de decis
    acionados) que maximiza o lucro, e o ROI nesse ponto.
    """
    if curve.empty:
        return {}
    melhor = curve.loc[curve["lucro"].idxmax()]
    return {
        "faixa_atraso": melhor["faixa_atraso"],
        "canal": melhor["canal"],
        "decis_acionados_otimo": int(melhor["decis_acionados"]),
        "n_contratos_otimo": int(melhor["n_contratos"]),
        "lucro_max": float(melhor["lucro"]),
        "roi_no_corte": float(melhor["roi"]) if not pd.isna(melhor["roi"]) else np.nan,
    }


def run_finance_analysis(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "cura_total",
    saldo_col: str = "saldo_atraso",
    faixa_col: str = "faixa_atraso",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda `profit_curve` + `optimal_cutoff` para cada faixa de atraso
    delinquente presente em `df`. Retorna (curva_completa, resumo_por_faixa).
    """
    faixas = [f for f in df[faixa_col].dropna().unique() if f != "em_dia"]
    curvas = []
    resumos = []
    for faixa in faixas:
        curva = profit_curve(
            df, faixa, score_col=score_col, target_col=target_col, saldo_col=saldo_col, faixa_col=faixa_col, **kwargs
        )
        if curva.empty:
            continue
        curvas.append(curva)
        resumos.append(optimal_cutoff(curva))

    curva_completa = pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame()
    resumo_df = pd.DataFrame(resumos)
    return curva_completa, resumo_df

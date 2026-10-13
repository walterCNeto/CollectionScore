"""Avaliacao financeira da estrategia de cobranca.

Dado um book escorado (score de `pagou_1parcela` por contrato), simula o
resultado de acionar os top decis de cada faixa de atraso atraves de um
canal de custo crescente (sms -> negativacao -> humano -> juridico), e
deriva a curva de lucro L(corte) = caixa incremental - custo de
acionamento, o corte otimo por faixa/canal e o ROI da estrategia.

Cada canal tem um `saldo_minimo_elegivel`: contratos com saldo em atraso
abaixo desse piso nao sao acionaveis por aquele canal (ex.: nao vale a pena
acionar o juridico, custo fixo alto, para um saldo pequeno) e sao excluidos
da populacao acionavel da faixa.

"Caixa incremental" e o caixa recuperado (parcela paga) acima de uma taxa de
regularizacao baseline que ocorreria sem qualquer acionamento (self-cure),
multiplicado pela efetividade do canal (`EFETIVIDADE_CANAL`). Essa
efetividade e uma PROXY simplificada de resposta diferencial por canal (nem
todo contrato acionado por sms responde como um contrato acionado pelo
juridico) -- e um passo intermediario, ate que o framework tenha dados
suficientes para uma modelagem causal de uplift por canal (ver o flag
`flag_self_cure_evento` de `simulate.py`, reservado para esse fim).

Por padrao a baseline de self-cure e estimada como uma fracao
(`baseline_self_cure_pct`) da taxa de regularizacao observada, aplicada
apenas as faixas iniciais de atraso (<=30 dias).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANAL_CUSTOS = {"sms": 0.15, "negativacao": 3.00, "humano": 8.00, "juridico": 150.00}

CANAL_POR_FAIXA = {
    "em_dia": None,
    "1-15": "sms",
    "16-30": "sms",
    "31-60": "negativacao",
    "61-90": "negativacao",
    "91-180": "juridico",
    "180+": "juridico",
}

SALDO_MINIMO_ELEGIVEL = {"sms": 0.0, "negativacao": 0.0, "humano": 0.0, "juridico": 5000.00}

EFETIVIDADE_CANAL = {"sms": 0.40, "negativacao": 0.65, "humano": 1.00, "juridico": 1.00}

FAIXAS_SELF_CURE = ("1-15", "16-30")
BASELINE_SELF_CURE_PCT = 0.30


def profit_curve(
    df: pd.DataFrame,
    faixa: str,
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado_col: str | None = "valor_recuperado",
    faixa_col: str = "faixa_atraso",
    canal_custos: dict | None = None,
    canal_por_faixa: dict | None = None,
    saldo_minimo_elegivel: dict | None = None,
    efetividade_canal: dict | None = None,
    baseline_self_cure_pct: float = BASELINE_SELF_CURE_PCT,
    faixas_self_cure: tuple = FAIXAS_SELF_CURE,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Curva de lucro L(corte) para uma faixa de atraso, acionando os top
    `k` decis de score (k = 1..n_deciles, do mais ao menos propenso ao
    evento) dentro da populacao elegivel ao canal daquela faixa (acima do
    piso de saldo em atraso). O caixa incremental e multiplicado pela
    efetividade do canal (ver `EFETIVIDADE_CANAL` no docstring do modulo).

    Retorna uma linha por numero de decis acionados, com custo, caixa
    incremental, lucro e ROI acumulados.
    """
    canal_custos = canal_custos or CANAL_CUSTOS
    canal_por_faixa = canal_por_faixa or CANAL_POR_FAIXA
    saldo_minimo_elegivel = saldo_minimo_elegivel or SALDO_MINIMO_ELEGIVEL
    efetividade_canal = efetividade_canal or EFETIVIDADE_CANAL

    cols = [score_col, target_col, saldo_col]
    if valor_recuperado_col and valor_recuperado_col in df.columns:
        cols.append(valor_recuperado_col)
    sub = df.loc[df[faixa_col] == faixa, cols].dropna(subset=[score_col, target_col, saldo_col])
    if sub.empty:
        return pd.DataFrame()

    canal = canal_por_faixa.get(faixa)
    custo_unitario = canal_custos.get(canal, 0.0) if canal else 0.0
    piso = saldo_minimo_elegivel.get(canal, 0.0) if canal else 0.0
    efetividade = efetividade_canal.get(canal, 1.0) if canal else 1.0

    sub = sub.loc[sub[saldo_col] >= piso]
    if sub.empty:
        return pd.DataFrame()

    if valor_recuperado_col and valor_recuperado_col in sub.columns:
        sub = sub.rename(columns={valor_recuperado_col: "valor_recuperado"})
    else:
        sub = sub.assign(valor_recuperado=sub[target_col] * sub[saldo_col])

    taxa_reg_total = sub[target_col].mean()
    baseline_rate = taxa_reg_total * baseline_self_cure_pct if faixa in faixas_self_cure else 0.0
    incremental_rate_pop = max(taxa_reg_total - baseline_rate, 0.0)
    fator_incremental = incremental_rate_pop / taxa_reg_total if taxa_reg_total > 0 else 0.0

    ranks = sub[score_col].rank(method="first")
    sub = sub.assign(decil=pd.qcut(ranks, min(n_deciles, len(sub)), labels=False, duplicates="drop") + 1)
    n_deciles_efetivo = sub["decil"].max()

    linhas = []
    for k in range(1, n_deciles_efetivo + 1):
        acionados = sub[sub["decil"] > n_deciles_efetivo - k]
        n_contratos = len(acionados)
        custo_total = n_contratos * custo_unitario

        caixa_bruto = acionados["valor_recuperado"].sum()
        caixa_incremental = caixa_bruto * fator_incremental * efetividade

        lucro = caixa_incremental - custo_total
        roi = caixa_incremental / custo_total if custo_total > 0 else np.nan

        linhas.append(
            {
                "faixa_atraso": faixa,
                "canal": canal,
                "saldo_minimo_elegivel": piso,
                "efetividade_canal": efetividade,
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
        "saldo_minimo_elegivel": melhor["saldo_minimo_elegivel"],
        "efetividade_canal": melhor["efetividade_canal"],
        "decis_acionados_otimo": int(melhor["decis_acionados"]),
        "n_contratos_otimo": int(melhor["n_contratos"]),
        "custo_total": float(melhor["custo_total"]),
        "lucro_max": float(melhor["lucro"]),
        "roi_no_corte": float(melhor["roi"]) if not pd.isna(melhor["roi"]) else np.nan,
    }


def run_finance_analysis(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado_col: str | None = "valor_recuperado",
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
            df,
            faixa,
            score_col=score_col,
            target_col=target_col,
            saldo_col=saldo_col,
            valor_recuperado_col=valor_recuperado_col,
            faixa_col=faixa_col,
            **kwargs,
        )
        if curva.empty:
            continue
        curvas.append(curva)
        resumos.append(optimal_cutoff(curva))

    curva_completa = pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame()
    resumo_df = pd.DataFrame(resumos)
    return curva_completa, resumo_df

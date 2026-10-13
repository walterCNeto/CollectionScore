"""Avaliacao financeira da estrategia de cobranca.

Dado um book escorado (score de `pagou_1parcela` por contrato), simula o
resultado de acionar os top decis de cada faixa de atraso atraves de um
canal de custo crescente (sms -> negativacao -> call_center -> juridico), e
deriva a curva de lucro L(corte) = caixa incremental - custo de
acionamento, o corte otimo por faixa/canal e o ROI da estrategia.

Cada canal tem um `saldo_minimo_elegivel`: contratos com saldo em atraso
abaixo desse piso nao sao acionaveis por aquele canal (ex.: nao vale a pena
acionar o juridico, custo fixo alto, para um saldo pequeno) e sao excluidos
da populacao acionavel da faixa.

"Caixa incremental" e o caixa recuperado (parcela paga) acima de uma taxa de
regularizacao baseline que ocorreria sem qualquer acionamento (self-cure),
multiplicado pela efetividade da CELULA canal x faixa (`EFETIVIDADE`). Essa
efetividade e uma PROXY simplificada de resposta diferencial -- um sms tem
efeito diferente sobre uma faixa inicial de atraso do que sobre uma faixa
profunda, e o mesmo vale para os demais canais -- e um passo intermediario,
ate que o framework tenha dados suficientes para uma modelagem causal de
uplift por canal (ver o flag `flag_self_cure_evento` de `simulate.py`,
reservado para esse fim). Uma efetividade 0.0 (ex.: juridico nas faixas
iniciais) torna o canal inelegivel NA PRATICA para aquela faixa, alem de
qualquer piso de saldo.

Alem do mapeamento fixo faixa->canal (`CANAL_POR_FAIXA`, modo "manual"),
este modulo oferece uma escolha ENDOGENA de canal: `compare_channels`
testa, para uma faixa, todos os canais elegiveis (independente do
mapeamento fixo) e `optimal_strategy` escolhe por faixa o canal vencedor
segundo um criterio (lucro ou ROI no corte otimo), compondo a estrategia
completa e seus totais. As duas estrategias (lucro e ROI) sao dois pontos
extremos de uma fronteira orcamento x lucro: a de ROI gasta pouco e maximiza
retorno por R$ investido, a de lucro gasta mais para maximizar o retorno
absoluto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANAL_CUSTOS = {"sms": 0.15, "negativacao": 3.00, "call_center": 8.00, "juridico": 150.00}

CANAL_POR_FAIXA = {
    "em_dia": None,
    "1-15": "sms",
    "16-30": "sms",
    "31-60": "negativacao",
    "61-90": "negativacao",
    "91-180": "juridico",
    "180+": "juridico",
}

SALDO_MINIMO_ELEGIVEL = {"sms": 0.0, "negativacao": 0.0, "call_center": 0.0, "juridico": 5000.00}

# Efetividade = proxy de resposta diferencial, por celula canal x faixa.
# juridico com efetividade 0.0 nas faixas iniciais (1-15, 16-30, 31-60) =
# inelegivel na pratica para essas faixas, alem do piso de saldo.
EFETIVIDADE = {
    "sms": {"1-15": 0.50, "16-30": 0.45, "31-60": 0.30, "61-90": 0.20, "91-180": 0.10, "180+": 0.05},
    "negativacao": {"1-15": 0.40, "16-30": 0.55, "31-60": 0.65, "61-90": 0.60, "91-180": 0.45, "180+": 0.25},
    "call_center": {"1-15": 0.60, "16-30": 0.65, "31-60": 0.70, "61-90": 0.65, "91-180": 0.50, "180+": 0.30},
    "juridico": {"1-15": 0.0, "16-30": 0.0, "31-60": 0.0, "61-90": 0.70, "91-180": 0.90, "180+": 0.85},
}

FAIXAS_SELF_CURE = ("1-15", "16-30")
BASELINE_SELF_CURE_PCT = 0.30


def _lookup_efetividade(efetividade: dict, canal: str | None, faixa: str) -> float:
    """Efetividade da celula (canal, faixa); 0.0 se o canal nao foi acionado
    ou se a combinacao nao estiver definida na matriz.
    """
    if not canal:
        return 0.0
    return efetividade.get(canal, {}).get(faixa, 0.0)


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
    efetividade: dict | None = None,
    baseline_self_cure_pct: float = BASELINE_SELF_CURE_PCT,
    faixas_self_cure: tuple = FAIXAS_SELF_CURE,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Curva de lucro L(corte) para uma faixa de atraso, acionando os top
    `k` decis de score (k = 1..n_deciles, do mais ao menos propenso ao
    evento) dentro da populacao elegivel ao canal daquela faixa (acima do
    piso de saldo em atraso). O caixa incremental e multiplicado pela
    efetividade da celula canal x faixa (ver `EFETIVIDADE` no docstring do
    modulo).

    Retorna uma linha por numero de decis acionados, com custo, caixa
    incremental, lucro e ROI acumulados.
    """
    canal_custos = canal_custos or CANAL_CUSTOS
    canal_por_faixa = canal_por_faixa or CANAL_POR_FAIXA
    saldo_minimo_elegivel = saldo_minimo_elegivel or SALDO_MINIMO_ELEGIVEL
    efetividade = efetividade or EFETIVIDADE

    cols = [score_col, target_col, saldo_col]
    if valor_recuperado_col and valor_recuperado_col in df.columns:
        cols.append(valor_recuperado_col)
    sub = df.loc[df[faixa_col] == faixa, cols].dropna(subset=[score_col, target_col, saldo_col])
    if sub.empty:
        return pd.DataFrame()

    canal = canal_por_faixa.get(faixa)
    custo_unitario = canal_custos.get(canal, 0.0) if canal else 0.0
    piso = saldo_minimo_elegivel.get(canal, 0.0) if canal else 0.0
    efetividade_celula = _lookup_efetividade(efetividade, canal, faixa)

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
        caixa_incremental = caixa_bruto * fator_incremental * efetividade_celula

        lucro = caixa_incremental - custo_total
        roi = caixa_incremental / custo_total if custo_total > 0 else np.nan

        linhas.append(
            {
                "faixa_atraso": faixa,
                "canal": canal,
                "saldo_minimo_elegivel": piso,
                "efetividade": efetividade_celula,
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
        "efetividade": melhor["efetividade"],
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
    delinquente presente em `df`, usando o canal fixo de `canal_por_faixa`
    (modo "manual"). Retorna (curva_completa, resumo_por_faixa).
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


# --------------------------------------------------------------------------- escolha endogena de canal


def compare_channels(
    df: pd.DataFrame,
    faixa: str,
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado_col: str | None = "valor_recuperado",
    faixa_col: str = "faixa_atraso",
    canal_custos: dict | None = None,
    efetividade: dict | None = None,
    saldo_minimo_elegivel: dict | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compara, para UMA faixa de atraso, o resultado de aciona-la por CADA
    canal elegivel (todos os canais definidos em `canal_custos`),
    independente do mapeamento fixo `CANAL_POR_FAIXA`. Para cada canal, roda
    `profit_curve` (usando a efetividade da celula canal x faixa) e extrai
    o corte que maximiza o lucro (`optimal_cutoff`).

    Retorna uma linha por canal com: faixa_atraso, canal, decis_otimo,
    n_contratos, custo, lucro_max, roi_no_corte.
    """
    canal_custos = canal_custos or CANAL_CUSTOS
    efetividade = efetividade or EFETIVIDADE
    saldo_minimo_elegivel = saldo_minimo_elegivel or SALDO_MINIMO_ELEGIVEL

    linhas = []
    for canal in canal_custos:
        curva = profit_curve(
            df,
            faixa,
            score_col=score_col,
            target_col=target_col,
            saldo_col=saldo_col,
            valor_recuperado_col=valor_recuperado_col,
            faixa_col=faixa_col,
            canal_custos=canal_custos,
            canal_por_faixa={faixa: canal},
            saldo_minimo_elegivel=saldo_minimo_elegivel,
            efetividade=efetividade,
            n_deciles=n_deciles,
        )
        if curva.empty:
            continue
        melhor = optimal_cutoff(curva)
        linhas.append(
            {
                "faixa_atraso": faixa,
                "canal": canal,
                "decis_otimo": melhor["decis_acionados_otimo"],
                "n_contratos": melhor["n_contratos_otimo"],
                "custo": melhor["custo_total"],
                "lucro_max": melhor["lucro_max"],
                "roi_no_corte": melhor["roi_no_corte"],
            }
        )
    return pd.DataFrame(linhas)


def run_channel_comparison(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado_col: str | None = "valor_recuperado",
    faixa_col: str = "faixa_atraso",
    **kwargs,
) -> pd.DataFrame:
    """Roda `compare_channels` para cada faixa de atraso delinquente
    presente em `df`. Retorna a tabela comparativa canal x faixa completa.
    """
    faixas = [f for f in df[faixa_col].dropna().unique() if f != "em_dia"]
    tabelas = [
        compare_channels(
            df,
            faixa,
            score_col=score_col,
            target_col=target_col,
            saldo_col=saldo_col,
            valor_recuperado_col=valor_recuperado_col,
            faixa_col=faixa_col,
            **kwargs,
        )
        for faixa in faixas
    ]
    tabelas = [t for t in tabelas if not t.empty]
    return pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()


def optimal_strategy(
    df: pd.DataFrame,
    criterio: str = "lucro",
    score_col: str = "score",
    target_col: str = "pagou_1parcela",
    saldo_col: str = "saldo_atraso",
    valor_recuperado_col: str | None = "valor_recuperado",
    faixa_col: str = "faixa_atraso",
    **kwargs,
) -> dict:
    """Escolhe, para cada faixa, o canal vencedor segundo `criterio`
    ("lucro" maximiza `lucro_max`, "roi" maximiza `roi_no_corte`) dentre
    todos os canais comparados por `run_channel_comparison`.

    As estrategias "lucro" e "roi" sao dois pontos extremos de uma
    fronteira orcamento x lucro: "roi" tende a gastar pouco e maximizar o
    retorno por R$ investido, "lucro" gasta mais para maximizar o retorno
    absoluto.

    Retorna um dict com `estrategia` (DataFrame faixa -> canal escolhido,
    corte, custo, lucro, ROI), `criterio`, `lucro_total`, `custo_total` e
    `roi_agregado` (caixa incremental total / custo total da estrategia).
    """
    if criterio not in ("lucro", "roi"):
        raise ValueError("criterio deve ser 'lucro' ou 'roi'")

    comparacao = run_channel_comparison(
        df,
        score_col=score_col,
        target_col=target_col,
        saldo_col=saldo_col,
        valor_recuperado_col=valor_recuperado_col,
        faixa_col=faixa_col,
        **kwargs,
    )
    if comparacao.empty:
        return {
            "estrategia": pd.DataFrame(),
            "criterio": criterio,
            "lucro_total": 0.0,
            "custo_total": 0.0,
            "roi_agregado": float("nan"),
        }

    coluna_criterio = "lucro_max" if criterio == "lucro" else "roi_no_corte"
    comp = comparacao.dropna(subset=[coluna_criterio])
    if comp.empty:
        return {
            "estrategia": pd.DataFrame(),
            "criterio": criterio,
            "lucro_total": 0.0,
            "custo_total": 0.0,
            "roi_agregado": float("nan"),
        }

    idx_vencedores = comp.groupby(faixa_col)[coluna_criterio].idxmax()
    estrategia = comp.loc[idx_vencedores].sort_values(faixa_col).reset_index(drop=True)

    lucro_total = float(estrategia["lucro_max"].sum())
    custo_total = float(estrategia["custo"].sum())
    caixa_incremental_total = lucro_total + custo_total
    roi_agregado = caixa_incremental_total / custo_total if custo_total > 0 else float("nan")

    return {
        "estrategia": estrategia,
        "criterio": criterio,
        "lucro_total": round(lucro_total, 3),
        "custo_total": round(custo_total, 3),
        "roi_agregado": round(roi_agregado, 3) if not np.isnan(roi_agregado) else float("nan"),
    }


def fronteira_orcamento_lucro(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """As estrategias "lucro" e "roi" como dois pontos da fronteira
    orcamento x lucro: para cada criterio, (custo_total, lucro_total,
    roi_agregado). Util para visualizar o trade-off lado a lado.
    """
    linhas = []
    for criterio in ("roi", "lucro"):
        resultado = optimal_strategy(df, criterio=criterio, **kwargs)
        linhas.append(
            {
                "criterio": criterio,
                "custo_total": resultado["custo_total"],
                "lucro_total": resultado["lucro_total"],
                "roi_agregado": resultado["roi_agregado"],
            }
        )
    return pd.DataFrame(linhas)

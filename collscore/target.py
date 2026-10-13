"""Targets de collection scoring com janela M -> M+1.

Todos os targets sao calculados por contrato, comparando o estado no snapshot
M com o estado no snapshot M+1 (quando existente). A populacao de referencia
("book" de cobranca) e, por padrao, restrita a contratos delinquentes em M
(faixa_atraso != 'em_dia'); contratos sem M+1 disponivel (ultima foto ou
liquidacao) recebem NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from collscore import FAIXA_SEVERIDADE

TARGET_COLS = ["cura_total", "melhora_faixa", "pagou_1parcela"]


def _attach_next(
    panel: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Faz merge do estado em M+1 nas colunas `cols`, respeitando buracos no
    painel (contratos liquidados) -- nao usa shift() ingenuo.
    """
    nxt = panel[["contrato_id", "snapshot"] + cols].copy()
    nxt["snapshot"] = nxt["snapshot"] - 1
    nxt = nxt.rename(columns={c: f"{c}_next" for c in cols})
    out = panel.merge(nxt, on=["contrato_id", "snapshot"], how="left")
    return out


def compute_targets(
    panel: pd.DataFrame,
    restrict_to_delinquent: bool = True,
    faixa_col: str = "faixa_atraso",
    parcelas_col: str = "parcelas_pagas",
) -> pd.DataFrame:
    """Calcula os 3 targets parametrizaveis e retorna um DataFrame indexado
    por (contrato_id, snapshot) com as colunas de `TARGET_COLS`.
    """
    df = _attach_next(panel, [faixa_col, parcelas_col])

    sev = df[faixa_col].map(FAIXA_SEVERIDADE)
    sev_next = df[f"{faixa_col}_next"].map(FAIXA_SEVERIDADE)
    tem_next = df[f"{faixa_col}_next"].notna()

    cura_total = np.where(tem_next, (sev_next == 0).astype(float), np.nan)
    melhora_faixa = np.where(tem_next, (sev_next < sev).astype(float), np.nan)
    pagou_1parcela = np.where(
        tem_next,
        (df[f"{parcelas_col}_next"] > df[parcelas_col]).astype(float),
        np.nan,
    )

    if restrict_to_delinquent:
        delinquente = (sev > 0).to_numpy()
        cura_total = np.where(delinquente, cura_total, np.nan)
        melhora_faixa = np.where(delinquente, melhora_faixa, np.nan)
        pagou_1parcela = np.where(delinquente, pagou_1parcela, np.nan)

    out = pd.DataFrame(
        {
            "contrato_id": df["contrato_id"],
            "snapshot": df["snapshot"],
            "cura_total": cura_total,
            "melhora_faixa": melhora_faixa,
            "pagou_1parcela": pagou_1parcela,
        }
    )
    return out

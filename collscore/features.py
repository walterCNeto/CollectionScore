"""Construcao do book de features a partir do painel simulado, no ponto-no-tempo
de cada snapshot M (sem olhar para M+1 -- isso e papel de target.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "pct_saldo_atraso",
    "pct_parcelas_pagas",
    "dias_atraso",
    "recencia_pgto",
    "atraso_max_12m",
    "mob_rel",
    "quebra_promessa",
    "n_acionamentos_3m",
    "ticket_rel",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Recebe o painel de `simulate_portfolio` e retorna um DataFrame com as
    colunas de identificacao (contrato_id, snapshot, produto, faixa_atraso) e
    as features do book (`FEATURE_COLS`), uma linha por (contrato_id, snapshot).
    """
    df = panel.sort_values(["contrato_id", "snapshot"]).reset_index(drop=True).copy()

    pct_saldo_atraso = np.where(
        df["saldo_devedor"] > 0, np.clip(df["saldo_atraso"] / df["saldo_devedor"], 0, 1), 0.0
    )
    pct_parcelas_pagas = np.where(
        df["parcelas_vencidas"] > 0,
        np.clip(df["parcelas_pagas"] / df["parcelas_vencidas"].replace(0, np.nan), 0, 1),
        1.0,
    )
    pct_parcelas_pagas = np.nan_to_num(pct_parcelas_pagas, nan=1.0)

    mob_rel = np.clip(df["mob"] / df["prazo"].replace(0, np.nan), 0, 1)
    mob_rel = np.nan_to_num(mob_rel, nan=0.0)

    grp = df.groupby("contrato_id", sort=False)
    atraso_max_12m = grp["dias_atraso"].transform(
        lambda s: s.rolling(window=12, min_periods=1).max()
    )
    n_acionamentos_3m = grp["acionamentos_mes"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )

    ticket_medio_produto = df.groupby("produto")["valor_financiado"].transform("mean")
    ticket_rel = df["valor_financiado"] / ticket_medio_produto

    out = pd.DataFrame(
        {
            "contrato_id": df["contrato_id"],
            "snapshot": df["snapshot"],
            "produto": df["produto"],
            "faixa_atraso": df["faixa_atraso"],
            "saldo_devedor": df["saldo_devedor"],
            "saldo_atraso": df["saldo_atraso"],
            "pct_saldo_atraso": pct_saldo_atraso,
            "pct_parcelas_pagas": pct_parcelas_pagas,
            "dias_atraso": df["dias_atraso"],
            "recencia_pgto": df["recencia_pgto"],
            "atraso_max_12m": atraso_max_12m,
            "mob_rel": mob_rel,
            "quebra_promessa": df["quebra_promessa_mes"],
            "n_acionamentos_3m": n_acionamentos_3m,
            "ticket_rel": ticket_rel,
        }
    )
    return out

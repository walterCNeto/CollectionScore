import numpy as np
import pandas as pd
import pytest

from collscore import FAIXAS
from collscore.metrics import (
    decile_table,
    ks_gini_por_faixa,
    model_ks_gini,
    resumo_regularizacao,
    roll_rate_saldo,
)


def test_roll_rate_saldo_rows_sum_to_one(panel_small):
    matriz = roll_rate_saldo(panel_small)
    assert set(matriz.index) <= set(FAIXAS)
    assert list(matriz.columns) == FAIXAS
    somas = matriz.sum(axis=1)
    # cada linha (faixa de origem) e uma distribuicao de probabilidade do
    # saldo entre as faixas de destino -- deve somar 1.000 (tolerancia pelo
    # arredondamento a 3 casas decimais de cada celula).
    assert np.allclose(somas.to_numpy(), 1.0, atol=0.01)


def test_roll_rate_saldo_matches_manual_weighted_calc():
    panel = pd.DataFrame(
        {
            "contrato_id": [1, 1, 2, 2, 3, 3],
            "snapshot": [0, 1, 0, 1, 0, 1],
            "faixa_atraso": ["1-15", "em_dia", "1-15", "16-30", "1-15", "1-15"],
            "saldo_devedor": [100.0, 100.0, 300.0, 300.0, 600.0, 600.0],
        }
    )
    matriz = roll_rate_saldo(panel)
    # origem '1-15': saldo total = 100+300+600=1000; vai para em_dia(100),
    # 16-30(300), 1-15(600)
    assert matriz.loc["1-15", "em_dia"] == pytest.approx(0.1)
    assert matriz.loc["1-15", "16-30"] == pytest.approx(0.3)
    assert matriz.loc["1-15", "1-15"] == pytest.approx(0.6)


def test_resumo_regularizacao_tem_linha_total(book_small):
    data = book_small.dropna(subset=["pagou_1parcela"])
    resumo = resumo_regularizacao(data, target_col="pagou_1parcela")
    assert "TOTAL" in resumo["faixa_atraso"].to_numpy()
    total_row = resumo[resumo["faixa_atraso"] == "TOTAL"].iloc[0]
    assert total_row["n_elegivel"] == len(data)
    assert 0.0 <= total_row["taxa_reg_contratos"] <= 1.0


def test_model_ks_gini_single_row():
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, 500)
    target = (rng.uniform(0, 1, 500) < score).astype(int)
    out = model_ks_gini(score, target)
    assert len(out) == 1
    assert {"n", "ks", "gini"} <= set(out.columns)
    assert out.loc[0, "gini"] > 0


def test_ks_gini_por_faixa_uma_linha_por_faixa_no_teste():
    rng = np.random.default_rng(2)
    faixas_presentes = ["1-15", "16-30", "31-60"]
    n = 600
    score = rng.uniform(0, 1, n)
    target = (rng.uniform(0, 1, n) < score).astype(int)
    faixa_atraso = rng.choice(faixas_presentes, size=n)
    df = pd.DataFrame({"score": score, "pagou_1parcela": target, "faixa_atraso": faixa_atraso})

    out = ks_gini_por_faixa(df)

    assert len(out) == len(faixas_presentes)
    assert set(out["faixa_atraso"]) == set(faixas_presentes)
    assert out["n"].sum() == n
    assert (out["ks"].dropna() >= 0).all() and (out["ks"].dropna() <= 1).all()
    assert (out["gini"].dropna() >= -1).all() and (out["gini"].dropna() <= 1).all()


def test_decile_table_ordered_by_score_descending():
    rng = np.random.default_rng(1)
    score = rng.uniform(0, 1, 200)
    target = (rng.uniform(0, 1, 200) < score).astype(int)
    saldo = np.full(200, 100.0)
    out = decile_table(score, target, saldo)
    assert list(out["decil"]) == list(range(10, 0, -1))
    # decil 10 (maior score) deve ter taxa de regularizacao >= decil 1
    assert out.loc[out["decil"] == 10, "taxa_reg_contratos"].iloc[0] >= out.loc[
        out["decil"] == 1, "taxa_reg_contratos"
    ].iloc[0]

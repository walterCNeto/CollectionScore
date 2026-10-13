import numpy as np
import pytest

from collscore import FAIXAS
from collscore.simulate import BASE_ROLL_MATRIX, _saldo_devedor, simulate_portfolio


def test_roll_matrix_rows_sum_to_one():
    assert np.allclose(BASE_ROLL_MATRIX.sum(axis=1), 1.0)
    assert BASE_ROLL_MATRIX.shape == (len(FAIXAS), len(FAIXAS))
    assert (BASE_ROLL_MATRIX >= 0).all()


def test_linha_em_dia_calibrada_e_soma_um():
    """Entrada em atraso (em_dia -> 1-15) recalibrada para 0.025/mes
    (inadimplencia nova tipica de varejo, antes 0.05); a linha continua
    renormalizada e somando 1.000.
    """
    linha_em_dia = BASE_ROLL_MATRIX[0]
    assert linha_em_dia[1] == pytest.approx(0.025)
    assert linha_em_dia[0] == pytest.approx(0.975)
    assert linha_em_dia.sum() == pytest.approx(1.0)
    assert (linha_em_dia[2:] == 0).all()


def test_saldo_devedor_mass_conservation():
    """O saldo devedor e sempre exatamente reproduzivel (tabela Price) a
    partir de (valor_financiado, taxa, prazo, parcelas_pagas): no inicio do
    contrato equivale ao valor financiado e, apos todas as parcelas pagas,
    cai a zero -- sem drift de arredondamento acumulado.
    """
    valor_financiado = np.array([10_000.0, 5_000.0, 20_000.0])
    taxa = np.array([0.02, 0.03, 0.015])
    prazo = np.array([24, 12, 48])

    saldo_inicio = _saldo_devedor(
        _parcela(valor_financiado, taxa, prazo), taxa, prazo, np.zeros(3)
    )
    assert np.allclose(saldo_inicio, valor_financiado, rtol=1e-9)

    saldo_fim = _saldo_devedor(_parcela(valor_financiado, taxa, prazo), taxa, prazo, prazo)
    assert np.allclose(saldo_fim, 0.0, atol=1e-6)

    # recalcular duas vezes com os mesmos insumos tem que dar exatamente igual
    k = np.array([10.0, 5.0, 30.0])
    s1 = _saldo_devedor(_parcela(valor_financiado, taxa, prazo), taxa, prazo, k)
    s2 = _saldo_devedor(_parcela(valor_financiado, taxa, prazo), taxa, prazo, k)
    assert np.array_equal(s1, s2)


def _parcela(valor_financiado, taxa, prazo):
    return valor_financiado * taxa / (1.0 - (1.0 + taxa) ** (-prazo))


def test_simulate_portfolio_shape_and_consistency(panel_small):
    df = panel_small
    assert set(df["faixa_atraso"].unique()) <= set(FAIXAS)
    assert (df["saldo_devedor"] >= -1e-6).all()
    assert (df["saldo_atraso"] >= -1e-6).all()
    assert (df["parcelas_pagas"] >= 0).all()
    assert (df["parcelas_pagas"] <= df["prazo"] + 1e-6).all()
    # painel nao deve ter duplicatas de (contrato_id, snapshot)
    assert not df.duplicated(subset=["contrato_id", "snapshot"]).any()
    # snapshot 0 deve existir para todo contrato observado desde o inicio
    assert df["snapshot"].min() == 0
    assert df["snapshot"].max() <= 11


def test_simulate_portfolio_reproducible_with_seed():
    df1 = simulate_portfolio(n_contratos=500, n_snapshots=4, seed=7)
    df2 = simulate_portfolio(n_contratos=500, n_snapshots=4, seed=7)
    pd_cols = ["contrato_id", "snapshot", "faixa_atraso", "saldo_devedor", "parcelas_pagas"]
    assert df1[pd_cols].equals(df2[pd_cols])


def test_simulate_portfolio_size_replenishment():
    df = simulate_portfolio(n_contratos=1000, n_snapshots=6, seed=1)
    counts = df.groupby("snapshot").size()
    # reposicao mensal deve manter o tamanho ativo proximo do alvo (nao
    # encolher continuamente por falta de novas originacoes)
    assert counts.min() > 0.8 * 1000

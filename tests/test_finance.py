import pandas as pd
import pytest

from collscore.finance import SALDO_MINIMO_ELEGIVEL, optimal_cutoff, profit_curve


@pytest.fixture
def caso_pequeno() -> pd.DataFrame:
    # faixa '31-60' -> canal 'humano' (custo=8.00), sem piso de saldo e sem
    # baseline de self-cure (nao esta em FAIXAS_SELF_CURE), logo
    # fator_incremental=1.0 e caixa = target * saldo_atraso (sem haircut).
    scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    targets = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # 3 eventos
    return pd.DataFrame(
        {
            "score": scores,
            "pagou_1parcela": targets,
            "saldo_atraso": [1000.0] * 10,
            "faixa_atraso": ["31-60"] * 10,
        }
    )


def test_profit_curve_matches_manual_calculation(caso_pequeno):
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    curva = curva.set_index("decis_acionados")

    # k=1: 1 contrato acionado (o de maior score), 1 evento -> caixa=1000
    assert curva.loc[1, "n_contratos"] == 1
    assert curva.loc[1, "custo_total"] == pytest.approx(8.00)
    assert curva.loc[1, "caixa_incremental"] == pytest.approx(1000.0)
    assert curva.loc[1, "lucro"] == pytest.approx(992.0)
    assert curva.loc[1, "roi"] == pytest.approx(125.0)

    # k=3: top-3 contem 2 eventos (o 3o contrato tem target=0)
    assert curva.loc[3, "n_contratos"] == 3
    assert curva.loc[3, "custo_total"] == pytest.approx(24.00)
    assert curva.loc[3, "caixa_incremental"] == pytest.approx(2000.0)
    assert curva.loc[3, "lucro"] == pytest.approx(1976.0)

    # k=4: top-4 contem os 3 eventos existentes -> caixa maximo
    assert curva.loc[4, "n_contratos"] == 4
    assert curva.loc[4, "caixa_incremental"] == pytest.approx(3000.0)
    assert curva.loc[4, "lucro"] == pytest.approx(2968.0)

    # k=5: acrescenta 1 contrato sem evento -> apenas custo extra, lucro cai
    assert curva.loc[5, "caixa_incremental"] == pytest.approx(3000.0)
    assert curva.loc[5, "lucro"] == pytest.approx(2960.0)


def test_optimal_cutoff_picks_global_profit_max(caso_pequeno):
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    resultado = optimal_cutoff(curva)

    assert resultado["decis_acionados_otimo"] == 4
    assert resultado["lucro_max"] == pytest.approx(2968.0)
    assert resultado["roi_no_corte"] == pytest.approx(93.75)
    assert resultado["canal"] == "humano"
    assert resultado["saldo_minimo_elegivel"] == 0.0


def test_saldo_minimo_elegivel_exclui_contratos_do_juridico():
    """Faixas atendidas pelo canal juridico tem piso de saldo (default
    R$5000): contratos com saldo em atraso abaixo do piso nao sao
    acionaveis e devem ser totalmente excluidos da curva de lucro.
    """
    assert SALDO_MINIMO_ELEGIVEL["juridico"] == 5000.0

    saldos = [10_000.0, 8_000.0, 6_000.0, 4_000.0, 3_000.0, 1_000.0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    df = pd.DataFrame(
        {
            "score": scores,
            "pagou_1parcela": [1, 1, 1, 1, 1, 1],
            "saldo_atraso": saldos,
            "faixa_atraso": ["91-180"] * 6,
        }
    )

    curva = profit_curve(df, "91-180", n_deciles=10)

    n_elegivel = sum(s >= 5000.0 for s in saldos)
    assert n_elegivel == 3
    assert curva["n_contratos"].max() == n_elegivel
    assert curva["saldo_minimo_elegivel"].iloc[0] == 5000.0

    # nenhum contrato abaixo do piso contribui para o caixa recuperado
    caixa_maximo_possivel = sum(s for s in saldos if s >= 5000.0)
    assert curva["caixa_incremental"].max() <= caixa_maximo_possivel + 1e-6


def test_saldo_minimo_elegivel_zero_para_sms_e_humano():
    df = pd.DataFrame(
        {
            "score": [0.9, 0.1],
            "pagou_1parcela": [1, 0],
            "saldo_atraso": [10.0, 5.0],
            "faixa_atraso": ["1-15", "1-15"],
        }
    )
    curva = profit_curve(df, "1-15", n_deciles=2)
    assert curva["n_contratos"].max() == 2
    assert curva["saldo_minimo_elegivel"].iloc[0] == 0.0

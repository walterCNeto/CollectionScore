import pandas as pd
import pytest

from collscore.finance import optimal_cutoff, profit_curve


@pytest.fixture
def caso_pequeno() -> pd.DataFrame:
    # faixa '31-60' -> canal 'humano' (custo=8.00), haircut=0.80, sem baseline
    # de self-cure (nao esta em FAIXAS_SELF_CURE), logo fator_incremental=1.0.
    scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    targets = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    return pd.DataFrame(
        {
            "score": scores,
            "cura_total": targets,
            "saldo_atraso": [1000.0] * 10,
            "faixa_atraso": ["31-60"] * 10,
        }
    )


def test_profit_curve_matches_manual_calculation(caso_pequeno):
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    curva = curva.set_index("decis_acionados")

    # k=1: 1 contrato acionado (o de maior score), 1 cura -> caixa=1000*0.80=800
    assert curva.loc[1, "n_contratos"] == 1
    assert curva.loc[1, "custo_total"] == pytest.approx(8.00)
    assert curva.loc[1, "caixa_incremental"] == pytest.approx(800.0)
    assert curva.loc[1, "lucro"] == pytest.approx(792.0)
    assert curva.loc[1, "roi"] == pytest.approx(100.0)

    # k=3: top-3 contem 2 curas (o 3o contrato tem target=0)
    assert curva.loc[3, "n_contratos"] == 3
    assert curva.loc[3, "custo_total"] == pytest.approx(24.00)
    assert curva.loc[3, "caixa_incremental"] == pytest.approx(1600.0)
    assert curva.loc[3, "lucro"] == pytest.approx(1576.0)

    # k=4: top-4 contem as 4 curas existentes -> caixa maximo
    assert curva.loc[4, "n_contratos"] == 4
    assert curva.loc[4, "caixa_incremental"] == pytest.approx(2400.0)
    assert curva.loc[4, "lucro"] == pytest.approx(2368.0)

    # k=5: acrescenta 1 contrato sem cura -> apenas custo extra, lucro cai
    assert curva.loc[5, "caixa_incremental"] == pytest.approx(2400.0)
    assert curva.loc[5, "lucro"] == pytest.approx(2360.0)


def test_optimal_cutoff_picks_global_profit_max(caso_pequeno):
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    resultado = optimal_cutoff(curva)

    assert resultado["decis_acionados_otimo"] == 4
    assert resultado["lucro_max"] == pytest.approx(2368.0)
    assert resultado["roi_no_corte"] == pytest.approx(75.0)
    assert resultado["canal"] == "humano"

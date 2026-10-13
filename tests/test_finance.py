import numpy as np
import pandas as pd
import pytest

from collscore.finance import (
    CANAL_CUSTOS,
    CANAL_POR_FAIXA,
    EFETIVIDADE,
    SALDO_MINIMO_ELEGIVEL,
    compare_channels,
    fronteira_orcamento_lucro,
    optimal_cutoff,
    optimal_strategy,
    profit_curve,
    run_finance_analysis,
)


@pytest.fixture
def caso_pequeno() -> pd.DataFrame:
    # faixa '31-60' -> canal default 'negativacao'; EFETIVIDADE['negativacao']['31-60']
    # = 0.65. Sem piso de saldo e sem baseline de self-cure (nao esta em
    # FAIXAS_SELF_CURE), logo fator_incremental=1.0 e caixa_incremental =
    # target * saldo_atraso * efetividade_da_celula.
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
    assert EFETIVIDADE["negativacao"]["31-60"] == pytest.approx(0.65)

    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    curva = curva.set_index("decis_acionados")

    # k=1: 1 contrato acionado (o de maior score), 1 evento
    # caixa_incremental = 1000 * 0.65 = 650
    assert curva.loc[1, "n_contratos"] == 1
    assert curva.loc[1, "canal"] == "negativacao"
    assert curva.loc[1, "efetividade"] == pytest.approx(0.65)
    assert curva.loc[1, "custo_total"] == pytest.approx(3.00)
    assert curva.loc[1, "caixa_incremental"] == pytest.approx(650.0)
    assert curva.loc[1, "lucro"] == pytest.approx(647.0)
    assert curva.loc[1, "roi"] == pytest.approx(216.667)

    # k=3: top-3 contem 2 eventos (o 3o contrato tem target=0)
    assert curva.loc[3, "n_contratos"] == 3
    assert curva.loc[3, "custo_total"] == pytest.approx(9.00)
    assert curva.loc[3, "caixa_incremental"] == pytest.approx(1300.0)
    assert curva.loc[3, "lucro"] == pytest.approx(1291.0)

    # k=4: top-4 contem os 3 eventos existentes -> caixa maximo
    assert curva.loc[4, "n_contratos"] == 4
    assert curva.loc[4, "custo_total"] == pytest.approx(12.00)
    assert curva.loc[4, "caixa_incremental"] == pytest.approx(1950.0)
    assert curva.loc[4, "lucro"] == pytest.approx(1938.0)

    # k=5: acrescenta 1 contrato sem evento -> apenas custo extra, lucro cai
    assert curva.loc[5, "caixa_incremental"] == pytest.approx(1950.0)
    assert curva.loc[5, "lucro"] == pytest.approx(1935.0)


def test_optimal_cutoff_picks_global_profit_max(caso_pequeno):
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    resultado = optimal_cutoff(curva)

    assert resultado["decis_acionados_otimo"] == 4
    assert resultado["lucro_max"] == pytest.approx(1938.0)
    assert resultado["roi_no_corte"] == pytest.approx(162.5)
    assert resultado["canal"] == "negativacao"
    assert resultado["saldo_minimo_elegivel"] == 0.0
    assert resultado["efetividade"] == pytest.approx(0.65)


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


def test_saldo_minimo_elegivel_zero_para_canais_baratos():
    assert SALDO_MINIMO_ELEGIVEL["sms"] == 0.0
    assert SALDO_MINIMO_ELEGIVEL["negativacao"] == 0.0
    assert SALDO_MINIMO_ELEGIVEL["call_center"] == 0.0

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


# --------------------------------------------------------------- negativacao


def test_mapeamento_default_usa_negativacao_em_31_60_e_61_90():
    assert CANAL_POR_FAIXA["31-60"] == "negativacao"
    assert CANAL_POR_FAIXA["61-90"] == "negativacao"
    # os canais nas pontas da escada de custo nao mudaram
    assert CANAL_POR_FAIXA["1-15"] == "sms"
    assert CANAL_POR_FAIXA["16-30"] == "sms"
    assert CANAL_POR_FAIXA["91-180"] == "juridico"
    assert CANAL_POR_FAIXA["180+"] == "juridico"
    assert "negativacao" in CANAL_CUSTOS
    assert "negativacao" in EFETIVIDADE


def test_custo_negativacao_entra_corretamente_na_curva(caso_pequeno):
    """O custo total acionado deve ser exatamente n_contratos * custo do
    canal (R$3.00 por inclusao, default), independente da efetividade.
    """
    curva = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    assert (curva["custo_total"] == curva["n_contratos"] * CANAL_CUSTOS["negativacao"]).all()

    # com um custo customizado, o total deve escalar proporcionalmente
    curva_custom = profit_curve(
        caso_pequeno, "31-60", n_deciles=10, canal_custos={"negativacao": 10.0}
    )
    assert (curva_custom["custo_total"] == curva_custom["n_contratos"] * 10.0).all()


def test_efetividade_zero_zera_caixa_incremental(caso_pequeno):
    """Efetividade 0.0 para a celula (canal, faixa) deve anular todo o
    caixa incremental (e portanto o lucro passa a ser so o negativo do
    custo de acionamento), mantendo o custo normal.
    """
    curva_zero = profit_curve(
        caso_pequeno, "31-60", n_deciles=10, efetividade={"negativacao": {"31-60": 0.0}}
    )
    assert (curva_zero["caixa_incremental"] == 0.0).all()
    assert (curva_zero["lucro"] == -curva_zero["custo_total"]).all()
    assert (curva_zero["efetividade"] == 0.0).all()

    # custo nao e afetado pela efetividade
    curva_base = profit_curve(caso_pequeno, "31-60", n_deciles=10)
    assert (curva_zero["custo_total"] == curva_base["custo_total"]).all()


# -------------------------------------------------------- rename humano -> call_center


def test_nenhum_canal_se_chama_humano():
    for canal in CANAL_CUSTOS:
        assert canal != "humano"
    for canal in SALDO_MINIMO_ELEGIVEL:
        assert canal != "humano"
    for canal in EFETIVIDADE:
        assert canal != "humano"
    for canal in CANAL_POR_FAIXA.values():
        assert canal != "humano"
    assert "call_center" in CANAL_CUSTOS


# --------------------------------------------------- matriz de efetividade canal x faixa


def test_matriz_efetividade_tem_as_4_faixas_e_6_canais():
    canais_esperados = {"sms", "negativacao", "call_center", "juridico"}
    faixas_esperadas = {"1-15", "16-30", "31-60", "61-90", "91-180", "180+"}
    assert set(EFETIVIDADE.keys()) == canais_esperados
    for canal in canais_esperados:
        assert set(EFETIVIDADE[canal].keys()) == faixas_esperadas
        for faixa, valor in EFETIVIDADE[canal].items():
            assert 0.0 <= valor <= 1.0, (canal, faixa, valor)


def test_juridico_lucro_zero_ou_negativo_em_faixas_com_efetividade_zero():
    """Em 1-15, 16-30 e 31-60 a efetividade de `juridico` e 0.0 (torna o
    canal inelegivel NA PRATICA, mesmo que o piso de saldo seja ignorado):
    o caixa incremental deve ser zero e o lucro, portanto, <= 0 (= -custo)
    em qualquer corte.
    """
    faixas_efetividade_zero = ["1-15", "16-30", "31-60"]
    for faixa in faixas_efetividade_zero:
        assert EFETIVIDADE["juridico"][faixa] == 0.0

    saldos = [10_000.0, 9_000.0, 8_000.0, 7_000.0]  # acima do piso de juridico
    scores = [0.9, 0.8, 0.7, 0.6]
    for faixa in faixas_efetividade_zero:
        df = pd.DataFrame(
            {
                "score": scores,
                "pagou_1parcela": [1, 1, 1, 1],
                "saldo_atraso": saldos,
                "faixa_atraso": [faixa] * 4,
            }
        )
        curva = profit_curve(df, faixa, canal_por_faixa={faixa: "juridico"})
        assert not curva.empty, faixa
        assert (curva["caixa_incremental"] == 0.0).all(), faixa
        assert (curva["lucro"] <= 0.0).all(), faixa
        assert (curva["lucro"] == -curva["custo_total"]).all(), faixa


def test_juridico_lucro_positivo_possivel_em_faixas_profundas():
    """Em 61-90, 91-180 e 180+ a efetividade de juridico e > 0 -- o canal
    pode, em princípio, gerar lucro positivo (nao e estruturalmente zerado
    como nas faixas iniciais).
    """
    for faixa in ["61-90", "91-180", "180+"]:
        assert EFETIVIDADE["juridico"][faixa] > 0.0


# -------------------------------------------------------- escolha endogena de canal


@pytest.fixture
def book_multi_faixa() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    faixas = rng.choice(["1-15", "16-30", "31-60", "61-90", "91-180", "180+"], size=n)
    df = pd.DataFrame(
        {
            "score": rng.uniform(0, 1, n),
            "pagou_1parcela": (rng.uniform(0, 1, n) < 0.3).astype(int),
            "saldo_atraso": rng.uniform(100, 15_000, n),
            "faixa_atraso": faixas,
        }
    )
    df["valor_recuperado"] = df["pagou_1parcela"] * rng.uniform(50, 300, n)
    return df


def test_compare_channels_uma_linha_por_canal_elegivel(book_multi_faixa):
    cc = compare_channels(book_multi_faixa, "31-60")
    assert set(cc["canal"]) == set(CANAL_CUSTOS.keys())
    assert len(cc) == len(CANAL_CUSTOS)
    assert (cc["faixa_atraso"] == "31-60").all()


def test_optimal_strategy_lucro_nao_perde_para_mapeamento_manual(book_multi_faixa):
    """A estrategia endogena por lucro escolhe, para cada faixa, o melhor
    canal dentre TODOS os canais -- incluindo o canal do mapeamento manual.
    Logo o lucro total nunca pode ser menor que o do mapeamento manual.
    """
    _, resumo_manual = run_finance_analysis(book_multi_faixa)
    lucro_manual = resumo_manual["lucro_max"].sum()

    estrategia = optimal_strategy(book_multi_faixa, criterio="lucro")

    assert estrategia["lucro_total"] >= lucro_manual - 1e-6
    assert estrategia["criterio"] == "lucro"
    assert not estrategia["estrategia"].empty


def test_optimal_strategy_roi_prioriza_canais_mais_baratos(book_multi_faixa):
    estrategia_roi = optimal_strategy(book_multi_faixa, criterio="roi")
    estrategia_lucro = optimal_strategy(book_multi_faixa, criterio="lucro")

    # otimizar por ROI tende a favorecer canais de custo baixo (ROI por
    # unidade acionada e maior quando o denominador -- custo -- e pequeno)
    assert estrategia_roi["roi_agregado"] >= estrategia_lucro["roi_agregado"]


def test_fronteira_orcamento_lucro_tem_dois_pontos(book_multi_faixa):
    fronteira = fronteira_orcamento_lucro(book_multi_faixa)
    assert set(fronteira["criterio"]) == {"roi", "lucro"}
    assert len(fronteira) == 2
    assert {"custo_total", "lucro_total", "roi_agregado"} <= set(fronteira.columns)

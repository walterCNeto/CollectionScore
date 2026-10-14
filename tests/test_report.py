import re

import pandas as pd
import pytest

from collscore.features import FEATURE_COLS, build_features
from collscore.finance import CANAL_CUSTOS, CANAL_POR_FAIXA, EFETIVIDADE, SALDO_MINIMO_ELEGIVEL
from collscore.metrics import (
    decile_table,
    ks_gini_por_faixa,
    model_ks_gini,
    resumo_regularizacao,
    roll_rate_saldo,
    taxas_evento_por_faixa,
)
from collscore.model import train_pooled
from collscore.report import build_html_report
from collscore.simulate import simulate_portfolio
from collscore.target import compute_targets

TARGET_COL = "pagou_1parcela"

SECOES_ESPERADAS = [
    "1. Objetivo e escopo",
    "2. Base de dados",
    "3. Target e janela de performance",
    "4. Book de variaveis",
    "5. Metodologia",
    "6. Desempenho",
    "7. Estabilidade e dinamica",
    "8. Uso do score",
    "9. Limitacoes e recomendacoes",
]


def _train_test_split(book: pd.DataFrame, test_frac: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    contratos = book["contrato_id"].unique()
    rng.shuffle(contratos)
    n_test = int(len(contratos) * test_frac)
    contratos_test = set(contratos[:n_test])
    is_test = book["contrato_id"].isin(contratos_test)
    return book.loc[~is_test].copy(), book.loc[is_test].copy()


def _montar_resultados_e_parametros(n_contratos: int, n_snapshots: int, seed: int) -> tuple[dict, dict]:
    """Constroi (resultados, parametros) diretamente a partir dos modulos do
    pacote, espelhando `app._montar_pipeline` sem depender de importar o
    modulo `app` (que executa codigo de sidebar Streamlit na importacao).
    """
    burn_in = n_snapshots // 2
    panel = simulate_portfolio(n_contratos=n_contratos, n_snapshots=n_snapshots, seed=seed)
    panel_pos_burnin = panel[panel["snapshot"] >= burn_in]

    feats = build_features(panel)
    targets = compute_targets(panel)

    book_full = feats.merge(
        targets[["contrato_id", "snapshot", "pagou_1parcela", "cura_total", "melhora_faixa"]],
        on=["contrato_id", "snapshot"],
    )
    book_full = book_full[book_full["snapshot"] >= burn_in]
    book = book_full[book_full["faixa_atraso"] != "em_dia"].copy()
    book_elegivel = book.dropna(subset=[TARGET_COL]).copy()
    book_elegivel["valor_recuperado"] = book_elegivel[TARGET_COL] * book_elegivel["parcela_valor"]

    train, test = _train_test_split(book_elegivel, test_frac=0.3, seed=seed)
    modelo = train_pooled(train, FEATURE_COLS, TARGET_COL)
    test = test.copy()
    test["score"] = modelo.predict_proba(test)

    resultados = dict(
        resumo_exec=resumo_regularizacao(book_elegivel, target_col=TARGET_COL),
        rr_saldo=roll_rate_saldo(panel_pos_burnin),
        ks_gini=model_ks_gini(test["score"].to_numpy(), test[TARGET_COL].to_numpy()),
        ks_gini_faixa=ks_gini_por_faixa(test, score_col="score", target_col=TARGET_COL),
        decile=decile_table(
            test["score"].to_numpy(),
            test[TARGET_COL].to_numpy(),
            test["saldo_atraso"].to_numpy(),
            test["valor_recuperado"].to_numpy(),
        ),
        test=test,
        targets_alternativos=taxas_evento_por_faixa(book, ["pagou_1parcela", "cura_total", "melhora_faixa"]),
        feature_describe=book_elegivel[FEATURE_COLS].describe(percentiles=[0.25, 0.5, 0.75]).T.round(3),
        volumetria_snapshot=(
            panel_pos_burnin.assign(delinquente=(panel_pos_burnin["faixa_atraso"] != "em_dia").astype(float))
            .groupby("snapshot")
            .agg(n_contratos=("contrato_id", "size"), pct_delinquente=("delinquente", "mean"))
            .round(3)
            .reset_index()
        ),
        n_train=len(train),
        n_test=len(test),
    )

    parametros = dict(
        fonte="Base de exemplo (simulada)",
        n_contratos=n_contratos,
        n_snapshots=n_snapshots,
        burn_in=burn_in,
        seed=seed,
        target_col=TARGET_COL,
        canal_custos=dict(CANAL_CUSTOS),
        estrategia_modo="Automatica (maximizar lucro)",
        canal_por_faixa_manual=dict(CANAL_POR_FAIXA),
        saldo_minimo_elegivel=dict(SALDO_MINIMO_ELEGIVEL),
        efetividade={canal: dict(faixas) for canal, faixas in EFETIVIDADE.items()},
    )
    return resultados, parametros


@pytest.fixture(scope="module")
def resultados_parametros_padrao():
    return _montar_resultados_e_parametros(n_contratos=3000, n_snapshots=10, seed=11)


def test_build_html_report_contem_9_secoes_e_imagens(resultados_parametros_padrao):
    resultados, parametros = resultados_parametros_padrao
    html = build_html_report(resultados, parametros)

    assert html.strip().lower().startswith("<!doctype html>")
    assert "</html>" in html.lower()

    for secao in SECOES_ESPERADAS:
        assert secao in html, f"secao ausente: {secao}"

    n_imagens = len(re.findall(r"data:image/png;base64,", html))
    assert n_imagens >= 2, f"esperava pelo menos 2 imagens base64, achou {n_imagens}"


def test_build_html_report_funciona_com_base_minima():
    """Base minima mas ainda operavel pelo pipeline (poucos contratos/poucos
    snapshots) -- build_html_report nao deve lancar excecao e ainda deve
    conter as 9 secoes.
    """
    resultados, parametros = _montar_resultados_e_parametros(n_contratos=600, n_snapshots=6, seed=3)

    html = build_html_report(resultados, parametros)

    for secao in SECOES_ESPERADAS:
        assert secao in html, f"secao ausente (base minima): {secao}"
    assert len(re.findall(r"data:image/png;base64,", html)) >= 2


def test_build_html_report_numeros_com_3_casas_decimais(resultados_parametros_padrao):
    resultados, parametros = resultados_parametros_padrao
    html = build_html_report(resultados, parametros)

    # procura floats com 1 ou 2 casas decimais fora de contexto de data/hora
    # (uma amostra grosseira: nenhum numero do tipo "X.Y" ou "X.YY" solto em
    # uma celula de tabela -- heuristica: todo float em celula <td> deve ter
    # exatamente 3 casas decimais)
    celulas_float = re.findall(r"<td>(-?\d+\.\d+)</td>", html)
    for valor in celulas_float:
        casas = len(valor.split(".")[1])
        assert casas == 3, f"valor com {casas} casas decimais: {valor}"

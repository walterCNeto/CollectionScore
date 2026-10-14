"""Pipeline ponta a ponta do collscore: simula -> features -> target
(pagou_1parcela) -> treina modelo pooled -> metricas de regularizacao
financeira -> financas (manual e estrategia endogena de canal) ->
relatorio.md.

Aplica burn-in: descarta os snapshots 0-5 (carteira ainda estabilizando a
distribuicao de faixas apos a reposicao mensal) e treina/mede apenas nos
snapshots 6-11.

Uso: python run_demo.py [--n-contratos 50000] [--seed 42] [--output relatorio.md]
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd

from collscore.features import FEATURE_COLS, build_features
from collscore.finance import (
    CANAL_CUSTOS,
    CANAL_POR_FAIXA,
    EFETIVIDADE,
    SALDO_MINIMO_ELEGIVEL,
    fronteira_orcamento_lucro,
    optimal_strategy,
    run_channel_comparison,
    run_finance_analysis,
)
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
BURN_IN_SNAPSHOT = 6


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sem dados)_\n"
    return df.to_markdown(index=False) + "\n"


def train_test_split_book(book: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    contratos = book["contrato_id"].unique()
    rng.shuffle(contratos)
    n_test = int(len(contratos) * test_frac)
    contratos_test = set(contratos[:n_test])
    is_test = book["contrato_id"].isin(contratos_test)
    return book.loc[~is_test].copy(), book.loc[is_test].copy()


def main(
    n_contratos: int = 50_000,
    n_snapshots: int = 12,
    seed: int = 42,
    output: str = "relatorio.md",
    html_output: str = "relatorio_modelo.html",
):
    buf = io.StringIO()

    def w(txt: str = ""):
        buf.write(txt + "\n")

    print("Simulando carteira...")
    panel = simulate_portfolio(n_contratos=n_contratos, n_snapshots=n_snapshots, seed=seed)
    panel_pos_burnin = panel[panel["snapshot"] >= BURN_IN_SNAPSHOT].copy()

    print("Construindo features...")
    feats = build_features(panel)

    print("Calculando target pagou_1parcela...")
    targets = compute_targets(panel)

    book_full = feats.merge(
        targets[["contrato_id", "snapshot", "pagou_1parcela", "cura_total", "melhora_faixa"]],
        on=["contrato_id", "snapshot"],
    )
    book_full = book_full[book_full["snapshot"] >= BURN_IN_SNAPSHOT]
    book = book_full[book_full["faixa_atraso"] != "em_dia"].copy()
    book_elegivel = book.dropna(subset=[TARGET_COL]).copy()
    book_elegivel["valor_recuperado"] = book_elegivel[TARGET_COL] * book_elegivel["parcela_valor"]

    train, test = train_test_split_book(book_elegivel, test_frac=0.3, seed=seed)

    # --------------------------------------------------------------- treino
    print("Treinando modelo pooled (target pagou_1parcela)...")
    modelo = train_pooled(train, FEATURE_COLS, TARGET_COL)

    test = test.copy()
    test["score"] = modelo.predict_proba(test)

    # -------------------------------------------------------- metricas/financas
    print("Calculando metricas de regularizacao e curva de lucro...")
    resumo_exec = resumo_regularizacao(book_elegivel, target_col=TARGET_COL)
    rr_saldo = roll_rate_saldo(panel_pos_burnin)

    dec_contrato = decile_table(
        test["score"].to_numpy(),
        test[TARGET_COL].to_numpy(),
        test["saldo_atraso"].to_numpy(),
        test["valor_recuperado"].to_numpy(),
    )

    curva_lucro, resumo_otimo = run_finance_analysis(test, score_col="score", target_col=TARGET_COL)

    ks_gini = model_ks_gini(test["score"].to_numpy(), test[TARGET_COL].to_numpy())
    ks_gini_faixa = ks_gini_por_faixa(test, score_col="score", target_col=TARGET_COL)

    # tabela executiva: saldo em atraso | taxa_reg_contratos | taxa_reg_caixa | custo acionamento | ROI
    exec_fin = resumo_exec.merge(
        resumo_otimo[["faixa_atraso", "canal", "custo_total", "roi_no_corte"]],
        on="faixa_atraso",
        how="left",
    )
    exec_fin = exec_fin[
        ["faixa_atraso", "saldo_atraso_total", "taxa_reg_contratos", "taxa_reg_caixa", "canal", "custo_total", "roi_no_corte"]
    ].rename(columns={"custo_total": "custo_acionamento", "roi_no_corte": "roi"})

    print("Comparando canais e montando estrategias endogenas...")
    comparacao_canais = run_channel_comparison(test, score_col="score", target_col=TARGET_COL)
    estrategia_lucro = optimal_strategy(test, criterio="lucro", score_col="score", target_col=TARGET_COL)
    estrategia_roi = optimal_strategy(test, criterio="roi", score_col="score", target_col=TARGET_COL)
    fronteira = fronteira_orcamento_lucro(test, score_col="score", target_col=TARGET_COL)

    # --------------------------------------------------------------- relatorio
    w("# Relatorio collscore")
    w()
    w(
        f"Carteira simulada: `n_contratos={n_contratos}`, `n_snapshots={n_snapshots}`, `seed={seed}`. "
        f"Burn-in: snapshots 0-{BURN_IN_SNAPSHOT - 1} descartados; treino e avaliacao em snapshots "
        f"{BURN_IN_SNAPSHOT}-{n_snapshots - 1}. Target: `{TARGET_COL}`."
    )
    w()

    w("## (a) Tabela executiva por faixa")
    w()
    w(
        "Saldo em atraso, taxa de regularizacao (contratos e caixa), canal, "
        "custo de acionamento e ROI no corte otimo (linha TOTAL sem canal/custo/ROI, "
        "pois a estrategia e definida por faixa; canal/custo/ROI aqui seguem o "
        "mapeamento MANUAL de `CANAL_POR_FAIXA` -- ver secao (f) para a estrategia "
        "endogena de canal)."
    )
    w()
    w(md_table(exec_fin))
    w(
        "**Definicao:** caixa recuperado = 1 parcela x evento `pagou_1parcela`; "
        "`taxa_reg_caixa` e estruturalmente decrescente com a faixa (o estoque em "
        "atraso acumula multiplas parcelas, enquanto o caixa recuperado por evento "
        "continua valendo so 1 parcela)."
    )
    w()

    w("## (b) Roll rate de saldo M -> M+1")
    w()
    w("Fracao do saldo devedor (R$) na faixa de origem (linha) que migra para cada faixa de destino (coluna) no mes seguinte.")
    w()
    w(md_table(rr_saldo.reset_index()))

    w(f"## (c) Decis de score com caixa recuperado (conjunto de teste, target `{TARGET_COL}`)")
    w()
    w(md_table(dec_contrato))

    w("## (d) Curva de lucro e corte otimo por canal (mapeamento manual, com piso de saldo elegivel)")
    w()
    w("Curva completa (lucro acumulado por numero de decis acionados):")
    w()
    w(md_table(curva_lucro))
    w("Corte otimo por faixa/canal (maximiza lucro):")
    w()
    w(md_table(resumo_otimo))
    w(
        "O score define corte apenas onde o custo de acionamento e material "
        "(juridico); nos canais baratos ele apenas ordena a fila."
    )
    w()

    w("## (e) KS / Gini do modelo (conjunto de teste)")
    w()
    w("Global (mistura faixas com taxas de evento bem diferentes -- infla o numero em relacao ao poder discriminante dentro de cada faixa):")
    w()
    w(md_table(ks_gini))
    w("Por faixa (mesmo score, sem retreinar nada por faixa):")
    w()
    w(md_table(ks_gini_faixa))

    w("## (f) Escolha endogena de canal: comparacao e estrategias otimas")
    w()
    w(
        "Comparacao canal x faixa: para cada faixa, o resultado de aciona-la por "
        "CADA canal elegivel (independente do mapeamento fixo de (a)/(d))."
    )
    w()
    w(md_table(comparacao_canais))
    w(
        "As estrategias 'lucro' e 'roi' sao dois pontos extremos de uma fronteira "
        "orcamento x lucro: ROI gasta pouco e maximiza retorno por R$ investido; "
        "lucro gasta mais para maximizar o retorno absoluto. Totais lado a lado:"
    )
    w()
    w(md_table(fronteira))
    w("Estrategia otima por lucro (canal vencedor por faixa):")
    w()
    w(md_table(estrategia_lucro["estrategia"]))
    w("Estrategia otima por ROI (canal vencedor por faixa):")
    w()
    w(md_table(estrategia_roi["estrategia"]))

    with open(output, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Relatorio escrito em {output}")

    # ------------------------------------------------- relatorio HTML de validacao
    print("Gerando relatorio HTML de desenvolvimento e validacao...")
    targets_alternativos = taxas_evento_por_faixa(book, ["pagou_1parcela", "cura_total", "melhora_faixa"])
    feature_describe = book_elegivel[FEATURE_COLS].describe(percentiles=[0.25, 0.5, 0.75]).T.round(3)
    volumetria_snapshot = (
        panel_pos_burnin.assign(delinquente=(panel_pos_burnin["faixa_atraso"] != "em_dia").astype(float))
        .groupby("snapshot")
        .agg(n_contratos=("contrato_id", "size"), pct_delinquente=("delinquente", "mean"))
        .round(3)
        .reset_index()
    )

    resultados_html = dict(
        resumo_exec=resumo_exec,
        rr_saldo=rr_saldo,
        ks_gini=ks_gini,
        ks_gini_faixa=ks_gini_faixa,
        decile=dec_contrato,
        test=test,
        targets_alternativos=targets_alternativos,
        feature_describe=feature_describe,
        volumetria_snapshot=volumetria_snapshot,
        n_train=len(train),
        n_test=len(test),
    )
    parametros_html = dict(
        fonte="Base de exemplo (simulada)",
        n_contratos=n_contratos,
        n_snapshots=n_snapshots,
        burn_in=BURN_IN_SNAPSHOT,
        seed=seed,
        target_col=TARGET_COL,
        canal_custos=dict(CANAL_CUSTOS),
        estrategia_modo="Automatica (maximizar lucro)",
        canal_por_faixa_manual=dict(CANAL_POR_FAIXA),
        saldo_minimo_elegivel=dict(SALDO_MINIMO_ELEGIVEL),
        efetividade={canal: dict(faixas) for canal, faixas in EFETIVIDADE.items()},
    )
    html = build_html_report(resultados_html, parametros_html)
    with open(html_output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Relatorio HTML escrito em {html_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-contratos", type=int, default=50_000)
    parser.add_argument("--n-snapshots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="relatorio.md")
    parser.add_argument("--html-output", type=str, default="relatorio_modelo.html")
    args = parser.parse_args()
    main(
        n_contratos=args.n_contratos,
        n_snapshots=args.n_snapshots,
        seed=args.seed,
        output=args.output,
        html_output=args.html_output,
    )

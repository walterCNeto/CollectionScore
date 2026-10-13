"""Pipeline ponta a ponta do collscore: simula -> features -> target
(pagou_1parcela) -> treina modelo pooled -> metricas de regularizacao
financeira -> financas -> relatorio.md.

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
from collscore.finance import run_finance_analysis
from collscore.metrics import decile_table, model_ks_gini, resumo_regularizacao, roll_rate_saldo
from collscore.model import train_pooled
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


def main(n_contratos: int = 50_000, n_snapshots: int = 12, seed: int = 42, output: str = "relatorio.md"):
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

    book_full = feats.merge(targets[["contrato_id", "snapshot", TARGET_COL]], on=["contrato_id", "snapshot"])
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

    # tabela executiva: saldo em atraso | taxa_reg_contratos | taxa_reg_caixa | custo acionamento | ROI
    exec_fin = resumo_exec.merge(
        resumo_otimo[["faixa_atraso", "canal", "custo_total", "roi_no_corte"]],
        on="faixa_atraso",
        how="left",
    )
    exec_fin = exec_fin[
        ["faixa_atraso", "saldo_atraso_total", "taxa_reg_contratos", "taxa_reg_caixa", "canal", "custo_total", "roi_no_corte"]
    ].rename(columns={"custo_total": "custo_acionamento", "roi_no_corte": "roi"})

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
        "pois a estrategia e definida por faixa)."
    )
    w()
    w(md_table(exec_fin))

    w("## (b) Roll rate de saldo M -> M+1")
    w()
    w("Fracao do saldo devedor (R$) na faixa de origem (linha) que migra para cada faixa de destino (coluna) no mes seguinte.")
    w()
    w(md_table(rr_saldo.reset_index()))

    w(f"## (c) Decis de score com caixa recuperado (conjunto de teste, target `{TARGET_COL}`)")
    w()
    w(md_table(dec_contrato))

    w("## (d) Curva de lucro e corte otimo por canal (com piso de saldo elegivel)")
    w()
    w("Curva completa (lucro acumulado por numero de decis acionados):")
    w()
    w(md_table(curva_lucro))
    w("Corte otimo por faixa/canal (maximiza lucro):")
    w()
    w(md_table(resumo_otimo))

    w("## (e) KS / Gini globais do modelo (conjunto de teste)")
    w()
    w(md_table(ks_gini))

    with open(output, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Relatorio escrito em {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-contratos", type=int, default=50_000)
    parser.add_argument("--n-snapshots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="relatorio.md")
    args = parser.parse_args()
    main(n_contratos=args.n_contratos, n_snapshots=args.n_snapshots, seed=args.seed, output=args.output)

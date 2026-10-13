"""Pipeline ponta a ponta do collscore: simula -> features -> targets ->
treina (por faixa e pooled) -> metricas -> financas -> relatorio.md.

Uso: python run_demo.py [--n-contratos 50000] [--seed 42] [--output relatorio.md]
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd

from collscore import FAIXAS
from collscore.features import FEATURE_COLS, build_features
from collscore.finance import run_finance_analysis
from collscore.metrics import (
    decile_table,
    decile_table_cash,
    metrics_by_faixa,
    psi_between_snapshots,
)
from collscore.model import train_per_faixa, train_pooled
from collscore.simulate import simulate_portfolio
from collscore.target import TARGET_COLS, compute_targets

PRIMARY_TARGET = "cura_total"


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


def score_per_faixa(modelos: dict, df: pd.DataFrame, faixa_col: str = "faixa_atraso") -> np.ndarray:
    score = np.full(len(df), np.nan)
    for faixa, fm in modelos.items():
        mask = (df[faixa_col] == faixa).to_numpy()
        if mask.any():
            score[mask] = fm.predict_proba(df.loc[mask])
    return score


def main(n_contratos: int = 50_000, n_snapshots: int = 12, seed: int = 42, output: str = "relatorio.md"):
    buf = io.StringIO()

    def w(txt: str = ""):
        buf.write(txt + "\n")

    print("Simulando carteira...")
    panel = simulate_portfolio(n_contratos=n_contratos, n_snapshots=n_snapshots, seed=seed)

    print("Construindo features...")
    feats = build_features(panel)

    print("Calculando targets...")
    targets = compute_targets(panel)

    book_full = feats.merge(targets, on=["contrato_id", "snapshot"])
    book = book_full[book_full["faixa_atraso"] != "em_dia"].copy()

    train, test = train_test_split_book(book, test_frac=0.3, seed=seed)

    # --------------------------------------------------------------- treino
    print("Treinando modelos (por faixa e pooled) para os 3 targets...")
    resultados_targets = {}
    for target_col in TARGET_COLS:
        train_t = train.dropna(subset=[target_col])
        test_t = test.dropna(subset=[target_col])

        modelos_faixa = train_per_faixa(train_t, FEATURE_COLS, target_col)
        modelo_pooled = train_pooled(train_t, FEATURE_COLS, target_col)

        test_t = test_t.copy()
        test_t["score_pooled"] = modelo_pooled.predict_proba(test_t)
        test_t["score_faixa"] = score_per_faixa(modelos_faixa, test_t)

        met_pooled = metrics_by_faixa(test_t, "score_pooled", target_col).add_suffix("_pooled")
        met_faixa = metrics_by_faixa(
            test_t.dropna(subset=["score_faixa"]), "score_faixa", target_col
        ).add_suffix("_faixa")

        comparacao = met_pooled.merge(
            met_faixa, left_on="faixa_atraso_pooled", right_on="faixa_atraso_faixa", how="outer"
        )
        comparacao["faixa_atraso"] = comparacao["faixa_atraso_pooled"].fillna(comparacao["faixa_atraso_faixa"])
        comparacao = comparacao[
            ["faixa_atraso", "n_pooled", "auc_pooled", "auc_faixa", "ks_pooled", "ks_faixa", "gini_pooled", "gini_faixa"]
        ]

        resultados_targets[target_col] = dict(
            modelos_faixa=modelos_faixa,
            modelo_pooled=modelo_pooled,
            test_t=test_t,
            comparacao=comparacao,
        )

    # ----------------------------------------------------- financas (cura_total)
    print("Rodando analise financeira...")
    primary = resultados_targets[PRIMARY_TARGET]
    test_primary = primary["test_t"].copy()
    test_primary["score"] = test_primary["score_faixa"].fillna(test_primary["score_pooled"])

    dec_contrato = decile_table(
        test_primary["score"].to_numpy(),
        test_primary[PRIMARY_TARGET].to_numpy(),
        test_primary["saldo_atraso"].to_numpy(),
    )
    dec_caixa = decile_table_cash(
        test_primary["score"].to_numpy(),
        test_primary[PRIMARY_TARGET].to_numpy(),
        test_primary["saldo_atraso"].to_numpy(),
    )

    curva_lucro, resumo_otimo = run_finance_analysis(test_primary, score_col="score", target_col=PRIMARY_TARGET)

    # score pooled aplicado ao book completo (todas as fotos) para PSI
    scored_full = book.dropna(subset=[PRIMARY_TARGET]).copy()
    scored_full["score_pooled"] = primary["modelo_pooled"].predict_proba(scored_full)
    psi_df = psi_between_snapshots(scored_full, "score_pooled")

    # --------------------------------------------------------------- relatorio
    w("# Relatorio collscore")
    w()
    w(f"Carteira simulada: `n_contratos={n_contratos}`, `n_snapshots={n_snapshots}`, `seed={seed}`.")
    w()

    w("## 1. Resumo da simulacao")
    w()
    panel_tmp = panel.assign(delinquente=(panel["faixa_atraso"] != "em_dia").astype(float))
    resumo_sim = (
        panel_tmp.groupby("snapshot")
        .agg(
            n_contratos=("contrato_id", "size"),
            pct_delinquente=("delinquente", "mean"),
            saldo_devedor_total=("saldo_devedor", "sum"),
            saldo_atraso_total=("saldo_atraso", "sum"),
        )
        .round(3)
        .reset_index()
    )
    w(md_table(resumo_sim))

    w("## 2. Targets de collection scoring (book delinquente, janela M->M+1)")
    w()
    w("Media dos 3 targets no book completo (todas as fotos, contratos delinquentes em M):")
    w()
    medias_targets = book[TARGET_COLS].mean().round(3).rename("media").reset_index().rename(columns={"index": "target"})
    w(md_table(medias_targets))

    w("## 3. Comparacao por faixa vs pooled (AUC / KS / Gini no conjunto de teste)")
    w()
    for target_col in TARGET_COLS:
        w(f"### Target: `{target_col}`")
        w()
        w(md_table(resultados_targets[target_col]["comparacao"]))

    w(f"## 4. Tabela de decis por contrato (target `{PRIMARY_TARGET}`, score={{faixa com fallback pooled}})")
    w()
    w(md_table(dec_contrato))

    w(f"## 5. Tabela de decis por caixa / volume financeiro (target `{PRIMARY_TARGET}`)")
    w()
    w(md_table(dec_caixa))

    w("## 6. PSI do score pooled entre fotos mensais (vs. snapshot base)")
    w()
    w(md_table(psi_df))

    w("## 7. Curva de lucro e corte otimo por faixa")
    w()
    w("Curva completa (lucro acumulado por numero de decis acionados, canal por faixa):")
    w()
    w(md_table(curva_lucro))
    w("Corte otimo por faixa (maximiza lucro):")
    w()
    w(md_table(resumo_otimo))

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

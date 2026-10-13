"""App Streamlit do framework collscore.

Reutiliza os modulos do pacote `collscore` (simulate, features, target,
model, metrics, finance) sem duplicar logica de negocio -- este arquivo so
contem orquestracao, cache, validacao de upload e visualizacao.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from collscore import FAIXAS
from collscore.features import FEATURE_COLS, build_features
from collscore.finance import CANAL_CUSTOS, CANAL_POR_FAIXA, SALDO_MINIMO_ELEGIVEL, run_finance_analysis
from collscore.metrics import decile_table, model_ks_gini, resumo_regularizacao, roll_rate_saldo
from collscore.model import train_pooled
from collscore.simulate import FAIXA_DIAS_MEDIO, simulate_portfolio
from collscore.target import compute_targets
from run_demo import TARGET_COL, train_test_split_book

FAIXAS_DELINQUENTES = [f for f in FAIXAS if f != "em_dia"]
CANAIS = list(CANAL_CUSTOS.keys())

REQUIRED_UPLOAD_COLS = [
    "id_contrato",
    "snapshot",
    "faixa_atraso",
    "saldo_devedor",
    "saldo_atraso",
    "parcela_valor",
    "parcelas_pagas",
    "parcelas_vencidas",
]

st.set_page_config(page_title="collscore", layout="wide")


# --------------------------------------------------------------------------- upload: validacao e adaptacao


def validar_upload(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de erros de validacao do painel enviado (vazia se ok)."""
    erros = []

    faltantes = [c for c in REQUIRED_UPLOAD_COLS if c not in df.columns]
    if faltantes:
        erros.append("Colunas obrigatorias ausentes: " + ", ".join(faltantes))
        return erros  # sem as colunas minimas nao ha o que validar a seguir

    colunas_numericas = [
        "snapshot",
        "saldo_devedor",
        "saldo_atraso",
        "parcela_valor",
        "parcelas_pagas",
        "parcelas_vencidas",
    ]
    for c in colunas_numericas:
        original_na = df[c].isna()
        coerced = pd.to_numeric(df[c], errors="coerce")
        valores_invalidos = coerced.isna() & ~original_na
        if valores_invalidos.any():
            erros.append(f"Coluna '{c}' contem {int(valores_invalidos.sum())} valor(es) nao numerico(s).")
        elif (coerced.dropna() < 0).any():
            erros.append(f"Coluna '{c}' tem valores negativos.")

    faixas_invalidas = sorted(set(df["faixa_atraso"].dropna().unique()) - set(FAIXAS))
    if faixas_invalidas:
        erros.append(
            f"Valores de 'faixa_atraso' nao reconhecidos: {faixas_invalidas}. Esperado um de {FAIXAS}."
        )

    if "id_contrato" not in df.columns or df["id_contrato"].isna().all():
        erros.append("Coluna 'id_contrato' vazia ou ausente.")
    else:
        snap_num = pd.to_numeric(df["snapshot"], errors="coerce")
        tmp = df.assign(_snap=snap_num).dropna(subset=["id_contrato", "_snap"])

        def _tem_consecutivo(s: pd.Series) -> bool:
            vals = sorted(s.unique())
            return any(b - a == 1 for a, b in zip(vals, vals[1:]))

        if tmp.empty:
            erros.append("Nao ha linhas com id_contrato e snapshot validos.")
        else:
            ok = tmp.groupby("id_contrato")["_snap"].apply(_tem_consecutivo)
            if not ok.any():
                erros.append(
                    "Nenhum contrato possui pelo menos 2 snapshots consecutivos (M e M+1) -- "
                    "impossivel calcular o target pagou_1parcela."
                )

    return erros


def adaptar_upload_para_painel(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o painel minimo enviado pelo usuario no formato esperado por
    `collscore.features.build_features`, preenchendo com defaults neutros as
    colunas que o upload nao traz (produto, valor_financiado, prazo, mob,
    dias_atraso, recencia_pgto, acionamentos_mes, quebra_promessa_mes).

    `dias_atraso` quando ausente e estimado a partir de `faixa_atraso` via o
    mesmo mapeamento usado pelo simulador (`FAIXA_DIAS_MEDIO`), em vez de um
    default arbitrario.
    """
    painel = df.rename(columns={"id_contrato": "contrato_id"}).copy()
    painel["contrato_id"] = painel["contrato_id"].astype(str)
    painel["snapshot"] = pd.to_numeric(painel["snapshot"], errors="coerce").astype(int)

    for c in ["saldo_devedor", "saldo_atraso", "parcela_valor", "parcelas_pagas", "parcelas_vencidas"]:
        painel[c] = pd.to_numeric(painel[c], errors="coerce")

    if "produto" not in painel.columns:
        painel["produto"] = "desconhecido"
    if "valor_financiado" not in painel.columns:
        painel["valor_financiado"] = painel["saldo_devedor"]
    if "prazo" not in painel.columns:
        painel["prazo"] = 999
    if "mob" not in painel.columns:
        painel["mob"] = painel["snapshot"]
    if "dias_atraso" not in painel.columns:
        painel["dias_atraso"] = painel["faixa_atraso"].map(FAIXA_DIAS_MEDIO).fillna(0.0)
    if "recencia_pgto" not in painel.columns:
        painel["recencia_pgto"] = 0.0
    if "acionamentos_mes" not in painel.columns:
        painel["acionamentos_mes"] = 0.0
    if "quebra_promessa_mes" not in painel.columns:
        painel["quebra_promessa_mes"] = 0

    return painel


@st.cache_data(show_spinner=False)
def gerar_template_csv() -> bytes:
    """Gera 50 linhas de exemplo no layout esperado pelo upload, reusando o
    proprio simulador para garantir consistencia de definicoes.
    """
    panel = simulate_portfolio(n_contratos=5, n_snapshots=12, seed=7)
    cols = [
        "contrato_id",
        "snapshot",
        "faixa_atraso",
        "saldo_devedor",
        "saldo_atraso",
        "parcela_valor",
        "parcelas_pagas",
        "parcelas_vencidas",
    ]
    out = panel[cols].rename(columns={"contrato_id": "id_contrato"}).head(50)
    return out.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- pipeline (cacheado)


def _montar_pipeline(panel: pd.DataFrame, burn_in: int, seed: int) -> dict:
    feats = build_features(panel)
    targets = compute_targets(panel)

    book_full = feats.merge(targets[["contrato_id", "snapshot", TARGET_COL]], on=["contrato_id", "snapshot"])
    book_full = book_full[book_full["snapshot"] >= burn_in]
    book = book_full[book_full["faixa_atraso"] != "em_dia"].copy()
    book_elegivel = book.dropna(subset=[TARGET_COL]).copy()

    if book_elegivel.empty or book_elegivel[TARGET_COL].nunique() < 2:
        raise ValueError(
            "Nao ha dados suficientes apos o burn-in para treinar o modelo "
            "(book elegivel vazio ou sem variacao no target pagou_1parcela)."
        )

    book_elegivel["valor_recuperado"] = book_elegivel[TARGET_COL] * book_elegivel["parcela_valor"]

    train, test = train_test_split_book(book_elegivel, test_frac=0.3, seed=seed)
    if train.empty or train[TARGET_COL].nunique() < 2:
        raise ValueError(
            "Particao de treino vazia ou sem variacao no target -- envie uma base com mais contratos."
        )
    if test.empty:
        raise ValueError(
            "Particao de teste vazia apos o split -- envie uma base com mais contratos elegiveis "
            "(inadimplentes com M e M+1 disponiveis)."
        )

    modelo = train_pooled(train, FEATURE_COLS, TARGET_COL)
    test = test.copy()
    test["score"] = modelo.predict_proba(test)

    panel_pos_burnin = panel[panel["snapshot"] >= burn_in]

    resumo_exec = resumo_regularizacao(book_elegivel, target_col=TARGET_COL)
    rr_saldo = roll_rate_saldo(panel_pos_burnin)
    ks_gini = model_ks_gini(test["score"].to_numpy(), test[TARGET_COL].to_numpy())
    dec = decile_table(
        test["score"].to_numpy(),
        test[TARGET_COL].to_numpy(),
        test["saldo_atraso"].to_numpy(),
        test["valor_recuperado"].to_numpy(),
    )

    return dict(
        resumo_exec=resumo_exec,
        rr_saldo=rr_saldo,
        ks_gini=ks_gini,
        decile=dec,
        test=test,
        n_train=len(train),
        n_test=len(test),
    )


@st.cache_data(show_spinner="Simulando carteira e treinando modelo...")
def pipeline_simulado(n_contratos: int, n_snapshots: int, seed: int, burn_in: int) -> dict:
    panel = simulate_portfolio(n_contratos=n_contratos, n_snapshots=n_snapshots, seed=seed)
    return _montar_pipeline(panel, burn_in=burn_in, seed=seed)


@st.cache_data(show_spinner="Validando upload e treinando modelo...")
def pipeline_upload(arquivo_bytes: bytes, nome_arquivo: str, seed: int) -> dict:
    if nome_arquivo.lower().endswith(".parquet"):
        df_raw = pd.read_parquet(io.BytesIO(arquivo_bytes))
    else:
        df_raw = pd.read_csv(io.BytesIO(arquivo_bytes))

    erros = validar_upload(df_raw)
    if erros:
        raise ValueError("ERROS_VALIDACAO::" + "||".join(erros))

    panel = adaptar_upload_para_painel(df_raw)
    return _montar_pipeline(panel, burn_in=0, seed=seed)


def fmt3(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(3)


# --------------------------------------------------------------------------- sidebar: fonte de dados

st.sidebar.header("Fonte de dados")
fonte = st.sidebar.radio("Fonte", ["Base de exemplo (simulada)", "Upload de base própria"])

pipeline = None
erro_pipeline = None

if fonte == "Base de exemplo (simulada)":
    n_contratos = st.sidebar.slider("n_contratos", 5000, 50_000, 15_000, step=1000)
    n_snapshots = st.sidebar.slider("n_snapshots", 6, 12, 8)
    burn_in = n_snapshots // 2
    st.sidebar.caption(
        f"Burn-in proporcional: descarta snapshots 0-{burn_in - 1}; "
        f"treina e mede em {burn_in}-{n_snapshots - 1}."
    )
    try:
        pipeline = pipeline_simulado(n_contratos, n_snapshots, seed=42, burn_in=burn_in)
    except ValueError as e:
        erro_pipeline = str(e)

else:
    st.sidebar.download_button(
        "Baixar template CSV",
        data=gerar_template_csv(),
        file_name="template_collscore.csv",
        mime="text/csv",
    )
    arquivo = st.sidebar.file_uploader("Painel (CSV ou Parquet)", type=["csv", "parquet"])
    if arquivo is not None:
        try:
            pipeline = pipeline_upload(arquivo.getvalue(), arquivo.name, seed=42)
        except ValueError as e:
            msg = str(e)
            if msg.startswith("ERROS_VALIDACAO::"):
                erro_pipeline = msg.split("::", 1)[1].split("||")
            else:
                erro_pipeline = [msg]

# --------------------------------------------------------------------------- sidebar: parametros financeiros

st.sidebar.header("Parametros financeiros")
st.sidebar.caption("Aplicados ao vivo (sem retreinar o modelo).")

canal_custos = {}
for canal, default in CANAL_CUSTOS.items():
    canal_custos[canal] = st.sidebar.number_input(
        f"Custo {canal} (R$)", min_value=0.0, value=float(default), step=0.01, format="%.3f", key=f"custo_{canal}"
    )

st.sidebar.subheader("Canal por faixa")
canal_por_faixa = {"em_dia": None}
for faixa in FAIXAS_DELINQUENTES:
    default_canal = CANAL_POR_FAIXA.get(faixa, "sms")
    canal_por_faixa[faixa] = st.sidebar.selectbox(
        f"Faixa {faixa}", CANAIS, index=CANAIS.index(default_canal), key=f"canal_{faixa}"
    )

st.sidebar.subheader("Saldo minimo elegivel")
saldo_minimo_elegivel = {}
for canal, default in SALDO_MINIMO_ELEGIVEL.items():
    saldo_minimo_elegivel[canal] = st.sidebar.number_input(
        f"Piso {canal} (R$)", min_value=0.0, value=float(default), step=100.0, format="%.3f", key=f"piso_{canal}"
    )

# --------------------------------------------------------------------------- corpo principal

st.title("collscore -- collection scoring e regularizacao financeira")

if erro_pipeline is not None:
    st.error("Nao foi possivel montar o pipeline:")
    for e in erro_pipeline if isinstance(erro_pipeline, list) else [erro_pipeline]:
        st.error(f"- {e}")
    st.stop()

if pipeline is None:
    st.info("Envie um arquivo no painel lateral para continuar.")
    st.stop()

resumo_exec = pipeline["resumo_exec"]
rr_saldo = pipeline["rr_saldo"]
ks_gini = pipeline["ks_gini"]
dec = pipeline["decile"]
test = pipeline["test"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Visao executiva", "Roll rate de saldo", "Score e decis", "Estrategia financeira"]
)

# ---------------------------------------------------------------- tab 1: visao executiva
with tab1:
    total_row = resumo_exec[resumo_exec["faixa_atraso"] == "TOTAL"].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Taxa de regularizacao -- contratos", f"{total_row['taxa_reg_contratos']:.3f}")
    c2.metric("Taxa de regularizacao -- caixa", f"{total_row['taxa_reg_caixa']:.3f}")

    st.caption(
        "Caixa recuperado = 1 parcela paga x evento (nao a quitacao do saldo em atraso). "
        "Por isso `taxa_reg_caixa` e estruturalmente decrescente com a faixa: o saldo em "
        "atraso cresce muito mais rapido do que o valor de uma unica parcela ao longo das "
        "faixas, entao a mesma taxa de regularizacao em # contratos cobre uma fracao cada "
        "vez menor do saldo."
    )

    st.subheader("Tabela executiva por faixa")
    st.dataframe(fmt3(resumo_exec), width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 2: roll rate de saldo
with tab2:
    st.subheader("Roll rate de saldo M -> M+1")
    st.caption(
        "Fracao do saldo devedor (R$) na faixa de origem (linha) que migra para cada faixa "
        "de destino (coluna) no mes seguinte. Cada linha soma 1.000."
    )

    styled = rr_saldo.style.format("{:.3f}").background_gradient(cmap="Blues", axis=1)
    st.dataframe(styled, width="stretch")

    somas = rr_saldo.sum(axis=1)
    st.caption(f"Soma das linhas -- min: {somas.min():.3f} | max: {somas.max():.3f}")

# ---------------------------------------------------------------- tab 3: score e decis
with tab3:
    st.subheader(f"Avaliacao do modelo pooled (target `{TARGET_COL}`)")
    c1, c2, c3 = st.columns(3)
    c1.metric("KS", f"{ks_gini.loc[0, 'ks']:.3f}")
    c2.metric("Gini", f"{ks_gini.loc[0, 'gini']:.3f}")
    c3.metric("n (teste)", f"{int(ks_gini.loc[0, 'n'])}")

    st.subheader("Tabela de decis (conjunto de teste)")
    st.dataframe(fmt3(dec), width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dec))
    width = 0.35
    ax.bar(x - width / 2, dec["taxa_reg_contratos"], width, label="Taxa reg. contratos")
    ax.bar(x + width / 2, dec["taxa_reg_caixa"], width, label="Taxa reg. caixa")
    ax.set_xticks(x)
    ax.set_xticklabels(dec["decil"])
    ax.set_xlabel("Decil (10 = maior score)")
    ax.set_ylabel("Taxa")
    ax.legend()
    ax.set_title("Taxa de regularizacao por decil")
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 4: estrategia financeira
with tab4:
    st.subheader("Curva de lucro e corte otimo por faixa/canal")

    curva_lucro, resumo_otimo = run_finance_analysis(
        test,
        score_col="score",
        target_col=TARGET_COL,
        canal_custos=canal_custos,
        canal_por_faixa=canal_por_faixa,
        saldo_minimo_elegivel=saldo_minimo_elegivel,
    )

    if curva_lucro.empty:
        st.warning("Nenhuma faixa teve populacao elegivel para acionamento com os parametros atuais.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for faixa, sub in curva_lucro.groupby("faixa_atraso"):
            ax.plot(sub["decis_acionados"], sub["lucro"], marker="o", label=faixa)
        ax.set_xlabel("Decis acionados")
        ax.set_ylabel("Lucro (R$)")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.legend(title="Faixa")
        ax.set_title("Lucro acumulado x decis acionados")
        st.pyplot(fig)

        st.subheader("Corte otimo por faixa/canal")
        st.dataframe(fmt3(resumo_otimo), width="stretch", hide_index=True)

        if not resumo_otimo.empty:
            juridico = resumo_otimo[resumo_otimo["canal"] == "juridico"]
            baratos = resumo_otimo[resumo_otimo["canal"] != "juridico"]

            partes = []
            if not juridico.empty:
                cortes = ", ".join(
                    f"{row['faixa_atraso']} ({row['decis_acionados_otimo']}/10)"
                    for _, row in juridico.iterrows()
                )
                partes.append(f"no juridico o score **corta** a fila: {cortes}")
            if not baratos.empty:
                cortes = ", ".join(
                    f"{row['faixa_atraso']} ({row['decis_acionados_otimo']}/10)"
                    for _, row in baratos.iterrows()
                )
                partes.append(f"nos canais baratos o score **so ordena** a fila: {cortes}")

            if partes:
                st.info(" ; ".join(partes).capitalize() + ".")

        st.subheader("Curva completa")
        st.dataframe(fmt3(curva_lucro), width="stretch", hide_index=True)

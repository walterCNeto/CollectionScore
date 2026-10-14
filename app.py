"""App Streamlit do framework collscore.

Reutiliza os modulos do pacote `collscore` (simulate, features, target,
model, metrics, finance, report) sem duplicar logica de negocio -- este
arquivo so contem orquestracao, cache, validacao de upload e visualizacao.

Nada recalcula automaticamente: os controles ficam dentro de um `st.form` e
so sao aplicados no clique de "Aplicar e rodar modelo". O resultado (e os
parametros usados para gera-lo) ficam em `st.session_state`; as abas
renderizam exclusivamente a partir de la. No primeiro carregamento, o
pipeline roda uma vez com os defaults para o app nao abrir vazio.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from collscore import FAIXAS
from collscore.features import FEATURE_COLS, build_features
from collscore.finance import (
    CANAL_CUSTOS,
    CANAL_POR_FAIXA,
    EFETIVIDADE,
    SALDO_MINIMO_ELEGIVEL,
    fronteira_orcamento_lucro,
    resolve_canal_por_faixa,
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

FONTE_SIMULADA = "Base de exemplo (simulada)"
FONTE_UPLOAD = "Upload de base própria"

DEFAULTS = dict(
    fonte=FONTE_SIMULADA,
    n_contratos=15_000,
    n_snapshots=8,
    seed=42,
    arquivo_bytes=None,
    arquivo_nome=None,
    canal_custos=dict(CANAL_CUSTOS),
    estrategia_modo="Automatica (maximizar lucro)",
    canal_por_faixa_manual={"em_dia": None, **{f: CANAL_POR_FAIXA.get(f, "sms") for f in FAIXAS_DELINQUENTES}},
    saldo_minimo_elegivel=dict(SALDO_MINIMO_ELEGIVEL),
    efetividade={canal: dict(faixas) for canal, faixas in EFETIVIDADE.items()},
    target_col=TARGET_COL,
)
DEFAULTS["burn_in"] = DEFAULTS["n_snapshots"] // 2

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

    book_full = feats.merge(
        targets[["contrato_id", "snapshot", "pagou_1parcela", "cura_total", "melhora_faixa"]],
        on=["contrato_id", "snapshot"],
    )
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
    ks_gini_faixa = ks_gini_por_faixa(test, score_col="score", target_col=TARGET_COL)
    dec = decile_table(
        test["score"].to_numpy(),
        test[TARGET_COL].to_numpy(),
        test["saldo_atraso"].to_numpy(),
        test["valor_recuperado"].to_numpy(),
    )
    targets_alternativos = taxas_evento_por_faixa(book, ["pagou_1parcela", "cura_total", "melhora_faixa"])
    feature_describe = book_elegivel[FEATURE_COLS].describe(percentiles=[0.25, 0.5, 0.75]).T.round(3)

    volumetria_snapshot = (
        panel_pos_burnin.assign(delinquente=(panel_pos_burnin["faixa_atraso"] != "em_dia").astype(float))
        .groupby("snapshot")
        .agg(n_contratos=("contrato_id", "size"), pct_delinquente=("delinquente", "mean"))
        .round(3)
        .reset_index()
    )

    return dict(
        resumo_exec=resumo_exec,
        rr_saldo=rr_saldo,
        ks_gini=ks_gini,
        ks_gini_faixa=ks_gini_faixa,
        decile=dec,
        test=test,
        targets_alternativos=targets_alternativos,
        feature_describe=feature_describe,
        volumetria_snapshot=volumetria_snapshot,
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


def normalizar_estrategia_modo(estrategia_modo: str) -> str:
    if estrategia_modo.startswith("Manual"):
        return "manual"
    return "lucro" if "lucro" in estrategia_modo else "roi"


def executar_pipeline(params: dict) -> None:
    """Roda o pipeline (simulado ou upload) com os parametros dados e guarda
    resultados + parametros + erro em `st.session_state`. Chamado no
    primeiro carregamento (com `DEFAULTS`) e a cada submit do formulario.
    """
    try:
        if params["fonte"] == FONTE_SIMULADA:
            resultados = pipeline_simulado(
                params["n_contratos"], params["n_snapshots"], params["seed"], params["burn_in"]
            )
        else:
            if params.get("arquivo_bytes") is None:
                raise ValueError("Selecione um arquivo (CSV ou Parquet) antes de aplicar.")
            resultados = pipeline_upload(params["arquivo_bytes"], params["arquivo_nome"], params["seed"])
    except ValueError as e:
        msg = str(e)
        if msg.startswith("ERROS_VALIDACAO::"):
            st.session_state["erro"] = msg.split("::", 1)[1].split("||")
        else:
            st.session_state["erro"] = [msg]
        # mantem os ultimos resultados bons (se houver) visiveis junto do erro,
        # em vez de esvaziar o dashboard por causa de um submit invalido
        return

    st.session_state["erro"] = None
    st.session_state["resultados"] = resultados
    st.session_state["parametros"] = params


# --------------------------------------------------------------------------- primeiro carregamento

if "resultados" not in st.session_state:
    executar_pipeline(DEFAULTS)

# --------------------------------------------------------------------------- sidebar

st.sidebar.title("collscore")

st.sidebar.download_button(
    "Baixar template CSV",
    data=gerar_template_csv(),
    file_name="template_collscore.csv",
    mime="text/csv",
    width="stretch",
)

with st.sidebar.form("parametros_form"):
    st.header("Fonte de dados")
    fonte = st.radio("Fonte", [FONTE_SIMULADA, FONTE_UPLOAD])

    st.subheader("Base de exemplo (simulada)")
    n_contratos = st.slider("n_contratos", 5000, 50_000, DEFAULTS["n_contratos"], step=1000)
    n_snapshots = st.slider("n_snapshots", 6, 12, DEFAULTS["n_snapshots"])
    st.caption("Burn-in = metade dos snapshots, aplicado ao clicar em 'Aplicar e rodar modelo'.")

    st.subheader("Upload de base propria (CSV ou Parquet)")
    arquivo = st.file_uploader("Painel", type=["csv", "parquet"])
    st.caption("Colunas obrigatorias: " + ", ".join(REQUIRED_UPLOAD_COLS))

    st.header("Parametros financeiros")
    canal_custos_form = {}
    for canal, default in CANAL_CUSTOS.items():
        canal_custos_form[canal] = st.number_input(
            f"Custo {canal} (R$)", min_value=0.0, value=float(default), step=0.01, format="%.3f", key=f"custo_{canal}"
        )

    st.subheader("Estrategia de canal")
    estrategia_modo = st.radio(
        "Estrategia de canal",
        [
            "Automatica (maximizar lucro)",
            "Automatica (maximizar ROI)",
            "Manual (mapeamento por faixa)",
        ],
        label_visibility="collapsed",
    )

    st.subheader("Canal por faixa (usado apenas no modo Manual)")
    canal_por_faixa_manual = {"em_dia": None}
    for faixa in FAIXAS_DELINQUENTES:
        default_canal = CANAL_POR_FAIXA.get(faixa, "sms")
        canal_por_faixa_manual[faixa] = st.selectbox(
            f"Faixa {faixa}", CANAIS, index=CANAIS.index(default_canal), key=f"canal_{faixa}"
        )

    st.subheader("Saldo minimo elegivel")
    saldo_minimo_elegivel_form = {}
    for canal, default in SALDO_MINIMO_ELEGIVEL.items():
        saldo_minimo_elegivel_form[canal] = st.number_input(
            f"Piso {canal} (R$)", min_value=0.0, value=float(default), step=100.0, format="%.3f", key=f"piso_{canal}"
        )

    st.subheader("Efetividade (canal x faixa)")
    st.caption(
        "Proxy de resposta diferencial por celula canal x faixa -- valores ARBITRADOS, "
        "nao estimados de dados; passo intermediario ate uma modelagem causal de uplift."
    )
    matriz_efetividade_base = pd.DataFrame(
        {faixa: [EFETIVIDADE[canal].get(faixa, 0.0) for canal in CANAIS] for faixa in FAIXAS_DELINQUENTES},
        index=CANAIS,
    )
    matriz_efetividade_editada = st.data_editor(
        matriz_efetividade_base,
        column_config={
            faixa: st.column_config.NumberColumn(faixa, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
            for faixa in FAIXAS_DELINQUENTES
        },
        num_rows="fixed",
        key="matriz_efetividade",
        width="stretch",
    )

    submitted = st.form_submit_button("Aplicar e rodar modelo", width="stretch")

if submitted:
    efetividade_form = {
        canal: {
            faixa: float(np.clip(matriz_efetividade_editada.loc[canal, faixa], 0.0, 1.0))
            for faixa in FAIXAS_DELINQUENTES
        }
        for canal in CANAIS
    }
    params = dict(
        fonte=fonte,
        n_contratos=n_contratos,
        n_snapshots=n_snapshots,
        burn_in=n_snapshots // 2,
        seed=42,
        arquivo_bytes=arquivo.getvalue() if arquivo is not None else None,
        arquivo_nome=arquivo.name if arquivo is not None else None,
        canal_custos=canal_custos_form,
        estrategia_modo=estrategia_modo,
        canal_por_faixa_manual=canal_por_faixa_manual,
        saldo_minimo_elegivel=saldo_minimo_elegivel_form,
        efetividade=efetividade_form,
        target_col=TARGET_COL,
    )
    executar_pipeline(params)

# --------------------------------------------------------------------------- corpo principal

st.title("collscore -- collection scoring e regularizacao financeira")

if st.session_state.get("erro"):
    st.error("Nao foi possivel montar o pipeline com os ultimos parametros aplicados:")
    for e in st.session_state["erro"]:
        st.error(f"- {e}")

resultados = st.session_state.get("resultados")
parametros = st.session_state.get("parametros", DEFAULTS)

if resultados is None:
    st.info("Ajuste os parametros na barra lateral e clique em 'Aplicar e rodar modelo'.")
    st.stop()

st.caption(
    f"Resultados vigentes: fonte = {parametros.get('fonte')} &middot; "
    f"estrategia de canal = {parametros.get('estrategia_modo')}"
)

resumo_exec = resultados["resumo_exec"]
rr_saldo = resultados["rr_saldo"]
ks_gini = resultados["ks_gini"]
ks_gini_faixa = resultados["ks_gini_faixa"]
dec = resultados["decile"]
test = resultados["test"]

# --------------------------------------------------------------------------- relatorio HTML

if st.sidebar.button("Gerar relatorio HTML", width="stretch"):
    st.session_state["relatorio_html"] = build_html_report(resultados, parametros)

if "relatorio_html" in st.session_state:
    st.sidebar.download_button(
        "Baixar relatorio HTML",
        data=st.session_state["relatorio_html"],
        file_name="relatorio_modelo.html",
        mime="text/html",
        width="stretch",
    )

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
    st.caption(
        "O KS/Gini GLOBAL mistura faixas com taxas de evento bem diferentes -- isso "
        "infla o numero em relacao ao poder discriminante do score DENTRO de cada "
        "faixa. Veja a quebra por faixa abaixo."
    )

    st.subheader("KS / Gini por faixa (mesmo score, sem retreinar)")
    st.dataframe(fmt3(ks_gini_faixa), width="stretch", hide_index=True)

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
    canal_custos = parametros["canal_custos"]
    saldo_minimo_elegivel = parametros["saldo_minimo_elegivel"]
    efetividade = parametros["efetividade"]
    estrategia_modo_atual = parametros["estrategia_modo"]

    st.caption(
        "Negativacao modela a inclusao em bureau de credito (Serasa/SPC). Por lei "
        "(CDC, art. 43) e exigida notificacao previa ao devedor antes da inclusao -- "
        "o custo default do canal ja embute essa notificacao."
    )
    st.caption(
        "A matriz de efetividade (canal x faixa) e uma calibracao ARBITRADA (nao "
        "estimada de dados) e a comparacao abaixo e entre canais PUROS -- reguas reais "
        "de cobranca sao sequenciais (sms, depois negativacao, depois "
        "call_center/juridico). As conclusoes de qual canal vence podem se inverter "
        "com outros valores: ajuste a matriz na sidebar e clique em 'Aplicar e rodar "
        "modelo' para uma analise de sensibilidade."
    )

    # ------------------------------------------------- comparacao canal x faixa
    st.subheader("Comparacao de canais por faixa")

    comparacao = run_channel_comparison(
        test,
        score_col="score",
        target_col=TARGET_COL,
        canal_custos=canal_custos,
        saldo_minimo_elegivel=saldo_minimo_elegivel,
        efetividade=efetividade,
    )

    if comparacao.empty:
        st.warning("Nenhuma faixa teve populacao elegivel para comparar canais com os parametros atuais.")
    else:
        idx_lucro = comparacao.groupby("faixa_atraso")["lucro_max"].idxmax()
        vencedores_lucro = set(
            zip(comparacao.loc[idx_lucro, "faixa_atraso"], comparacao.loc[idx_lucro, "canal"])
        )

        comp_roi = comparacao.dropna(subset=["roi_no_corte"])
        if not comp_roi.empty:
            idx_roi = comp_roi.groupby("faixa_atraso")["roi_no_corte"].idxmax()
            vencedores_roi = set(
                zip(comp_roi.loc[idx_roi, "faixa_atraso"], comp_roi.loc[idx_roi, "canal"])
            )
        else:
            vencedores_roi = set()

        comp_display = fmt3(comparacao)
        chaves = list(zip(comp_display["faixa_atraso"], comp_display["canal"]))
        comp_display = comp_display.assign(
            vencedor_lucro=[chave in vencedores_lucro for chave in chaves],
            vencedor_roi=[chave in vencedores_roi for chave in chaves],
        )
        st.dataframe(comp_display, width="stretch", hide_index=True)

    # --------------------------------------------- fronteira orcamento x lucro
    st.subheader("Fronteira orcamento x lucro: lucro vs ROI")
    st.caption(
        "As estrategias 'lucro' e 'roi' sao dois pontos extremos de uma fronteira "
        "orcamento x lucro: ROI gasta pouco e maximiza retorno por R$ investido; "
        "lucro gasta mais para maximizar o retorno absoluto."
    )

    fronteira = fronteira_orcamento_lucro(
        test,
        score_col="score",
        target_col=TARGET_COL,
        canal_custos=canal_custos,
        saldo_minimo_elegivel=saldo_minimo_elegivel,
        efetividade=efetividade,
    )

    if fronteira.empty or fronteira["custo_total"].isna().all():
        st.warning("Sem estrategia disponivel para tracar a fronteira com os parametros atuais.")
    else:
        st.dataframe(fmt3(fronteira), width="stretch", hide_index=True)

        fig_front, ax_front = plt.subplots(figsize=(6, 4))
        ax_front.scatter(fronteira["custo_total"], fronteira["lucro_total"], s=80, zorder=3)
        for _, row in fronteira.iterrows():
            ax_front.annotate(
                row["criterio"],
                (row["custo_total"], row["lucro_total"]),
                textcoords="offset points",
                xytext=(8, 4),
            )
        ax_front.plot(fronteira["custo_total"], fronteira["lucro_total"], linestyle="--", color="grey", zorder=1)
        ax_front.set_xlabel("Custo total (R$)")
        ax_front.set_ylabel("Lucro total (R$)")
        ax_front.set_title("Fronteira orcamento x lucro")
        st.pyplot(fig_front)

    # ------------------------------------------------------------- estrategia
    st.subheader("Estrategia escolhida")

    canal_por_faixa_efetivo = resolve_canal_por_faixa(
        test,
        normalizar_estrategia_modo(estrategia_modo_atual),
        parametros["canal_por_faixa_manual"],
        score_col="score",
        target_col=TARGET_COL,
        canal_custos=canal_custos,
        saldo_minimo_elegivel=saldo_minimo_elegivel,
        efetividade=efetividade,
    )

    curva_lucro, resumo_otimo = run_finance_analysis(
        test,
        score_col="score",
        target_col=TARGET_COL,
        canal_custos=canal_custos,
        canal_por_faixa=canal_por_faixa_efetivo,
        saldo_minimo_elegivel=saldo_minimo_elegivel,
        efetividade=efetividade,
    )

    if resumo_otimo.empty:
        st.warning("Nenhuma faixa teve populacao elegivel para acionamento com os parametros atuais.")
    else:
        lucro_total = resumo_otimo["lucro_max"].sum()
        custo_total_estrategia = resumo_otimo["custo_total"].sum()
        caixa_incremental_total = lucro_total + custo_total_estrategia
        roi_agregado = (
            caixa_incremental_total / custo_total_estrategia if custo_total_estrategia > 0 else float("nan")
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Lucro total", f"R$ {lucro_total:,.3f}")
        c2.metric("Custo total", f"R$ {custo_total_estrategia:,.3f}")
        c3.metric("ROI agregado", f"{roi_agregado:.3f}")

        st.dataframe(fmt3(resumo_otimo), width="stretch", hide_index=True)

        juridico = resumo_otimo[resumo_otimo["canal"] == "juridico"]
        baratos = resumo_otimo[resumo_otimo["canal"] != "juridico"]

        partes = []
        if not juridico.empty:
            cortes = ", ".join(
                f"{row['faixa_atraso']} ({row['decis_acionados_otimo']}/10)"
                for _, row in juridico.iterrows()
            )
            partes.append(f"o score define corte apenas onde o custo de acionamento e material (juridico): {cortes}")
        if not baratos.empty:
            cortes = ", ".join(
                f"{row['faixa_atraso']} ({row['decis_acionados_otimo']}/10)"
                for _, row in baratos.iterrows()
            )
            partes.append(f"nos canais baratos ele apenas ordena a fila: {cortes}")

        if partes:
            st.info(" ; ".join(partes).capitalize() + ".")

    # ----------------------------------------------------- curva de lucro (grafico)
    st.subheader("Curva de lucro (estrategia aplicada)")
    if curva_lucro.empty:
        st.warning("Sem curva disponivel para os parametros atuais.")
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

        st.subheader("Curva completa")
        st.dataframe(fmt3(curva_lucro), width="stretch", hide_index=True)

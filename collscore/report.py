"""Relatorio HTML de desenvolvimento e validacao do modelo collscore.

`build_html_report(resultados, parametros)` gera um documento HTML
autocontido (CSS inline, graficos matplotlib embutidos como PNG base64, sem
dependencias externas) cobrindo objetivo, base de dados, target, variaveis,
metodologia, desempenho, estabilidade, uso financeiro do score (camada
ilustrativa) e limitacoes -- o material de validacao que acompanha o
modelo, nao um relatorio executivo de negocio (esse e o papel de
`run_demo.py` / `relatorio.md`).

`resultados` e o dict retornado por `app._montar_pipeline` (ou equivalente):
resumo_exec, rr_saldo, ks_gini, ks_gini_faixa, decile, test,
targets_alternativos, feature_describe, volumetria_snapshot, n_train, n_test.

`parametros` descreve a configuracao usada para gerar `resultados`: fonte,
n_contratos, n_snapshots, burn_in, seed, target_col, canal_custos,
estrategia_modo, canal_por_faixa_manual, saldo_minimo_elegivel,
efetividade.
"""
from __future__ import annotations

import base64
import io
import subprocess
from datetime import datetime
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collscore.features import FEATURE_COLS
from collscore.finance import fronteira_orcamento_lucro, resolve_canal_por_faixa, run_finance_analysis

FEATURE_DEFINICOES = {
    "pct_saldo_atraso": "Fracao do saldo devedor que esta em atraso (saldo em atraso / saldo devedor).",
    "pct_parcelas_pagas": "Fracao das parcelas ja vencidas que foram pagas (parcelas pagas / parcelas vencidas).",
    "dias_atraso": "Dias de atraso no snapshot (ponto-no-tempo).",
    "recencia_pgto": "Meses desde o ultimo pagamento registrado.",
    "atraso_max_12m": "Maior numero de dias de atraso observado nos ultimos 12 snapshots.",
    "mob_rel": "Month-on-book relativo ao prazo do contrato (mob / prazo).",
    "quebra_promessa": "Indicador de promessa de pagamento quebrada no mes.",
    "n_acionamentos_3m": "Numero de acionamentos de cobranca nos ultimos 3 meses.",
    "ticket_rel": "Valor financiado relativo a media do produto do contrato.",
}

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #1f2430;
       background: #fafafa; margin: 0; padding: 0; }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }
header.doc-header { border-bottom: 3px solid #2c3e50; padding-bottom: 16px; margin-bottom: 32px; }
header.doc-header h1 { margin: 0 0 4px; font-size: 1.6rem; color: #1f2430; }
header.doc-header p { margin: 2px 0; color: #5a6472; font-size: 0.85rem; }
h2.section-title { font-size: 1.15rem; color: #1f2430; border-left: 4px solid #2c3e50;
                    padding-left: 10px; margin-top: 40px; margin-bottom: 14px; }
h3 { font-size: 0.98rem; color: #33404f; margin-top: 22px; margin-bottom: 8px; }
p, li { font-size: 0.88rem; line-height: 1.5; }
.callout { background: #eef2f6; border-left: 3px solid #5a7897; padding: 10px 14px;
           font-size: 0.82rem; color: #33404f; margin: 10px 0 16px; border-radius: 0 4px 4px 0; }
.callout.warn { background: #fbf2e9; border-left-color: #b4701f; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 18px; font-size: 0.8rem; }
table caption { caption-side: top; text-align: left; font-weight: 600; margin-bottom: 4px; }
th, td { border: 1px solid #d8dde3; padding: 5px 8px; text-align: right; }
th { background: #eef2f6; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tr:nth-child(even) td { background: #fbfbfc; }
img.chart { max-width: 100%; display: block; margin: 12px 0; border: 1px solid #e0e4e8; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
footer { margin-top: 48px; padding-top: 12px; border-top: 1px solid #d8dde3;
         font-size: 0.75rem; color: #8a93a0; }
code { background: #eef2f6; padding: 1px 4px; border-radius: 3px; font-size: 0.85em; }
"""


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "n/d"
    except Exception:
        return "n/d"


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(fig, alt: str) -> str:
    b64 = _fig_to_base64(fig)
    return f'<img class="chart" alt="{escape(alt)}" src="data:image/png;base64,{b64}">'


def _df_html(df: pd.DataFrame, caption: str | None = None) -> str:
    if df is None or df.empty:
        cap = f"<caption>{escape(caption)}</caption>" if caption else ""
        return f"<table>{cap}<tr><td><em>(sem dados)</em></td></tr></table>"

    html = df.round(3).to_html(index=False, float_format=lambda x: f"{x:.3f}", na_rep="n/d", border=0)
    html = html.replace('<table border="0" class="dataframe">', "<table>", 1)
    if caption:
        html = html.replace("<thead>", f"<caption>{escape(caption)}</caption><thead>", 1)
    return html


def _grafico_decis(dec: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(dec))
    width = 0.35
    ax.bar(x - width / 2, dec["taxa_reg_contratos"], width, label="Taxa reg. contratos")
    ax.bar(x + width / 2, dec["taxa_reg_caixa"], width, label="Taxa reg. caixa")
    ax.set_xticks(x)
    ax.set_xticklabels(dec["decil"])
    ax.set_xlabel("Decil (10 = maior score)")
    ax.set_ylabel("Taxa")
    ax.legend(fontsize=8)
    ax.set_title("Taxa de regularizacao por decil", fontsize=10)
    return fig


def _grafico_roll_rate(rr_saldo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    dados = rr_saldo.to_numpy()
    im = ax.imshow(dados, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rr_saldo.columns)))
    ax.set_xticklabels(rr_saldo.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rr_saldo.index)))
    ax.set_yticklabels(rr_saldo.index, fontsize=8)
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            valor = dados[i, j]
            cor = "white" if valor > 0.55 else "#1f2430"
            ax.text(j, i, f"{valor:.3f}", ha="center", va="center", fontsize=7, color=cor)
    ax.set_xlabel("Faixa destino (M+1)")
    ax.set_ylabel("Faixa origem (M)")
    ax.set_title("Roll rate de saldo M -> M+1", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _grafico_fronteira(fronteira: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(fronteira["custo_total"], fronteira["lucro_total"], s=70, zorder=3)
    for _, row in fronteira.iterrows():
        ax.annotate(
            row["criterio"], (row["custo_total"], row["lucro_total"]), textcoords="offset points", xytext=(8, 4)
        )
    ax.plot(fronteira["custo_total"], fronteira["lucro_total"], linestyle="--", color="grey", zorder=1)
    ax.set_xlabel("Custo total (R$)")
    ax.set_ylabel("Lucro total (R$)")
    ax.set_title("Fronteira orcamento x lucro", fontsize=10)
    return fig


def _efetividade_para_df(efetividade: dict, faixas: list[str]) -> pd.DataFrame:
    canais = list(efetividade.keys())
    return pd.DataFrame(
        {faixa: [efetividade[canal].get(faixa, 0.0) for canal in canais] for faixa in faixas},
        index=canais,
    ).reset_index(names="canal")


def build_html_report(resultados: dict, parametros: dict) -> str:
    """Gera o relatorio HTML de desenvolvimento e validacao (string HTML
    completa, autocontida).
    """
    target_col = parametros.get("target_col", "pagou_1parcela")
    fonte = parametros.get("fonte", "Base de exemplo (simulada)")
    is_simulado = fonte == "Base de exemplo (simulada)"

    resumo_exec = resultados.get("resumo_exec", pd.DataFrame())
    rr_saldo = resultados.get("rr_saldo", pd.DataFrame())
    ks_gini = resultados.get("ks_gini", pd.DataFrame())
    ks_gini_faixa = resultados.get("ks_gini_faixa", pd.DataFrame())
    dec = resultados.get("decile", pd.DataFrame())
    test = resultados.get("test", pd.DataFrame())
    targets_alternativos = resultados.get("targets_alternativos", pd.DataFrame())
    feature_describe = resultados.get("feature_describe", pd.DataFrame())
    volumetria_snapshot = resultados.get("volumetria_snapshot", pd.DataFrame())
    n_train = resultados.get("n_train", 0)
    n_test = resultados.get("n_test", 0)

    faixas_presentes = sorted(resumo_exec.loc[resumo_exec["faixa_atraso"] != "TOTAL", "faixa_atraso"].unique()) if not resumo_exec.empty else []

    # -------------------------------------------------- secao 8: financeiro (ilustrativo)
    canal_custos = parametros.get("canal_custos", {})
    efetividade = parametros.get("efetividade", {})
    saldo_minimo_elegivel = parametros.get("saldo_minimo_elegivel", {})
    estrategia_modo = parametros.get("estrategia_modo", "Manual (mapeamento por faixa)")
    modo_norm = (
        "manual"
        if estrategia_modo.startswith("Manual")
        else ("lucro" if "lucro" in estrategia_modo else "roi")
    )

    financeiro_disponivel = not test.empty and "score" in test.columns and canal_custos
    curva_lucro = resumo_otimo = fronteira = pd.DataFrame()
    if financeiro_disponivel:
        try:
            canal_por_faixa_efetivo = resolve_canal_por_faixa(
                test,
                modo_norm,
                parametros.get("canal_por_faixa_manual"),
                score_col="score",
                target_col=target_col,
                canal_custos=canal_custos,
                saldo_minimo_elegivel=saldo_minimo_elegivel,
                efetividade=efetividade,
            )
            curva_lucro, resumo_otimo = run_finance_analysis(
                test,
                score_col="score",
                target_col=target_col,
                canal_custos=canal_custos,
                canal_por_faixa=canal_por_faixa_efetivo,
                saldo_minimo_elegivel=saldo_minimo_elegivel,
                efetividade=efetividade,
            )
            fronteira = fronteira_orcamento_lucro(
                test,
                score_col="score",
                target_col=target_col,
                canal_custos=canal_custos,
                saldo_minimo_elegivel=saldo_minimo_elegivel,
                efetividade=efetividade,
            )
        except Exception:
            financeiro_disponivel = False

    # -------------------------------------------------------------- graficos
    imagens = []
    if not dec.empty:
        imagens.append(_img_tag(_grafico_decis(dec), "Taxa de regularizacao por decil"))
    else:
        imagens.append("<p><em>(sem dados suficientes para a tabela de decis)</em></p>")

    if not rr_saldo.empty:
        imagens.append(_img_tag(_grafico_roll_rate(rr_saldo), "Roll rate de saldo M -> M+1"))
    else:
        imagens.append("<p><em>(sem dados suficientes para o roll rate)</em></p>")

    if not fronteira.empty and fronteira["custo_total"].notna().any():
        imagens.append(_img_tag(_grafico_fronteira(fronteira), "Fronteira orcamento x lucro"))
    else:
        imagens.append("<p><em>(sem estrategia financeira disponivel para a fronteira)</em></p>")

    gerado_em = datetime.now().strftime("%Y-%m-%d %H:%M")
    git_sha = _git_short_sha()

    partes = []
    partes.append(f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatorio collscore</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="doc-header">
  <h1>Relatorio de desenvolvimento e validacao -- collscore</h1>
  <p>Gerado em {escape(gerado_em)} &middot; versao {escape(git_sha)}</p>
  <p>Target: <code>{escape(target_col)}</code> &middot; Fonte de dados: {escape(fonte)}</p>
</header>
""")

    # ---------------------------------------------------------- 1. Objetivo e escopo
    partes.append(f"""
<h2 class="section-title">1. Objetivo e escopo</h2>
<p>Este relatorio documenta o desenvolvimento, a validacao e o uso pretendido de um
escore de propensao a regularizacao financeira (<code>{escape(target_col)}</code>) aplicado
ao book de contratos em atraso de uma carteira de credito (consignado, credito pessoal
e cartao). O escore prioriza a fila de acionamento de cobranca por faixa de atraso; a
camada financeira (secao 8) e ILUSTRATIVA e serve para situar a discussao de estrategia
de canal, nao para fixar precos ou custos operacionais reais.</p>
""")

    # ---------------------------------------------------------- 2. Base de dados
    partes.append(f"""
<h2 class="section-title">2. Base de dados</h2>
<p><strong>Fonte:</strong> {escape(fonte)}. <strong>Periodo:</strong> {parametros.get('n_snapshots', 'n/d')}
fotos mensais simuladas (ou detectadas no upload). <strong>Burn-in:</strong> {parametros.get('burn_in', 'n/d')}
snapshot(s) iniciais descartados (carteira ainda estabilizando a distribuicao de faixas).</p>
<h3>Volumetria por snapshot (pos burn-in)</h3>
{_df_html(volumetria_snapshot)}
<h3>Volumetria por faixa (book elegivel, contratos delinquentes em M com M+1 disponivel)</h3>
{_df_html(resumo_exec[["faixa_atraso", "n_elegivel"]] if not resumo_exec.empty else pd.DataFrame())}
""")

    # ---------------------------------------------------------- 3. Target
    taxa_evento_por_faixa = (
        resumo_exec[["faixa_atraso", "taxa_reg_contratos"]].rename(columns={"taxa_reg_contratos": target_col})
        if not resumo_exec.empty
        else pd.DataFrame()
    )
    partes.append(f"""
<h2 class="section-title">3. Target e janela de performance</h2>
<p><code>{escape(target_col)}</code> = contrato pagou ao menos 1 parcela entre o snapshot M e
M+1, restrito a contratos delinquentes (faixa de atraso != em_dia) em M. Janela M -> M+1
(1 mes). Taxa de evento por faixa (linha TOTAL = book inteiro):</p>
{_df_html(taxa_evento_por_faixa)}
<h3>Targets alternativos (justificativa da escolha)</h3>
<p><code>cura_total</code> (faixa volta a em_dia) e <code>melhora_faixa</code> (faixa melhora,
nao necessariamente para em_dia) sao mais raros e mais dependentes de faixas profundas
terem eventos suficientes para treinar um modelo estavel; <code>pagou_1parcela</code> tem
taxa de evento mais alta e mais homogenea entre faixas, favorecendo um unico modelo
pooled.</p>
{_df_html(targets_alternativos)}
""")

    # ---------------------------------------------------------- 4. Book de variaveis
    linhas_def = "".join(
        f"<li><code>{escape(c)}</code>: {escape(FEATURE_DEFINICOES.get(c, ''))}</li>" for c in FEATURE_COLS
    )
    partes.append(f"""
<h2 class="section-title">4. Book de variaveis</h2>
<ul>{linhas_def}</ul>
<h3>Estatisticas descritivas (book elegivel, media / dp / p25 / p50 / p75)</h3>
{_df_html(feature_describe.reset_index(names="feature") if not feature_describe.empty else pd.DataFrame())}
""")

    # ---------------------------------------------------------- 5. Metodologia
    partes.append(f"""
<h2 class="section-title">5. Metodologia</h2>
<p>Regressao logistica POOLED (um unico modelo para todas as faixas de atraso), com
features continuas padronizadas (<code>StandardScaler</code>) e a faixa de atraso
entrando como variaveis dummy (one-hot, sem a primeira categoria). Sem binning WoE.
Split treino/teste 70/30 por contrato (nao por linha), seed fixa.</p>
<p>Tamanho do treino: {n_train} contratos-snapshot &middot; tamanho do teste: {n_test}
contratos-snapshot.</p>
""")

    # ---------------------------------------------------------- 6. Desempenho
    partes.append(f"""
<h2 class="section-title">6. Desempenho</h2>
<div class="callout warn">O KS/Gini GLOBAL mistura faixas de atraso com taxas de evento
bem diferentes entre si -- isso INFLA o numero em relacao ao poder discriminante real do
score DENTRO de cada faixa (ver quebra por faixa abaixo).</div>
<h3>Global</h3>
{_df_html(ks_gini)}
<h3>Por faixa (mesmo score, sem retreinar nada por faixa)</h3>
{_df_html(ks_gini_faixa)}
<h3>Tabela de decis (contratos e caixa)</h3>
{_df_html(dec)}
{imagens[0]}
""")

    # ---------------------------------------------------------- 7. Estabilidade
    partes.append(f"""
<h2 class="section-title">7. Estabilidade e dinamica</h2>
<p>Fracao do saldo devedor (R$) na faixa de origem (linha) que migra para cada faixa de
destino (coluna) no mes seguinte M -> M+1. Cada linha soma 1.000.</p>
{imagens[1]}
{_df_html(rr_saldo.reset_index(names="faixa_atraso") if not rr_saldo.empty else pd.DataFrame())}
""")

    # ---------------------------------------------------------- 8. Uso do score (ilustrativo)
    efetividade_df = _efetividade_para_df(efetividade, faixas_presentes) if efetividade and faixas_presentes else pd.DataFrame()
    custos_df = pd.DataFrame(
        [{"canal": c, "custo_unitario": v, "saldo_minimo_elegivel": saldo_minimo_elegivel.get(c, 0.0)} for c, v in canal_custos.items()]
    )
    partes.append(f"""
<h2 class="section-title">8. Uso do score -- camada financeira (ILUSTRATIVA)</h2>
<div class="callout">Esta secao usa custos de canal e uma matriz de efetividade
canal x faixa ARBITRADOS (nao estimados de dados) para ilustrar como o score poderia
orientar uma estrategia de acionamento. A comparacao e entre canais PUROS; reguas reais
de cobranca sao sequenciais (ex.: sms, depois negativacao, depois call_center/juridico).
Nao deve ser lida como uma politica de custos ou precificacao real.</div>
<h3>Parametros vigentes</h3>
<p><strong>Custo e piso de saldo por canal:</strong></p>
{_df_html(custos_df)}
<p><strong>Matriz de efetividade (canal x faixa):</strong></p>
{_df_html(efetividade_df)}
<p><strong>Estrategia de canal selecionada:</strong> {escape(estrategia_modo)}</p>
<h3>Corte otimo por faixa/canal (estrategia aplicada)</h3>
{_df_html(resumo_otimo)}
<h3>Fronteira orcamento x lucro (lucro vs ROI)</h3>
<p>As estrategias "lucro" e "roi" sao dois pontos extremos de uma fronteira orcamento x
lucro: ROI gasta pouco e maximiza retorno por R$ investido; lucro gasta mais para
maximizar o retorno absoluto.</p>
{_df_html(fronteira)}
{imagens[2]}
""")

    # ---------------------------------------------------------- 9. Limitacoes
    nota_fonte = (
        "A base usada neste relatorio e SIMULADA (collscore.simulate); os numeros de "
        "desempenho e financeiros refletem a dinamica do simulador, nao uma carteira real."
        if is_simulado
        else "A base usada neste relatorio foi enviada pelo usuario (upload); features "
        "ausentes do book minimo foram preenchidas com defaults neutros (ver app.py)."
    )
    nota_self_cure = (
        "O simulador reserva um flag `flag_self_cure_evento` (cura espontanea, "
        "independente de acionamento, em faixas <=30 dias) para uso futuro em "
        "modelagem de uplift -- ainda nao incorporado neste framework."
        if is_simulado
        else "Nao ha, no momento, modelagem causal de uplift por canal (o efeito de "
        "acionar vs. nao acionar um contrato); a efetividade por canal e uma proxy "
        "arbitrada, nao estimada."
    )
    partes.append(f"""
<h2 class="section-title">9. Limitacoes e recomendacoes</h2>
<ul>
  <li>{escape(nota_fonte)}</li>
  <li>Janela de performance de apenas 1 mes (M -&gt; M+1); nao captura efeitos de
      acionamento que maturam em janelas mais longas.</li>
  <li>{escape(nota_self_cure)}</li>
  <li>A comparacao de canais (secao 8) e entre canais PUROS; uma regua real de
      cobranca e SEQUENCIAL (varios canais ao longo do ciclo de atraso de um mesmo
      contrato), o que nao e capturado pela comparacao atual.</li>
  <li>Recomenda-se validar a estabilidade do score (PSI) e recalibrar a matriz de
      efetividade com dados reais de resposta por canal antes de qualquer uso em
      producao.</li>
</ul>
""")

    partes.append(f"""
<footer>collscore &middot; relatorio gerado automaticamente por <code>collscore.report.build_html_report</code> em {escape(gerado_em)}.</footer>
</div>
</body>
</html>
""")

    return "".join(partes)

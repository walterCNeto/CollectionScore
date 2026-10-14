# collscore

Framework de collection scoring com base simulada e avaliacao financeira
centrada em regularizacao (pagamento de parcela). Inclui um pipeline CLI
(`run_demo.py`) e um app Streamlit interativo (`app.py`).

## Estrutura

- `collscore/simulate.py` -- simula uma carteira ativa de credito (12 fotos
  mensais, com roll rates por faixa de atraso e reposicao mensal).
- `collscore/features.py` -- book de features ponto-no-tempo por snapshot.
- `collscore/target.py` -- targets M -> M+1 (`pagou_1parcela` e outros,
  parametrizaveis).
- `collscore/model.py` -- regressao logistica pooled (features padronizadas
  + dummies de faixa de atraso), sem binning WoE.
- `collscore/metrics.py` -- metricas de regularizacao financeira por faixa,
  roll rate de saldo, decis de score, KS/Gini.
- `collscore/finance.py` -- custo por canal, piso de saldo elegivel, matriz
  de efetividade canal x faixa, curva de lucro, escolha endogena de canal
  (`compare_channels`/`optimal_strategy`) e fronteira orcamento x lucro.
- `collscore/report.py` -- relatorio HTML autocontido de desenvolvimento e
  validacao do modelo (`build_html_report`).
- `run_demo.py` -- pipeline ponta a ponta que gera `relatorio.md` e
  `relatorio_modelo.html`.
- `app.py` -- app Streamlit interativo sobre o mesmo pacote.
- `tests/` -- suite pytest.

## Instalacao

```bash
pip install -r requirements.txt
```

## Executar o pipeline CLI

```bash
python run_demo.py --n-contratos 50000 --seed 42 --output relatorio.md
```

## Executar o app Streamlit

```bash
streamlit run app.py
```

O app abre em `http://localhost:8501`. Todos os controles (fonte de dados,
upload, custos por canal, estrategia de canal, pisos de saldo, matriz de
efetividade) ficam dentro de um formulario na barra lateral -- nada
recalcula automaticamente. So ao clicar em **"Aplicar e rodar modelo"** o
pipeline (simulacao/upload -> features -> target -> treino -> metricas ->
financas) roda e os resultados ficam guardados em `st.session_state`; as
abas renderizam exclusivamente a partir dali. No primeiro carregamento, o
app roda uma vez com os defaults para nao abrir vazio.

- **Base de exemplo (simulada)**: `n_contratos` e `n_snapshots` via sliders;
  a simulacao e o treino do modelo ficam em cache (`st.cache_data`) para
  submits repetidos com os mesmos parametros.
- **Upload de base propria**: envia um CSV ou Parquet no formato de painel
  com as colunas `id_contrato, snapshot, faixa_atraso, saldo_devedor,
  saldo_atraso, parcela_valor, parcelas_pagas, parcelas_vencidas`. Um botao
  "Baixar template CSV" (fora do formulario) gera um exemplo no layout
  esperado.

Um botao **"Gerar relatorio HTML"** (fora do formulario, usa os resultados
vigentes) produz um relatorio de desenvolvimento e validacao autocontido
(objetivo, base de dados, target, variaveis, metodologia, desempenho,
estabilidade, uso financeiro ilustrativo do score e limitacoes), disponivel
para download logo em seguida.

## Testes

```bash
pytest tests/ -v
```

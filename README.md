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
- `collscore/finance.py` -- custo por canal, piso de saldo elegivel, curva
  de lucro e corte otimo por faixa.
- `run_demo.py` -- pipeline ponta a ponta que gera `relatorio.md`.
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

O app abre em `http://localhost:8501`. Na barra lateral e possivel escolher
entre:

- **Base de exemplo (simulada)**: ajusta `n_contratos` e `n_snapshots` via
  sliders; a simulacao e o treino do modelo ficam em cache
  (`st.cache_data`).
- **Upload de base propria**: envia um CSV ou Parquet no formato de painel
  com as colunas `id_contrato, snapshot, faixa_atraso, saldo_devedor,
  saldo_atraso, parcela_valor, parcelas_pagas, parcelas_vencidas`. Um botao
  "Baixar template CSV" gera um exemplo no layout esperado.

Os parametros financeiros (custo por canal, canal por faixa, saldo minimo
elegivel) sao aplicados ao vivo, sem precisar retreinar o modelo.

## Testes

```bash
pytest tests/ -v
```

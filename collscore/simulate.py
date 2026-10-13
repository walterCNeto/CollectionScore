"""Simulacao de carteira ativa de cobranca.

Gera um painel (contrato x foto mensal) de ~N contratos de credito (consignado,
credito pessoal "CP" e cartao) ao longo de 12 fotos mensais, com dinamica de
faixa de atraso governada por uma matriz de roll rates por faixa, modulada por
uma probabilidade de regularizacao (cura) derivada de um escore logistico
latente. O saldo devedor de cada contrato e sempre recalculado a partir do
numero de parcelas efetivamente pagas (tabela Price), o que garante
consistencia ("conservacao de massa") entre valor financiado, saldo devedor e
parcelas pagas em qualquer fotografia.

O componente de self-cure (cura espontanea, independente de acionamento) e
registrado em `flag_self_cure_evento` apenas como metadado para uso futuro em
modelagem de uplift; ele nao realimenta o escore de cura.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from collscore import FAIXAS, FAIXA_SEVERIDADE

N_SEVERIDADES = len(FAIXAS)

PRODUTOS = ["consignado", "CP", "cartao"]

# Parametros de originacao por produto: (valor_min, valor_max, prazo_min, prazo_max, taxa_min, taxa_max)
PRODUTO_PARAMS = {
    "consignado": dict(valor=(2000, 40000), prazo=(24, 84), taxa=(0.015, 0.025)),
    "CP": dict(valor=(500, 15000), prazo=(6, 48), taxa=(0.03, 0.06)),
    "cartao": dict(valor=(300, 8000), prazo=(1, 24), taxa=(0.08, 0.14)),
}

PRODUTO_PESOS = {"consignado": 0.35, "CP": 0.40, "cartao": 0.25}

# Distribuicao inicial (estoque) de faixas de atraso no m0 de uma carteira madura.
FAIXA_INICIAL_PESOS = {
    "em_dia": 0.82,
    "1-15": 0.06,
    "16-30": 0.04,
    "31-60": 0.03,
    "61-90": 0.02,
    "91-180": 0.018,
    "180+": 0.012,
}

# Dias de atraso tipicos (ponto medio) usados para inicializar dias_atraso por faixa.
FAIXA_DIAS_MEDIO = {
    "em_dia": 0,
    "1-15": 8,
    "16-30": 23,
    "31-60": 45,
    "61-90": 75,
    "91-180": 135,
    "180+": 220,
}

# Matriz de roll rates "base" (linha = faixa de origem, colunas = faixas destino,
# na ordem FAIXAS). Cada linha soma 1. Para faixas delinquentes, a massa de
# "melhora" (colunas com severidade menor que a origem) e modulada pela
# probabilidade de regularizacao individual antes da amostragem.
BASE_ROLL_MATRIX = np.array(
    [
        # em_dia -> em_dia, 1-15, 16-30, 31-60, 61-90, 91-180, 180+
        [0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        # 1-15
        [0.35, 0.25, 0.40, 0.00, 0.00, 0.00, 0.00],
        # 16-30
        [0.10, 0.15, 0.30, 0.45, 0.00, 0.00, 0.00],
        # 31-60
        [0.02, 0.03, 0.10, 0.35, 0.50, 0.00, 0.00],
        # 61-90
        [0.01, 0.02, 0.03, 0.09, 0.35, 0.50, 0.00],
        # 91-180
        [0.005, 0.01, 0.015, 0.02, 0.10, 0.35, 0.50],
        # 180+
        [0.002, 0.003, 0.005, 0.01, 0.02, 0.06, 0.90],
    ]
)

assert np.allclose(BASE_ROLL_MATRIX.sum(axis=1), 1.0), "linhas da matriz de roll rate devem somar 1"

# Probabilidade mensal (hazard) de quebra de promessa por faixa (0 p/ em_dia).
QUEBRA_PROMESSA_HAZARD = {
    "em_dia": 0.0,
    "1-15": 0.15,
    "16-30": 0.25,
    "31-60": 0.35,
    "61-90": 0.40,
    "91-180": 0.45,
    "180+": 0.50,
}

# Media de acionamentos de cobranca no mes, por faixa (Poisson).
ACIONAMENTOS_LAMBDA = {
    "em_dia": 0.0,
    "1-15": 0.8,
    "16-30": 1.5,
    "31-60": 2.5,
    "61-90": 3.2,
    "91-180": 3.8,
    "180+": 4.2,
}

SELF_CURE_PCT = 0.30  # fracao de curas totais partindo de faixas <=30d marcadas como self-cure
N_SNAPSHOTS = 12


def _parcela_valor(valor_financiado: np.ndarray, taxa: np.ndarray, prazo: np.ndarray) -> np.ndarray:
    """Parcela fixa (tabela Price): PMT = PV * i / (1 - (1+i)^-n)."""
    return valor_financiado * taxa / (1.0 - (1.0 + taxa) ** (-prazo))


def _saldo_devedor(parcela_valor: np.ndarray, taxa: np.ndarray, prazo: np.ndarray, parcelas_pagas: np.ndarray) -> np.ndarray:
    """Saldo devedor remanescente apos `parcelas_pagas` pagamentos em dia (tabela Price).

    Depende apenas de (parcela_valor, taxa, prazo, parcelas_pagas) -- nunca e
    decrementado incrementalmente -- o que garante que o saldo e sempre
    exatamente reproduzivel a partir do numero de parcelas pagas (consistencia
    testavel: "conservacao de massa").
    """
    n_restantes = np.maximum(prazo - parcelas_pagas, 0.0)
    saldo = parcela_valor * (1.0 - (1.0 + taxa) ** (-n_restantes)) / taxa
    return np.where(parcelas_pagas >= prazo, 0.0, saldo)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit_p_regularizacao(
    pct_saldo_atraso: np.ndarray,
    pct_parcelas_pagas: np.ndarray,
    recencia_pgto: np.ndarray,
    quebra_promessa: np.ndarray,
    mob_rel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Escore logistico latente de probabilidade de regularizacao (cura) do mes."""
    noise = rng.normal(0.0, 0.5, size=pct_saldo_atraso.shape[0])
    z = (
        -0.5
        - 2.2 * pct_saldo_atraso
        + 2.0 * pct_parcelas_pagas
        - 0.15 * recencia_pgto
        - 1.1 * quebra_promessa
        - 1.0 * mob_rel
        + noise
    )
    return _sigmoid(z)


def _init_portfolio(n_contratos: int, rng: np.random.Generator) -> pd.DataFrame:
    produtos = rng.choice(PRODUTOS, size=n_contratos, p=[PRODUTO_PESOS[p] for p in PRODUTOS])

    valor_financiado = np.empty(n_contratos)
    prazo = np.empty(n_contratos, dtype=int)
    taxa = np.empty(n_contratos)
    for p in PRODUTOS:
        mask = produtos == p
        n = mask.sum()
        vmin, vmax = PRODUTO_PARAMS[p]["valor"]
        pmin, pmax = PRODUTO_PARAMS[p]["prazo"]
        tmin, tmax = PRODUTO_PARAMS[p]["taxa"]
        valor_financiado[mask] = rng.uniform(vmin, vmax, size=n)
        prazo[mask] = rng.integers(pmin, pmax + 1, size=n)
        taxa[mask] = rng.uniform(tmin, tmax, size=n)

    mob_inicial = np.floor(rng.uniform(0, 1, size=n_contratos) * (prazo - 1)).astype(int)
    mob_inicial = np.clip(mob_inicial, 0, prazo - 1)

    faixa_inicial = rng.choice(FAIXAS, size=n_contratos, p=[FAIXA_INICIAL_PESOS[f] for f in FAIXAS])

    parcela_valor = _parcela_valor(valor_financiado, taxa, prazo)

    dias_atraso = np.array([FAIXA_DIAS_MEDIO[f] for f in faixa_inicial], dtype=float)
    dias_atraso *= rng.uniform(0.7, 1.3, size=n_contratos)
    dias_atraso = np.where(faixa_inicial == "em_dia", 0.0, dias_atraso)

    parcelas_atraso_inicial = np.ceil(dias_atraso / 30.0).astype(int)
    parcelas_atraso_inicial = np.minimum(parcelas_atraso_inicial, mob_inicial)
    parcelas_pagas = np.maximum(mob_inicial - parcelas_atraso_inicial, 0)

    ultimo_pagamento_mob = np.where(
        parcelas_atraso_inicial == 0, mob_inicial, mob_inicial - parcelas_atraso_inicial
    )

    df = pd.DataFrame(
        {
            "contrato_id": np.arange(n_contratos),
            "produto": produtos,
            "valor_financiado": valor_financiado,
            "prazo": prazo,
            "taxa": taxa,
            "parcela_valor": parcela_valor,
            "mob": mob_inicial,
            "parcelas_pagas": parcelas_pagas,
            "dias_atraso": dias_atraso,
            "faixa_atraso": faixa_inicial,
            "ultimo_pagamento_mob": ultimo_pagamento_mob,
        }
    )
    return df


def _faixa_from_severidade(idx: np.ndarray) -> np.ndarray:
    return np.array(FAIXAS, dtype=object)[idx]


def _severidade_from_faixa(faixa: np.ndarray) -> np.ndarray:
    return np.array([FAIXA_SEVERIDADE[f] for f in faixa])


def _step(state: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avanca o estado em um mes. Retorna (snapshot_atual_completo, novo_estado)."""
    n = len(state)
    mob = state["mob"].to_numpy()
    prazo = state["prazo"].to_numpy()
    parcelas_pagas = state["parcelas_pagas"].to_numpy().astype(float)
    parcelas_vencidas = mob.astype(float)  # uma parcela vence por mes de carteira
    parcela_valor = state["parcela_valor"].to_numpy()
    valor_financiado = state["valor_financiado"].to_numpy()
    taxa = state["taxa"].to_numpy()
    dias_atraso = state["dias_atraso"].to_numpy().astype(float)
    faixa_atraso = state["faixa_atraso"].to_numpy(dtype=object)
    ultimo_pagamento_mob = state["ultimo_pagamento_mob"].to_numpy().astype(float)

    severidade = _severidade_from_faixa(faixa_atraso)

    saldo_devedor = _saldo_devedor(parcela_valor, taxa, prazo, parcelas_pagas)
    parcelas_atraso = np.maximum(parcelas_vencidas - parcelas_pagas, 0.0)
    saldo_atraso = parcelas_atraso * parcela_valor

    pct_saldo_atraso = np.where(saldo_devedor > 0, np.clip(saldo_atraso / saldo_devedor, 0, 1), 0.0)
    pct_parcelas_pagas = np.where(parcelas_vencidas > 0, parcelas_pagas / np.maximum(parcelas_vencidas, 1), 1.0)
    recencia_pgto = mob - ultimo_pagamento_mob
    mob_rel = mob / np.maximum(prazo, 1)

    quebra_hazard = np.array([QUEBRA_PROMESSA_HAZARD[f] for f in faixa_atraso])
    quebra_promessa_mes = (rng.uniform(0, 1, size=n) < quebra_hazard).astype(int)

    acionamentos_lambda = np.array([ACIONAMENTOS_LAMBDA[f] for f in faixa_atraso])
    acionamentos_mes = rng.poisson(acionamentos_lambda)

    p_regularizacao = _logit_p_regularizacao(
        pct_saldo_atraso, pct_parcelas_pagas, recencia_pgto, quebra_promessa_mes, mob_rel, rng
    )

    # Snapshot (estado observado ANTES da transicao para o proximo mes).
    snapshot = pd.DataFrame(
        {
            "contrato_id": state["contrato_id"].to_numpy(),
            "produto": state["produto"].to_numpy(),
            "valor_financiado": valor_financiado,
            "prazo": prazo,
            "taxa": taxa,
            "mob": mob,
            "parcela_valor": parcela_valor,
            "parcelas_pagas": parcelas_pagas,
            "parcelas_vencidas": parcelas_vencidas,
            "parcelas_atraso": parcelas_atraso,
            "saldo_devedor": saldo_devedor,
            "saldo_atraso": saldo_atraso,
            "dias_atraso": dias_atraso,
            "faixa_atraso": faixa_atraso,
            "recencia_pgto": recencia_pgto,
            "quebra_promessa_mes": quebra_promessa_mes,
            "acionamentos_mes": acionamentos_mes,
            "p_regularizacao_mes": p_regularizacao,
        }
    )

    # --- Transicao de faixa para o proximo mes ---
    novo_severidade = severidade.copy()
    novo_parcelas_pagas = parcelas_pagas.copy()
    flag_self_cure = np.zeros(n, dtype=bool)

    em_dia_mask = severidade == 0
    delinq_mask = ~em_dia_mask

    # em_dia: pequena probabilidade de entrar em atraso; caso contrario paga em dia.
    if em_dia_mask.any():
        idx = np.where(em_dia_mask)[0]
        row = BASE_ROLL_MATRIX[0]
        draws = rng.uniform(0, 1, size=idx.shape[0])
        entra_atraso = draws >= row[0]
        novo_severidade[idx[entra_atraso]] = 1
        novo_parcelas_pagas[idx[~entra_atraso]] += 1

    # faixas delinquentes: roll rate modulado por p_regularizacao.
    for sev in range(1, N_SEVERIDADES):
        idx = np.where(delinq_mask & (severidade == sev))[0]
        if idx.shape[0] == 0:
            continue
        row = BASE_ROLL_MATRIX[sev]
        melhora_cols = np.arange(sev)
        stay_col = sev
        piora_cols = np.arange(sev + 1, N_SEVERIDADES)

        base_melhora_mass = row[melhora_cols].sum()
        base_stay_mass = row[stay_col]
        base_piora_mass = row[piora_cols].sum()

        p_ind = p_regularizacao[idx]
        p_media_faixa = p_ind.mean() if p_ind.mean() > 1e-6 else 1e-6
        fator = np.clip(p_ind / p_media_faixa, 0.2, 3.0)

        nova_melhora_mass = np.clip(base_melhora_mass * fator, 0.0, 0.97)
        resto = 1.0 - nova_melhora_mass
        denom = base_stay_mass + base_piora_mass
        if denom <= 1e-9:
            nova_stay_mass = resto
            nova_piora_mass = np.zeros_like(resto)
        else:
            nova_stay_mass = resto * (base_stay_mass / denom)
            nova_piora_mass = resto * (base_piora_mass / denom)

        draws = rng.uniform(0, 1, size=idx.shape[0])
        vai_melhorar = draws < nova_melhora_mass
        vai_piorar = draws >= (nova_melhora_mass + nova_stay_mass)
        # fica na mesma faixa = caso remanescente (nem melhora nem piora)

        if vai_melhorar.any():
            sub_idx = idx[vai_melhorar]
            probs_melhora = row[melhora_cols]
            probs_melhora = probs_melhora / probs_melhora.sum() if probs_melhora.sum() > 0 else None
            if probs_melhora is not None:
                destino = rng.choice(melhora_cols, size=sub_idx.shape[0], p=probs_melhora)
            else:
                destino = np.zeros(sub_idx.shape[0], dtype=int)
            novo_severidade[sub_idx] = destino
            # paga a(s) parcela(s) de atraso referentes a melhora + a parcela do mes corrente
            parcelas_atraso_origem = parcelas_atraso[sub_idx]
            delta_severidade = sev - destino
            frac_regulariza = np.clip(delta_severidade / sev, 0, 1)
            pagas_extra = np.ceil(parcelas_atraso_origem * frac_regulariza)
            novo_parcelas_pagas[sub_idx] += pagas_extra + 1

            cura_total_mask = destino == 0
            if cura_total_mask.any() and sev <= FAIXA_SEVERIDADE["16-30"]:
                cura_idx = sub_idx[cura_total_mask]
                self_cure_draw = rng.uniform(0, 1, size=cura_idx.shape[0]) < SELF_CURE_PCT
                flag_self_cure[cura_idx[self_cure_draw]] = True

        if vai_piorar.any():
            sub_idx = idx[vai_piorar]
            probs_piora = row[piora_cols]
            probs_piora = probs_piora / probs_piora.sum()
            destino = rng.choice(piora_cols, size=sub_idx.shape[0], p=probs_piora)
            novo_severidade[sub_idx] = destino

        fica_mask = ~vai_melhorar & ~vai_piorar
        if fica_mask.any():
            # permanece na mesma faixa, mas pode pagar a parcela corrente sem
            # regularizar o atraso acumulado (pagamento parcial / promessa
            # cumprida parcialmente) -- mantem pagou_1parcela desacoplado de
            # melhora_faixa.
            sub_idx = idx[fica_mask]
            p_pay_partial = np.clip(p_regularizacao[idx][fica_mask] * 0.6, 0.0, 0.9)
            paga_parcial = rng.uniform(0, 1, size=sub_idx.shape[0]) < p_pay_partial
            novo_parcelas_pagas[sub_idx[paga_parcial]] += 1

    nova_faixa = _faixa_from_severidade(novo_severidade)
    pagou_mes = novo_parcelas_pagas > parcelas_pagas
    novo_ultimo_pagamento_mob = np.where(pagou_mes, mob + 1, ultimo_pagamento_mob)

    novo_dias_atraso = np.where(
        novo_severidade == 0,
        0.0,
        np.array([FAIXA_DIAS_MEDIO[f] for f in nova_faixa]) * rng.uniform(0.85, 1.15, size=n),
    )

    snapshot["flag_self_cure_evento"] = flag_self_cure

    novo_state = state.copy()
    novo_state["mob"] = mob + 1
    novo_state["parcelas_pagas"] = np.minimum(novo_parcelas_pagas, prazo)
    novo_state["dias_atraso"] = novo_dias_atraso
    novo_state["faixa_atraso"] = nova_faixa
    novo_state["ultimo_pagamento_mob"] = novo_ultimo_pagamento_mob

    return snapshot, novo_state


def _originar_novos(n_novos: int, proximo_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera novos contratos (mob=0, em_dia) para reposicao mensal da carteira ativa."""
    novos = _init_portfolio(n_novos, rng)
    novos["contrato_id"] = np.arange(proximo_id, proximo_id + n_novos)
    novos["mob"] = 0
    novos["parcelas_pagas"] = 0
    novos["dias_atraso"] = 0.0
    novos["faixa_atraso"] = "em_dia"
    novos["ultimo_pagamento_mob"] = 0
    return novos


def simulate_portfolio(
    n_contratos: int = 50_000, n_snapshots: int = N_SNAPSHOTS, seed: int = 42
) -> pd.DataFrame:
    """Simula a carteira ativa e retorna um painel longo (contrato x snapshot mensal).

    Contratos que atingem `mob >= prazo` (liquidados) deixam de aparecer nos
    snapshots seguintes; novas originacoes (mob=0, em_dia) reposicionam a
    carteira a cada mes para manter o tamanho ativo proximo de `n_contratos`,
    como em uma carteira ativa real em regime estacionario.
    """
    rng = np.random.default_rng(seed)
    state = _init_portfolio(n_contratos, rng)
    proximo_id = n_contratos

    snapshots = []
    for m in range(n_snapshots):
        ativo_mask = state["mob"] < state["prazo"]
        state_ativo = state.loc[ativo_mask].reset_index(drop=True)
        if state_ativo.empty:
            break
        snap, novo_state_ativo = _step(state_ativo, rng)
        snap.insert(1, "snapshot", m)
        snapshots.append(snap)

        deficit = n_contratos - len(novo_state_ativo)
        if deficit > 0:
            novos = _originar_novos(deficit, proximo_id, rng)
            proximo_id += deficit
            novo_state_ativo = pd.concat([novo_state_ativo, novos], ignore_index=True)
        state = novo_state_ativo

    panel = pd.concat(snapshots, ignore_index=True)
    panel = panel.sort_values(["contrato_id", "snapshot"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    df = simulate_portfolio()
    print(df.shape)
    print(df.head())

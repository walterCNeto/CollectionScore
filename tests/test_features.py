import numpy as np

from collscore.features import FEATURE_COLS


def test_features_no_nan_or_inf(features_small):
    sub = features_small[FEATURE_COLS]
    assert not sub.isna().any().any(), sub.isna().sum()
    assert np.isfinite(sub.to_numpy()).all()


def test_features_ranges(features_small):
    f = features_small
    assert (f["pct_saldo_atraso"] >= 0).all() and (f["pct_saldo_atraso"] <= 1).all()
    assert (f["pct_parcelas_pagas"] >= 0).all() and (f["pct_parcelas_pagas"] <= 1.0001).all()
    assert (f["mob_rel"] >= 0).all() and (f["mob_rel"] <= 1.0001).all()
    assert (f["dias_atraso"] >= 0).all()
    assert (f["n_acionamentos_3m"] >= 0).all()
    assert (f["ticket_rel"] > 0).all()


def test_atraso_max_12m_is_rolling_max_per_contract(features_small):
    sub = features_small[features_small["contrato_id"] == features_small["contrato_id"].iloc[0]]
    sub = sub.sort_values("snapshot")
    expanding_max = sub["dias_atraso"].rolling(window=12, min_periods=1).max()
    assert np.allclose(sub["atraso_max_12m"].to_numpy(), expanding_max.to_numpy())

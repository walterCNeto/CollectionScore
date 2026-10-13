import numpy as np
import pandas as pd

from collscore.features import FEATURE_COLS
from collscore.model import WoEBinner, train_pooled


def test_woe_binner_transform_shapes(book_small):
    X = book_small.dropna(subset=["cura_total"])[FEATURE_COLS]
    y = book_small.dropna(subset=["cura_total"])["cura_total"]
    binner = WoEBinner(n_bins=5)
    X_woe = binner.fit_transform(X, y)
    assert X_woe.shape == X.shape
    assert np.isfinite(X_woe.to_numpy()).all()


def test_score_monotonic_with_pct_parcelas_pagas(book_small):
    """O escore de cura deve crescer com `pct_parcelas_pagas`: quanto maior a
    fracao de parcelas pagas em dia, maior a propensao a regularizar -- essa
    e a relacao causal mais forte embutida no gerador (coeficiente positivo
    no escore logistico latente de `simulate.py`).
    """
    data = book_small.dropna(subset=["cura_total"])
    modelo = train_pooled(data, FEATURE_COLS, "cura_total")
    scores = modelo.predict_proba(data)

    df = pd.DataFrame({"score": scores, "pct_parcelas_pagas": data["pct_parcelas_pagas"].to_numpy()})
    df["bucket"] = pd.qcut(df["pct_parcelas_pagas"], 4, duplicates="drop")
    medias = df.groupby("bucket", observed=True)["score"].mean().to_numpy()

    assert np.all(np.diff(medias) >= -1e-6), medias

    corr = df["score"].corr(df["pct_parcelas_pagas"], method="spearman")
    assert corr > 0.1, corr

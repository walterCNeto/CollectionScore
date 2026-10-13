import numpy as np
import pandas as pd

from collscore.features import FEATURE_COLS
from collscore.model import FittedModel, train_pooled


def test_train_pooled_returns_fitted_model_with_dummies(book_small):
    data = book_small.dropna(subset=["pagou_1parcela"])
    modelo = train_pooled(data, FEATURE_COLS, "pagou_1parcela")
    assert isinstance(modelo, FittedModel)
    assert len(modelo.dummy_cols) == data["faixa_atraso"].nunique() - 1

    scores = modelo.predict_proba(data)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_score_monotonic_with_pct_parcelas_pagas(book_small):
    """O escore de `pagou_1parcela` deve crescer com `pct_parcelas_pagas`:
    quanto maior a fracao de parcelas pagas em dia, maior a propensao a
    pagar a proxima parcela -- essa e a relacao causal mais forte embutida
    no gerador (coeficiente positivo no escore logistico latente de
    `simulate.py`, que tambem governa o pagamento parcial).
    """
    data = book_small.dropna(subset=["pagou_1parcela"])
    modelo = train_pooled(data, FEATURE_COLS, "pagou_1parcela")
    scores = modelo.predict_proba(data)

    df = pd.DataFrame({"score": scores, "pct_parcelas_pagas": data["pct_parcelas_pagas"].to_numpy()})
    df["bucket"] = pd.qcut(df["pct_parcelas_pagas"], 4, duplicates="drop")
    medias = df.groupby("bucket", observed=True)["score"].mean().to_numpy()

    assert np.all(np.diff(medias) >= -1e-6), medias

    corr = df["score"].corr(df["pct_parcelas_pagas"], method="spearman")
    assert corr > 0.1, corr

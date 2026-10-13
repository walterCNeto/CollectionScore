import pandas as pd
import pytest

from collscore.features import build_features
from collscore.simulate import simulate_portfolio
from collscore.target import compute_targets


@pytest.fixture(scope="session")
def panel_small() -> pd.DataFrame:
    return simulate_portfolio(n_contratos=2000, n_snapshots=12, seed=123)


@pytest.fixture(scope="session")
def features_small(panel_small) -> pd.DataFrame:
    return build_features(panel_small)


@pytest.fixture(scope="session")
def targets_small(panel_small) -> pd.DataFrame:
    return compute_targets(panel_small)


@pytest.fixture(scope="session")
def book_small(features_small, targets_small) -> pd.DataFrame:
    merged = features_small.merge(targets_small, on=["contrato_id", "snapshot"])
    return merged[merged["faixa_atraso"] != "em_dia"]

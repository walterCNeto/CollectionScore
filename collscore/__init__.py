"""collscore: framework de collection scoring com base simulada e avaliacao financeira."""

__version__ = "0.1.0"

FAIXAS = ["em_dia", "1-15", "16-30", "31-60", "61-90", "91-180", "180+"]
FAIXA_SEVERIDADE = {f: i for i, f in enumerate(FAIXAS)}

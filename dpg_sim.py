"""Simulation mit gemeinsamen Zufallsströmen (nur numpy).

Alle verglichenen Politiken sehen denselben Nachfragestrom (gemeinsame Zufallszahlen). Die Streuung einer Differenz ist dann viel kleiner als
die der Einzelwerte. Numpy-Ströme zu festem Zufalls-Seed sind nicht über Versionen stabil: Tests prüfen Bänder, nie einzelne Zahlen.
Zeitfolge wie in dpg_algorithm: Entscheidung, Nachfrage, Kosten, Ankunft, die Pipeline rückt auf.
"""

import numpy as np

import dpg_constants as C
from dpg_algorithm import poisson_pmf


def demand_stream(seed: int, periods: int, chains: int, *, mean=C.MEAN, dmax=C.DMAX):
    """Nachfrage (periods, chains) aus der abgeschnittenen Poisson-Verteilung; der Rest der Verteilung liegt auf dmax."""
    cdf = np.cumsum(poisson_pmf(mean, dmax))
    rng = np.random.default_rng(seed)
    return np.minimum(np.searchsorted(cdf, rng.random((periods, chains))), dmax).astype(np.int16)


def simulate(L, policy, stream, *, burn=C.SIM_BURN, Q=C.Q_MAX, XM=C.XM, hold=C.HOLD, penalty=C.PENALTY):
    """Mittlere Kosten je Periode für jede Kette (Array der Länge chains). Start leer; die ersten `burn` Perioden zählen nicht."""
    T, N = stream.shape
    x = np.zeros(N, dtype=np.int64)
    pipe = np.zeros((N, max(L - 1, 0)), dtype=np.int64)
    total = np.zeros(N)
    for t in range(T):
        a = policy(x, pipe)
        d = stream[t].astype(np.int64)
        if t >= burn:
            total += hold * np.maximum(x - d, 0) + penalty * np.maximum(d - x, 0)
        left = np.maximum(x - d, 0)
        if L == 1:
            x = np.minimum(left + a, XM)
        else:
            x = np.minimum(left + pipe[:, 0], XM)
            pipe = np.column_stack([pipe[:, 1:], a])
    return total / (T - burn)


def position(x, pipe):
    """Bestandsposition: Bestand plus alles Unterwegs."""
    return x + (pipe.sum(axis=1) if pipe.shape[1] else 0)


def base_stock(S, Q=C.Q_MAX):
    return lambda x, pipe: np.clip(S - position(x, pipe), 0, Q)


def capped(S, cap, Q=C.Q_MAX):
    return lambda x, pipe: np.minimum(np.clip(S - position(x, pipe), 0, Q), cap)


def table_policy(model, table):
    """Politik aus einer Tabelle (X, Pn) der Klasse LostSales als Funktion von (x, pipe)."""
    weights = np.array([model.A ** (model.L - 2 - j) for j in range(model.L - 1)], dtype=np.int64)

    def policy(x, pipe):
        idx = (pipe * weights).sum(axis=1) if model.L > 1 else np.zeros(len(x), dtype=np.int64)
        return table[x, idx]

    return policy


def mean_and_se(values, other=None):
    """Mittel und Standardfehler über die Ketten; mit `other` die gepaarte Differenz values - other (gleiche Ketten, gleicher Strom)."""
    z = values if other is None else values - other
    return float(z.mean()), float(z.std(ddof=1) / np.sqrt(len(z)))

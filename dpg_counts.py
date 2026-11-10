"""Zustandszahlen: wie viele Zustände eine exakte Rückwärtsrechnung (DP) für drei Wachstumsrichtungen aufzählen muss.

Alles reine Formeln, keine Rechnung. Aufzählung gegen Formel prüft tests/test_oracle_dpg.py (Q = 2, XM = 3, L = 1 bis 5);
die Zählung des Modells der DP steht in tests/test_algorithm.py (`LostSales.S`).

Richtung 1, Lieferzeit (verlorene Nachfrage): Zustand = Bestand 0..XM und je Periode Lieferzeit eine Bestellung 0..Q unterwegs, bis auf die
   Bestellung der laufenden Periode, also (XM + 1) * (Q + 1)^(L - 1). Jede Periode Lieferzeit multipliziert mit Q + 1.
Richtung 2, Haltbarkeit (verderbliche Ware, Bestand je Restlebensdauer): m Altersklassen, bei Lieferzeit 0 sind es m - 1 Zahlen 0..C, bei
   Lieferzeit 1 sind es m Zahlen: (C + 1)^(m - 1) bzw. (C + 1)^m.
Richtung 3, zwei Stufen in Reihe (Aufbau der Demo „Lager und Händler“): Händlerbestand mit 56 Werten, Lagerbestand 0..18, dazu die
   Bestellungen unterwegs: 56 * 19 * (Q + 1)^(Lr + Lw - 2), wenn der Händler Lr und das Lager Lw Perioden Lieferzeit hat.
"""

import dpg_constants as C

DEALER_VALUES, WAREHOUSE_VALUES = 56, 19     # Händlerbestand −30..25, Lagerbestand 0..18 (Aufbau der Demo „Lager und Händler“)
PERISHABLE_CAP = 8                           # Bestandsgrenze je Altersklasse


def lost_sales_states(L: int, *, Q: int = C.Q_MAX, XM: int = C.XM) -> int:
    if L < 1:
        raise ValueError("Lieferzeit L >= 1")
    return (XM + 1) * (Q + 1) ** (L - 1)


def perishable_states(m: int, lead: int, *, cap: int = PERISHABLE_CAP) -> int:
    if m < 2 or lead not in (0, 1):
        raise ValueError("Haltbarkeit m >= 2, Lieferzeit 0 oder 1")
    return (cap + 1) ** (m - 1 + lead)


def two_stage_states(lr: int, lw: int, *, Q: int = C.Q_MAX) -> int:
    if lr < 1 or lw < 1:
        raise ValueError("beide Lieferzeiten >= 1")
    return DEALER_VALUES * WAREHOUSE_VALUES * (Q + 1) ** (lr + lw - 2)


def state_count(kind: str, **kw) -> int:
    """Einheit „Zustandszahl“: kind = 'lieferzeit' (L, Q, XM), 'haltbarkeit' (m, lead) oder 'zwei_stufen' (lr, lw)."""
    if kind == "lieferzeit":
        return lost_sales_states(kw["L"], Q=kw.get("Q", C.Q_MAX), XM=kw.get("XM", C.XM))
    if kind == "haltbarkeit":
        return perishable_states(kw["m"], kw["lead"])
    if kind == "zwei_stufen":
        return two_stage_states(kw["lr"], kw["lw"], Q=kw.get("Q", C.Q_MAX))
    raise ValueError(kind)


def state_pairs(states: int, actions: int = C.Q_MAX + 1) -> int:
    """Zustand-Aktion-Paare: so viele Werte muss ein Rückwärtsschritt bewerten."""
    return states * actions


def table_bytes(states: int) -> int:
    """Speicher einer Wertetabelle: 8 Byte je Zustand (die Rechnung hält mehrere solche Tabellen)."""
    return states * C.BYTES_PER_VALUE


def format_bytes(n: int) -> str:
    """Größe mit Einheit (Dezimalkomma); ab 0,5 GB in GB, damit 803,5 MB als „0,8 GB“ gelesen wird."""
    for unit, size, start in (("GB", 1e9, 5e8), ("MB", 1e6, 1e6), ("KB", 1e3, 1e3)):
        if n >= start:
            return f"{n / size:.1f} {unit}".replace(".", ",")
    return f"{n} Byte"


def format_count(n: int) -> str:
    """Zahl mit Punkt als Tausendertrenner (deutsche Schreibweise)."""
    return f"{n:,}".replace(",", ".")


def state_index(x: int, pipe, Q: int) -> int:
    """Index eines Zustands (x, p1, ..., p_{L-1}): x höchstwertig, dann p1 (wie in LostSales)."""
    idx = 0
    for v in pipe:
        idx = idx * (Q + 1) + v
    return x * (Q + 1) ** len(tuple(pipe)) + idx

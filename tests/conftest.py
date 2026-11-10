"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Mini-Instanz (Handbeispiel): Lieferzeit 2, Bestellmenge 0 oder 1 (Q = 1), Bestand höchstens 2 (XM = 2), Nachfrage 0, 1, 2 mit Wahrscheinlichkeit
1/4, 1/2, 1/4, Lagerkosten 1, Fehlmengenkosten 9, verlorene Nachfrage. Zustand (x, p1): 3 · 2 = 6 Zustände.
Handrechnung der erwarteten Periodenkosten: ec(0) = 9 · E[D] = 9, ec(1) = ¼ · 1 + ¼ · 9 = 2,5, ec(2) = 2 · ¼ + 1 · ½ = 1.
"""

import os
import sys
from pathlib import Path

# Vor dem ersten numpy-Import: ein Rechenthread je Matrixprodukt. Die Matrizen der DP sind klein; mehrere BLAS-Threads bremsen sie auf einem
# ausgelasteten Rechner um mehr als das Zehnfache (gemessen: L = 5 in 23 s statt 1,4 s, während andere Prozesse liefen).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dpg_algorithm as A  # noqa: E402

MINI_PD = [0.25, 0.5, 0.25]


@pytest.fixture
def mini():
    return A.LostSales(L=2, Q=1, XM=2, pd=MINI_PD, hold=1.0, penalty=9.0)


def mini_model(L=2):
    return A.LostSales(L=L, Q=1, XM=2, pd=MINI_PD, hold=1.0, penalty=9.0)

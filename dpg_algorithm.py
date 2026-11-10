"""Kern (nur numpy): exakte Rechnung für verlorene Nachfrage mit Lieferzeit L.

Zustand = (x, p1, ..., p_{L-1}): Bestand x und die Bestellungen, die noch unterwegs sind (p1 kommt in der nächsten Periode an).
Je Periode: Entscheidung a (Bestellmenge 0 bis Q, kommt nach L Perioden an), Nachfrage d, Kosten h*(x-d)^+ + p*(d-x)^+, danach
x' = min((x-d)^+ + p1, XM), die Pipeline rückt auf: (p2, ..., p_{L-1}, a). Bei L = 1 ist p1 = a (Ankunft in der nächsten Periode).

Die Nachfrage wirkt nur auf (x, p1), der Rest der Pipeline schiebt sich. Mit K[(x, p1), y] = P(x' = y) ist der Wertiterationsschritt
für Aktion a eine einzige dichte Matrixmultiplikation (K @ h[:, a::Q+1]); es braucht keine Übergangsmatrix je Aktion.

Einheiten (DEMO-PLAYBOOK §1b), je mit Mini-Instanz in tests/test_algorithm.py von Hand gerechnet:
  poisson_pmf, expected_cost, LostSales.finite_horizon (Rückwärtsrechnung), LostSales.rvi (relative Wertiteration),
  LostSales.evaluate (exakte Kosten einer festen Politik), policy_* (Regeln als Tabellen), reach_count, edge_mass.
"""

import time
from math import exp, factorial

import numpy as np

import dpg_constants as C


def poisson_pmf(mean: float, dmax: int) -> np.ndarray:
    """Poisson(mean) auf 0..dmax; der Rest der Wahrscheinlichkeit liegt auf dmax (Nachfrage bei dmax abgeschnitten)."""
    p = np.array([exp(-mean) * mean**k / factorial(k) for k in range(dmax + 1)])
    p[-1] += 1.0 - p.sum()
    return p


def expected_cost(x: int, pd, hold: float, penalty: float) -> float:
    """Erwartete Periodenkosten bei Bestand x: hold*E[(x-D)^+] + penalty*E[(D-x)^+]."""
    d = np.arange(len(pd))
    return float((np.asarray(pd) * (hold * np.maximum(x - d, 0) + penalty * np.maximum(d - x, 0))).sum())


class LostSales:
    """Verlorene Nachfrage, Lieferzeit L >= 1, Bestellmenge 0..Q, Bestandsgrenze XM. Alle Einstellungen nur mit Schlüsselwörtern."""

    def __init__(self, *, L, mean=C.MEAN, hold=C.HOLD, penalty=C.PENALTY, Q=C.Q_MAX, XM=C.XM, dmax=C.DMAX, pd=None):
        if L < 1:
            raise ValueError("Die Lieferzeit L muss mindestens 1 sein")
        self.L, self.Q, self.XM = L, Q, XM
        self.hold, self.penalty = hold, penalty
        self.pd = poisson_pmf(mean, dmax) if pd is None else np.asarray(pd, dtype=float)
        self.d = np.arange(len(self.pd))
        self.X, self.A = XM + 1, Q + 1
        self.Pn = self.A ** (L - 1)                        # Zahl der Pipeline-Zustände
        self.M = self.A ** (L - 2) if L >= 2 else 1        # Rest der Pipeline nach p1
        self.S = self.X * self.Pn                          # Zahl der Zustände
        self.ec = np.array([expected_cost(x, self.pd, hold, penalty) for x in range(self.X)])
        K = np.zeros((self.X * self.A, self.X))
        for x in range(self.X):
            for p1 in range(self.A):
                for d, pr in zip(self.d, self.pd):
                    K[x * self.A + p1, min(max(x - d, 0) + p1, XM)] += pr
        self.K = K
        sums = np.zeros(1, dtype=np.int64)
        for _ in range(L - 1):
            sums = (sums[:, None] + np.arange(self.A)[None, :]).reshape(-1)
        self.ip = np.arange(self.X)[:, None] + sums[None, :]   # Bestandsposition je Zustand (X, Pn): Bestand plus alles Unterwegs

    # ---- ein Schritt der Rückwärtsrechnung ------------------------------------------------------------------------------------------------
    def _next_value(self, h, a):
        """E[h(x', pipe')] für Aktion a, Form (X, Pn); h hat die Form (X, Pn)."""
        if self.L == 1:
            return (self.K[a::self.A] @ h[:, 0])[:, None]
        return (self.K @ h[:, a::self.A]).reshape(self.X, self.Pn)

    def finite_horizon(self, periods: int):
        """Rückwärtsrechnung über `periods` Perioden mit Endwert 0: Listen der Werte V_1..V_n und der besten Aktionen je Zustand.

        V_t(s) = ec(x) + min_a E[V_{t-1}(s')]; gibt (V, A) zurück, beide Listen mit Einträgen der Form (X, Pn); V[t-1] gehört zu t Perioden Restlaufzeit."""
        v_prev = np.zeros((self.X, self.Pn))
        values, actions = [], []
        for _ in range(periods):
            q = np.stack([self.ec[:, None] + self._next_value(v_prev, a) for a in range(self.A)])
            actions.append(q.argmin(axis=0).astype(np.int16))
            v_prev = q.min(axis=0)
            values.append(v_prev)
        return values, actions

    # ---- relative Wertiteration (durchschnittliche Kosten) --------------------------------------------------------------------------------
    def rvi(self, tol=1e-9, maxit=20000):
        """Relative Wertiteration, Bezugszustand (Bestand 0, leere Pipeline). Gibt (g, Politik (X, Pn), Wertetabelle, Iterationen).

        g sind die optimalen durchschnittlichen Kosten je Periode. Bei Gleichstand gewinnt die kleinere Bestellmenge."""
        h = np.zeros((self.X, self.Pn))
        for it in range(1, maxit + 1):
            v, pol = None, None
            for a in range(self.A):
                q = self.ec[:, None] + self._next_value(h, a)
                if v is None:
                    v, pol = q, np.zeros(q.shape, dtype=np.int16)
                else:
                    better = q < v - 1e-15
                    v = np.where(better, q, v)
                    pol[better] = a
            g = float(v[0, 0])
            h_new = v - g
            diff = float(np.max(np.abs(h_new - h)))
            h = h_new
            if diff < tol:
                break
        return g, pol, h, it

    # ---- exakte Auswertung einer festen Politik -------------------------------------------------------------------------------------------
    def stationary(self, pol, tol=1e-13, maxit=400000):
        """Stationäre Verteilung der Kette unter der Politik pol (Form (X, Pn)); träge Kette (halbe Masse bleibt stehen), damit sie aperiodisch ist."""
        pi = np.full((self.X, self.Pn), 1.0 / self.S)
        masks = [(pol == a) for a in range(self.A)]
        used = [a for a in range(self.A) if masks[a].any()]
        for it in range(1, maxit + 1):
            new = np.zeros_like(pi)
            for a in used:
                mass = np.where(masks[a], pi, 0.0)
                if self.L == 1:
                    new[:, 0] += self.K[a::self.A].T @ mass[:, 0]
                else:
                    moved = self.K.T @ mass.reshape(self.X * self.A, self.M)   # (X, M): neuer Bestand, Rest der Pipeline
                    new[:, a::self.A] += moved                                 # die neue Bestellung a wird die letzte Stelle
            new = 0.5 * pi + 0.5 * new
            if np.abs(new - pi).sum() < tol:
                pi = new
                break
            pi = new
        return pi / pi.sum(), it

    def evaluate(self, pol, tol=1e-13):
        """Durchschnittskosten je Periode einer festen Politik; bei verlorener Nachfrage hängen die Kosten nur am Bestand."""
        pi, _ = self.stationary(pol, tol=tol)
        return float((pi.sum(axis=1) * self.ec).sum()), pi

    # ---- Regeln als Tabellen --------------------------------------------------------------------------------------------------------------
    def policy_base_stock(self, S):
        """Bestelle bis zur Bestandsposition S (Bestand plus Unterwegs), höchstens Q."""
        return np.clip(S - self.ip, 0, self.Q).astype(np.int16)

    def policy_capped(self, S, cap):
        """Capped Base-Stock: wie Base-Stock, aber höchstens `cap` Stück je Bestellung."""
        return np.minimum(self.policy_base_stock(S), cap).astype(np.int16)

    def policy_constant(self, q):
        return np.full((self.X, self.Pn), q, dtype=np.int16)

    def best_base_stock(self, s_range):
        results = [(self.evaluate(self.policy_base_stock(s))[0], s) for s in s_range]
        return min(results)

    def best_capped(self, s_range, caps):
        results = [(self.evaluate(self.policy_capped(s, cap))[0], s, cap) for cap in caps for s in s_range]
        return min(results)

    def edge_mass(self, pi):
        """Anteil der Zeit bei Bestand = XM (Randmasse der Bestandsgrenze)."""
        return float(pi[self.XM].sum())

    def order_cap_mass(self, pi, pol):
        """Anteil der Zeit, in dem die Politik die Bestellgrenze Q ausschöpft."""
        return float(pi[pol == self.Q].sum())


def max_order(pi, pol, threshold=C.REACH_THRESHOLD) -> int:
    """Größte Bestellmenge, die die Politik in Zuständen mit mehr als `threshold` Anteil der Zeit wählt."""
    return int(pol[pi > threshold].max())


def reach_count(pi, threshold=C.REACH_THRESHOLD) -> int:
    """Zahl der Zustände, die unter der Politik mit mehr als `threshold` Anteil der Zeit besucht werden."""
    return int((pi > threshold).sum())


def timed_rvi(model: LostSales, tol=1e-9):
    """DP mit Zeitmessung: (g, Politik, Iterationen, Sekunden)."""
    t0 = time.perf_counter()
    g, pol, _, it = model.rvi(tol=tol)
    return g, pol, it, time.perf_counter() - t0

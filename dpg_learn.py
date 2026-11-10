"""Gelerntes Netz (Politikgradient mit gelerntem Basiswert, Adam), nur numpy.

Die App rechnet diese Läufe nicht selbst: ein Lauf mit 3.000 Updates (7,7e7 simulierte Perioden) braucht Minuten. Die Ergebnisse liegen in
dpg_data/curves.json, erzeugt von scripts/precompute_curves.py mit genau diesem Code. Hier stehen außerdem die Gradient-Bausteine, die
tests/test_oracle_dpg.py gegen endliche Differenzen prüft, und das Handbeispiel des Politikgradienten (Brüche).

Netz: zwei Netze (Strategie mit Softmax über die Bestellmengen 0..Q, Kritiker als Basiswert) mit je einer versteckten Schicht aus 32 tanh-Einheiten.
Eingabe je Zustand: Bestand/XM, die Bestellungen unterwegs/Q und die Bestandsposition, jeweils skaliert.
Je Update: 256 Ketten à 100 Perioden, Rückgabe = abgezinste Kosten-bis-Ende (Diskont 0,98), Vorteil = Rückgabe minus Kritiker, normiert,
Entropiebonus 0,01, Lernrate 3e-3. Das ist kein erwartungstreuer Gradient (Diskontierung, Normierung, Entropie, Adam): geprüft wird der
Rückwärtsschritt des Netzes, der Rest durch die Lernkurve.
"""

import time

import numpy as np

import dpg_constants as C
import dpg_sim
from dpg_algorithm import LostSales, poisson_pmf


class MLP:
    """Ein Netz mit einer versteckten tanh-Schicht und Adam-Schritt."""

    def __init__(self, nin, nhidden, nout, rng, scale=0.3):
        self.W1 = rng.normal(0, scale, (nin, nhidden))
        self.b1 = np.zeros(nhidden)
        self.W2 = rng.normal(0, scale, (nhidden, nout))
        self.b2 = np.zeros(nout)
        self.m = [np.zeros_like(a) for a in self.params()]
        self.v = [np.zeros_like(a) for a in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, f):
        """Vorwärtsschritt mit Zwischenspeicher für den Rückwärtsschritt."""
        self.f = f
        self.hid = np.tanh(f @ self.W1 + self.b1)
        return self.hid @ self.W2 + self.b2

    def apply(self, f):
        """Vorwärtsschritt ohne Zwischenspeicher."""
        return np.tanh(f @ self.W1 + self.b1) @ self.W2 + self.b2

    def gradients(self, dout):
        """Rückwärtsschritt: Gradienten (W1, b1, W2, b2) für die Ableitung der Zielfunktion nach den Ausgaben (dout), zum letzten forward()."""
        gW2 = self.hid.T @ dout
        gb2 = dout.sum(0)
        dh = (dout @ self.W2.T) * (1 - self.hid**2)
        gW1 = self.f.T @ dh
        gb1 = dh.sum(0)
        return [gW1, gb1, gW2, gb2]

    def adam(self, grads, lr):
        self.t += 1
        for i, (prm, g) in enumerate(zip(self.params(), grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9**self.t)
            vh = self.v[i] / (1 - 0.999**self.t)
            prm -= lr * mh / (np.sqrt(vh) + 1e-8)

    def backward_step(self, dout, lr):
        self.adam(self.gradients(dout), lr)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score_logit_gradient(pr, actions, weights):
    """Ableitung von weights * (log pi(a))-Score nach den Logits, Zeile für Zeile: weights * (onehot(a) - pr).

    Mit weights = Kosten-bis-Ende (minus ein Basiswert, der nur vom Zustand abhängt) ist das die REINFORCE-Ableitung der erwarteten Kosten."""
    onehot = np.zeros_like(pr)
    onehot[np.arange(len(actions)), actions] = 1.0
    return weights[:, None] * (onehot - pr)


def entropy_logit_gradient(pr, scale=1.0):
    """Ableitung von scale * (−Entropie) nach den Logits: scale * pr * (log pr + H)."""
    lp = np.log(pr + 1e-12)
    H = -(pr * lp).sum(axis=1, keepdims=True)
    return scale * pr * (lp + H)


def score_gradient_hand(pi, costs):
    """Exakte Ableitung der erwarteten Kosten sum_a pi_a c_a nach den Logits eines Softmax (Handbeispiel): pi_a * (c_a - mittlere Kosten)."""
    mean = sum(p * c for p, c in zip(pi, costs))
    return [p * (c - mean) for p, c in zip(pi, costs)]


def reinforce_expectation(pi, costs):
    """Erwartungswert der REINFORCE-Schätzung c_A * (onehot(A) - pi) über A ~ pi, ohne Basiswert (Handbeispiel); gleich score_gradient_hand."""
    n = len(pi)
    return [sum(pi[b] * costs[b] * ((1 if a == b else 0) - pi[a]) for b in range(n)) for a in range(n)]


def features(x, pipe, L, XM, Q):
    """Eingabe des Netzes je Zustand (Zeilen): Bestand/XM, Bestellungen unterwegs/Q, Bestandsposition skaliert."""
    ip = x + (pipe.sum(axis=1) if L > 1 else 0)
    cols = [x / XM] + ([pipe / Q] if L > 1 else []) + [ip / (XM + Q * (L - 1))]
    return np.column_stack(cols)


GRID_SHORT = (0, 50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750)   # Messpunkte (Updates), L <= 5
GRID_LONG = (0, 100, 250, 500, 1000, 1500, 2000, 2500)                                                # Messpunkte, L >= 6


def measurement_grid(L):
    return GRID_LONG if L >= 6 else GRID_SHORT


class Evaluator:
    """Bewertet die gierige Politik des Netzes: exakt (DP-Modell, L <= 6) oder mit der Simulation auf dem Strom 777 (L >= 7, und L = 6 vor Update 500)."""

    def __init__(self, L, *, XM=C.XM, mode=None):
        self.L, self.XM, self.Q = L, XM, C.Q_MAX
        self.mode = mode or ("exakt" if L <= 6 else "sim")
        self.model = LostSales(L=L, XM=XM) if self.mode == "exakt" else None
        self.stream = dpg_sim.demand_stream(C.DEFAULT_SIM_SEED, C.SIM_PERIODS, C.SIM_CHAINS) if (self.mode == "sim" or L >= 6) else None
        if self.model is not None:
            n = self.model.Pn
            self.digits = np.zeros((n, max(L - 1, 0)), dtype=np.int64)
            idx = np.arange(n)
            for j in range(L - 1):
                self.digits[:, j] = (idx // (self.model.A ** (L - 2 - j))) % self.model.A

    def table(self, actor):
        m = self.model
        pol = np.zeros((m.X, m.Pn), dtype=np.int16)
        for x in range(m.X):
            f = features(np.full(m.Pn, x), self.digits, self.L, self.XM, self.Q)
            pol[x] = actor.apply(f).argmax(axis=1)
        return pol

    def __call__(self, actor, update):
        """(Kosten, Standardfehler, Art der Bewertung) der gierigen Politik nach `update` Updates."""
        L = self.L
        if self.mode == "exakt" and not (L >= 6 and update < 500):
            g, _ = self.model.evaluate(self.table(actor), tol=1e-11)
            return g, 0.0, "exakt"
        if self.mode == "exakt":
            policy = dpg_sim.table_policy(self.model, self.table(actor))
            kind = "sim"
        else:
            policy = lambda x, pipe: actor.apply(features(x, pipe, L, self.XM, self.Q)).argmax(axis=1)  # noqa: E731
            kind = self.mode
        v = dpg_sim.simulate(L, policy, self.stream, XM=self.XM)
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), kind


def train(*, L, seed, updates=C.UPDATES, XM=C.XM, evaluator=None, grid=None, on_point=None):
    """Ein Lernlauf. Gibt die Liste der Messpunkte zurück (Dictionaries mit update, periods, g, se, mode, sek).

    Bei gleichem Zufalls-Seed, gleicher Python- und numpy-Version und gleichem Rechner ergibt derselbe Aufruf dieselben Zahlen; über
    Versionen hinweg ist das nicht garantiert (Reihenfolge der Gleitkommaoperationen, Zufallsströme)."""
    Q, A = C.Q_MAX, C.Q_MAX + 1
    B, T = C.CHAINS_PER_UPDATE, C.PERIODS_PER_UPDATE
    rng = np.random.default_rng(seed)
    nin = 1 + (L - 1) + 1
    actor = MLP(nin, C.HIDDEN, A, rng)
    critic = MLP(nin, C.HIDDEN, 1, rng, 0.1)
    cdf = np.cumsum(poisson_pmf(C.MEAN, C.DMAX))
    evaluator = evaluator or Evaluator(L, XM=XM)
    grid = set(measurement_grid(L) if grid is None else grid)
    points, t0 = [], time.time()

    def log(u):
        g, se, mode = evaluator(actor, u)
        rec = dict(update=u, periods=u * C.PERIODS_PER_STEP, g=g, se=se, mode=mode, sek=round(time.time() - t0))
        points.append(rec)
        if on_point:
            on_point(rec)

    for u in range(updates):
        if u in grid:
            log(u)
        x = np.zeros(B, dtype=int)
        pipe = np.zeros((B, max(L - 1, 0)), dtype=int)
        F, Acts, Cs = [], [], []
        for _ in range(T):
            f = features(x, pipe, L, XM, Q)
            pr = softmax(actor.forward(f))
            a = (pr.cumsum(axis=1) > rng.random((B, 1))).argmax(axis=1)
            d = np.minimum(np.searchsorted(cdf, rng.random(B)), C.DMAX)
            cost = C.HOLD * np.maximum(x - d, 0) + C.PENALTY * np.maximum(d - x, 0)
            F.append(f)
            Acts.append(a)
            Cs.append(cost)
            left = np.maximum(x - d, 0)
            if L == 1:
                x = np.minimum(left + a, XM)
            else:
                x = np.minimum(left + pipe[:, 0], XM)
                pipe = np.column_stack([pipe[:, 1:], a])
        Fm, Am, Cm = np.concatenate(F), np.concatenate(Acts), np.array(Cs)
        G = np.zeros_like(Cm)
        run = np.zeros(B)
        for t in range(T - 1, -1, -1):
            run = Cm[t] + C.GAMMA * run
            G[t] = run
        G = -G.reshape(-1)                                          # Belohnung = −Kosten
        V = critic.forward(Fm)[:, 0]
        adv = G - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        critic.backward_step(((V - G) / len(G))[:, None] * 2.0 / (np.abs(G).std() + 1e-8), C.LEARNING_RATE)
        pr = softmax(actor.forward(Fm))
        dz = (score_logit_gradient(pr, Am, -adv) + entropy_logit_gradient(pr, C.ENTROPY)) / len(Am)
        actor.backward_step(dz, C.LEARNING_RATE)
    log(updates)
    return points

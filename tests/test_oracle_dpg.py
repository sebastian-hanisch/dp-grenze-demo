"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2): jeder Kern gegen einen ANDEREN Rechenweg, zusammen unter etwa 10 s.

O1  Wertiteration mit Tensor-Schritt (dpg_algorithm) gegen Politik-Iteration auf einer dichten Übergangsmatrix, aufgebaut durch Aufzählung aller
    Zustandstupel (anderer Algorithmus, andere Datenstruktur), L = 1 bis 4 auf kleinen Instanzen
O2  exakte Auswertung fester Politiken (Base-Stock, Capped, konstant, DP-Politik) gegen lineares Gleichungssystem der dichten Kette
O3  Simulation mit gemeinsamen Zufallsströmen gegen exakte Auswertung (Band 4 Standardfehler)
O4  Rückwärtsschritt des Netzes gegen zentrale endliche Differenzen, exakt über alle 216 Pfade einer winzigen Instanz; Zweigprobe mit falschem Basiswert
O5  Zählformeln gegen Aufzählung aller Tupel; Index bijektiv
O6  Handbeispiel in Brüchen (fractions): Rückwärtsrechnung, Gradient −13/8
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest

import dpg_algorithm as A
import dpg_counts as N
import dpg_evaluation as E
import dpg_learn as LN
import dpg_sim as S


# ---- O1 und O2: dichte Referenz --------------------------------------------------------------------------------------------------------------

def dense_chain(L, Q, XM, pd, hold=1.0, penalty=9.0):
    """Zustände als Tupel (x, p1, ..., p_{L-1}) in Reihenfolge von itertools.product; Übergangsmatrix je Bestellmenge und Kosten je Zustand (hängen nur an x)."""
    states = list(itertools.product(range(XM + 1), *[range(Q + 1)] * (L - 1)))
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((Q + 1, n, n))
    cost = np.zeros(n)
    for s in states:
        x, pipe = s[0], s[1:]
        cost[index[s]] = sum(pr * (hold * max(x - d, 0) + penalty * max(d - x, 0)) for d, pr in enumerate(pd))
        for a in range(Q + 1):
            for d, pr in enumerate(pd):
                left = max(x - d, 0)
                if L == 1:
                    nxt = (min(left + a, XM),)
                else:
                    nxt = (min(left + pipe[0], XM),) + pipe[1:] + (a,)
                P[a, index[s], index[nxt]] += pr
    return states, P, cost


def poisson_trunc(mean, dmax):
    pd = [np.exp(-mean) * mean**k / np.prod(range(1, k + 1)) for k in range(dmax + 1)]
    pd[-1] += 1.0 - sum(pd)
    return pd


def gain_of_policy(P, cost, policy):
    """Durchschnittskosten und stationäre Verteilung (Einzelklasse): pi (P - I) = 0, sum pi = 1 per kleinster Quadrate."""
    n = len(cost)
    Pp = np.array([P[policy[i], i] for i in range(n)])
    A_ = np.vstack([Pp.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A_, b, rcond=None)[0]
    return float(pi @ cost), pi


def policy_iteration(P, cost, max_rounds=100):
    """Howard: Politik bewerten (Poisson-Gleichung g + h = c + P h mit h[0] = 0), verbessern, bis sich nichts mehr ändert."""
    A_n, n, _ = P.shape
    policy = np.zeros(n, dtype=int)
    for _ in range(max_rounds):
        Pp = np.array([P[policy[i], i] for i in range(n)])
        M = np.zeros((n, n))
        M[:, 0] = 1.0                                            # Unbekannte 0: g
        M[:, 1:] = (np.eye(n) - Pp)[:, 1:]                       # Unbekannte 1..n-1: h[1..], h[0] = 0
        sol = np.linalg.solve(M, cost)
        g, h = sol[0], np.concatenate([[0.0], sol[1:]])
        q = np.array([cost + P[a] @ h for a in range(A_n)])
        new = q.argmin(axis=0)
        keep = np.isclose(q[policy, np.arange(n)], q.min(axis=0), rtol=0, atol=1e-12)
        new = np.where(keep, policy, new)                        # bei Gleichstand die alte Aktion behalten (Abbruch sicher)
        if (new == policy).all():
            return float(g), policy
        policy = new
    raise AssertionError("Politik-Iteration konvergiert nicht")


@pytest.mark.parametrize("L,Q,XM,dmax", [(1, 3, 8, 8), (2, 3, 8, 8), (3, 2, 7, 8), (4, 2, 6, 8)])
def test_o1_value_iteration_against_policy_iteration(L, Q, XM, dmax):
    pd = poisson_trunc(3.0, dmax)
    states, P, cost = dense_chain(L, Q, XM, pd)
    g_ref, pol_ref = policy_iteration(P, cost)
    m = A.LostSales(L=L, Q=Q, XM=XM, pd=pd)
    g, pol, _, _ = m.rvi(tol=1e-12)
    assert g == pytest.approx(g_ref, abs=1e-9)
    # die Politik der Wertiteration ist ebenfalls optimal: ihre exakte Auswertung auf der dichten Kette liefert g
    g_pol, _ = gain_of_policy(P, cost, pol.reshape(-1))
    assert g_pol == pytest.approx(g_ref, abs=1e-8)


@pytest.mark.parametrize("L,Q,XM,dmax", [(2, 3, 8, 8), (3, 2, 7, 8)])
def test_o2_exact_evaluation_against_linear_system(L, Q, XM, dmax):
    pd = poisson_trunc(3.0, dmax)
    states, P, cost = dense_chain(L, Q, XM, pd)
    m = A.LostSales(L=L, Q=Q, XM=XM, pd=pd)
    g_opt, pol_opt, _, _ = m.rvi(tol=1e-12)
    tables = {"Base-Stock": m.policy_base_stock(XM - 3), "Capped": m.policy_capped(XM - 3, 1), "konstant 1": m.policy_constant(1), "DP": pol_opt}
    for name, tab in tables.items():
        g_dense, pi_dense = gain_of_policy(P, cost, tab.reshape(-1))
        g_tensor, pi_tensor = m.evaluate(tab, tol=1e-13)
        assert g_tensor == pytest.approx(g_dense, abs=1e-8), name
        assert np.abs(pi_tensor.reshape(-1) - pi_dense).max() < 1e-7, name


def test_o2_branch_rules_differ_from_each_other_and_from_the_optimum():
    """Zweigprobe: Base-Stock, Capped und DP geben auf der Instanz L = 2 verschiedene Kosten (sonst wäre ein Zweig tot)."""
    m = A.LostSales(L=2)
    g_dp = m.rvi()[0]
    g_bs = m.evaluate(m.policy_base_stock(12))[0]
    g_cap = m.evaluate(m.policy_capped(12, 4))[0]
    assert g_dp < g_cap < g_bs


# ---- O3: Simulation --------------------------------------------------------------------------------------------------------------------------

def test_o3_simulation_matches_exact_evaluation_within_four_standard_errors():
    stream = S.demand_stream(2024, 800, 1000)
    for L in (2, 4):
        m = A.LostSales(L=L)
        _, pol_dp, _, _ = m.rvi()
        cases = {"DP": pol_dp, "Base-Stock": m.policy_base_stock(3 * L + 6), "Capped": m.policy_capped(3 * L + 6, 4)}
        for name, tab in cases.items():
            exact, _ = m.evaluate(tab)
            mean, se = S.mean_and_se(S.simulate(L, S.table_policy(m, tab), stream, burn=200))
            assert abs(mean - exact) < 4 * se, (L, name, mean, exact, se)


def test_o3_common_stream_gives_a_smaller_error_for_a_difference():
    stream = S.demand_stream(3, 800, 1000)
    a = S.simulate(2, S.base_stock(12), stream, burn=200)
    b = S.simulate(2, S.capped(12, 4), stream, burn=200)
    se_paired = S.mean_and_se(a, b)[1]
    se_separate = float(np.hypot(S.mean_and_se(a)[1], S.mean_and_se(b)[1]))
    assert se_paired < 0.7 * se_separate


# ---- O4: Gradient ----------------------------------------------------------------------------------------------------------------------------

PD = [0.25, 0.5, 0.25]
QT, XMT, TT = 1, 2, 3


def _paths():
    """Alle (Aktionen, Zustandsfolge, Kosten je Periode, Wahrscheinlichkeit der Nachfragen): 2^3 · 3^3 = 216 Pfade der Mini-Instanz (L = 2)."""
    for acts in itertools.product(range(QT + 1), repeat=TT):
        for ds in itertools.product(range(3), repeat=TT):
            x, p1 = 0, 0
            states, costs = [], []
            for t in range(TT):
                states.append((x, p1))
                costs.append(max(x - ds[t], 0) + 9 * max(ds[t] - x, 0))
                x, p1 = min(max(x - ds[t], 0) + p1, XMT), acts[t]
            yield acts, states, costs, float(np.prod([PD[d] for d in ds]))


def _feat(x, p1):
    return np.array([[x / XMT, p1 / QT, (x + p1) / (XMT + QT)]])


def _expected_cost(net):
    cache, total = {}, 0.0
    for acts, states, costs, w in _paths():
        lp = 1.0
        for t in range(TT):
            if states[t] not in cache:
                cache[states[t]] = LN.softmax(net.apply(_feat(*states[t])))[0]
            lp *= cache[states[t]][acts[t]]
        total += w * lp * sum(costs)
    return total


def _reinforce(net, baseline=None):
    grads = [np.zeros_like(a) for a in net.params()]
    for acts, states, costs, w in _paths():
        pr = [LN.softmax(net.apply(_feat(*s)))[0] for s in states]
        weight = w * float(np.prod([pr[t][acts[t]] for t in range(TT)]))
        for t in range(TT):
            advantage = sum(costs[t:]) - (baseline(t, states[t], acts[t]) if baseline else 0.0)
            net.forward(_feat(*states[t]))
            dout = LN.score_logit_gradient(pr[t][None, :], np.array([acts[t]]), np.array([weight * advantage]))
            for G, g in zip(grads, net.gradients(dout)):
                G += g
    return grads


def _finite_differences(net):
    out = []
    for prm in net.params():
        grad = np.zeros_like(prm)
        for idx in np.ndindex(prm.shape):
            old = prm[idx]
            prm[idx] = old + 1e-6
            up = _expected_cost(net)
            prm[idx] = old - 1e-6
            down = _expected_cost(net)
            prm[idx] = old
            grad[idx] = (up - down) / 2e-6
        out.append(grad)
    return out


def test_o4_backward_step_matches_finite_differences_over_all_216_paths():
    rng = np.random.default_rng(11)
    net = LN.MLP(3, 4, QT + 1, rng, 0.5)
    net.b1 = rng.normal(0, 0.3, 4)
    net.b2 = rng.normal(0, 0.3, QT + 1)
    assert len(list(_paths())) == 216
    fd = _finite_differences(net)
    scale = max(np.abs(g).max() for g in fd)
    for variant, baseline in (("ohne Basiswert", None), ("Basiswert vom Zustand", lambda t, s, a: 3.0 * s[0] + 2.0 * s[1] + t)):
        got = _reinforce(net, baseline)
        err = max(np.abs(a - b).max() for a, b in zip(fd, got))
        assert err < 1e-5 * max(1.0, scale), (variant, err)
    # Zweigprobe: ein Basiswert, der von der Aktion abhängt, verfälscht den Gradienten (das Orakel ist nicht blind)
    bad = _reinforce(net, lambda t, s, a: 5.0 * a)
    assert max(np.abs(a - b).max() for a, b in zip(fd, bad)) > 1e-3


def test_o4_entropy_term_matches_finite_differences():
    z0 = np.array([[0.3, -0.7, 1.1, 0.0]])
    code = LN.entropy_logit_gradient(LN.softmax(z0))

    def neg_entropy(z):
        pr = LN.softmax(z)
        return float((pr * np.log(pr)).sum())

    fd = np.array([[(neg_entropy(z0 + 1e-6 * np.eye(4)[i]) - neg_entropy(z0 - 1e-6 * np.eye(4)[i])) / 2e-6 for i in range(4)]])
    assert np.abs(code - fd).max() < 1e-7


# ---- O5: Zählung -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("L", [1, 2, 3, 4, 5])
def test_o5_count_formula_against_enumeration_and_index_is_bijective(L):
    Q, XM = 2, 3
    tuples = list(itertools.product(range(XM + 1), *[range(Q + 1)] * (L - 1)))
    m = A.LostSales(L=L, Q=Q, XM=XM, dmax=6)
    assert len(tuples) == m.S == N.lost_sales_states(L, Q=Q, XM=XM) == (XM + 1) * (Q + 1) ** (L - 1)
    indices = {N.state_index(t[0], t[1:], Q) for t in tuples}
    assert indices == set(range(m.S))                                # bijektiv auf 0..S-1


def test_o5_perishable_and_two_stage_counts_against_enumeration():
    cap = 2
    for m in (2, 3, 4):
        assert N.perishable_states(m, 0, cap=cap) == len(list(itertools.product(range(cap + 1), repeat=m - 1)))
        assert N.perishable_states(m, 1, cap=cap) == len(list(itertools.product(range(cap + 1), repeat=m)))
    for lr, lw in ((1, 1), (1, 2), (2, 1), (2, 2)):
        Q = 1
        tuples = itertools.product(range(N.DEALER_VALUES), range(N.WAREHOUSE_VALUES), *[range(Q + 1)] * (lr + lw - 2))
        assert N.two_stage_states(lr, lw, Q=Q) == sum(1 for _ in tuples)


# ---- O6: Handbeispiel in Brüchen -------------------------------------------------------------------------------------------------------------

def test_o6_hand_example_in_fractions_matches_the_app():
    pd = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]
    ec = [sum(p * (max(x - d, 0) + 9 * max(d - x, 0)) for d, p in enumerate(pd)) for x in range(3)]
    states = [(x, p1) for x in range(3) for p1 in range(2)]
    V = {0: {s: Fraction(0) for s in states}}
    for t in (1, 2, 3):
        V[t] = {}
        for (x, p1) in states:
            V[t][(x, p1)] = min(ec[x] + sum(p * V[t - 1][(min(max(x - d, 0) + p1, 2), a)] for d, p in enumerate(pd)) for a in (0, 1))
    assert ec == [9, Fraction(5, 2), 1]
    assert V[2][(0, 0)] == 18 and V[3][(0, 0)] == Fraction(41, 2)
    h = E.hand_example()
    for t in (1, 2, 3):
        for i, s in enumerate(states):
            assert float(V[t][s]) == pytest.approx(float(h["values"][t - 1].reshape(-1)[i]))
    assert h["best_g"] == pytest.approx(1.75) == pytest.approx(h["g"]) == pytest.approx(h["always_one"])
    # Ableitung der erwarteten Kosten nach θ₀: ½ · (27 − 23,75) = 13/8 (der Belohnung −13/8), mit Brüchen gerechnet (dieselben Funktionen akzeptieren Fraction)
    half = [Fraction(1, 2), Fraction(1, 2)]
    costs = [ec[0] + V[2][(0, 0)], ec[0] + V[2][(0, 1)]]
    assert costs == [27, Fraction(41, 2)]
    assert LN.score_gradient_hand(half, costs)[0] == Fraction(13, 8)
    assert LN.reinforce_expectation(half, costs)[0] == Fraction(13, 8)

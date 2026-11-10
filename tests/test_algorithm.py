"""Einheiten des Kerns, jede an der Mini-Instanz (tests/conftest.py) von Hand gerechnet (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import dpg_algorithm as A
import dpg_constants as C
import dpg_counts as N
import dpg_sim as S
from conftest import MINI_PD, mini_model


def test_poisson_pmf_sums_to_one_and_puts_the_rest_on_the_last_value():
    p = A.poisson_pmf(3.0, 14)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert p[0] == pytest.approx(np.exp(-3.0))
    assert p[1] == pytest.approx(3 * np.exp(-3.0))
    untruncated_14 = np.exp(-3.0) * 3.0**14 / 87178291200          # 14! = 87.178.291.200
    assert p[14] > untruncated_14                                  # der abgeschnittene Rest liegt zusätzlich auf 14


def test_expected_cost_of_the_mini_instance_by_hand():
    assert [A.expected_cost(x, MINI_PD, 1.0, 9.0) for x in range(3)] == pytest.approx([9.0, 2.5, 1.0])


def test_state_count_of_the_mini_instance_for_lead_times_1_to_5():
    assert [mini_model(L).S for L in range(1, 6)] == [3, 6, 12, 24, 48]


def test_finite_horizon_by_hand(mini):
    """V1 = ec; V2(0,0) = 9 + 9 = 18, V2(0,1) = 9 + 2,5; V2(1,0) = 2,5 + (¼·2,5 + ¾·9) = 9,875; V3(0,0): Aktion 0 gibt 27, Aktion 1 gibt 20,5."""
    values, actions = mini.finite_horizon(3)
    assert values[0][:, 0] == pytest.approx([9.0, 2.5, 1.0])
    assert values[1] == pytest.approx(np.array([[18.0, 11.5], [9.875, 4.625], [4.75, 2.375]]))
    assert values[2][0, 0] == pytest.approx(20.5)
    assert (actions[2] == 1).all()                                  # bei drei Perioden ist Bestellen in allen sechs Zuständen optimal
    assert (actions[0] == 0).all() and (actions[1] == 0).all()      # bei ein oder zwei Perioden Restlaufzeit lohnt die Bestellung nicht mehr


def test_rvi_of_the_mini_instance_gives_1_75_and_always_order_one(mini):
    g, pol, _, _ = mini.rvi(tol=1e-12)
    assert g == pytest.approx(1.75, abs=1e-9)                       # ½ · 2,5 + ½ · 1
    assert (pol == 1).all()


def test_evaluate_always_one_is_1_75_and_never_ordering_costs_ec_zero(mini):
    g, pi = mini.evaluate(mini.policy_constant(1), tol=1e-12)
    assert g == pytest.approx(1.75, abs=1e-9)
    assert pi[0].sum() == pytest.approx(0.0, abs=1e-12)             # Bestand 0 kommt unter „immer 1 bestellen“ nicht vor
    g0, _ = mini.evaluate(mini.policy_constant(0), tol=1e-12)       # ohne Bestellung stirbt der Bestand aus: dauerhaft Bestand 0, 9 je Periode
    assert g0 == pytest.approx(9.0, abs=1e-9)


def test_policy_tables_by_hand(mini):
    base = mini.policy_base_stock(2)       # Bestandsposition x + p1; Ziel 2, höchstens Q = 1
    assert base.tolist() == [[1, 1], [1, 0], [0, 0]]
    capped = mini.policy_capped(5, 0)      # Deckel 0: nie bestellen
    assert (capped == 0).all()
    assert mini.policy_capped(5, 1).tolist() == base_stock_reference(mini, 5)


def base_stock_reference(model, target):
    return [[int(min(max(target - (x + p1), 0), model.Q)) for p1 in range(model.A)] for x in range(model.X)]


def test_lead_time_one_has_no_pipeline_and_two_has_one_place():
    one, two = mini_model(1), mini_model(2)
    assert (one.S, one.Pn) == (3, 1) and (two.S, two.Pn) == (6, 2)
    g1, _, _, _ = one.rvi(tol=1e-12)
    g2, _, _, _ = two.rvi(tol=1e-12)
    assert g1 < g2                                                  # mehr Lieferzeit kostet mehr


def test_lead_time_below_one_is_rejected():
    with pytest.raises(ValueError):
        A.LostSales(L=0)


def test_vehicle_state_counts_are_the_formula():
    assert [A.LostSales(L=L).S for L in (1, 2, 3)] == [21, 189, 1701]
    assert N.state_count("lieferzeit", L=3) == 1701


def test_counts_for_all_three_directions():
    assert [N.state_count("lieferzeit", L=L) for L in range(1, 10)] == [21, 189, 1701, 15309, 137781, 1240029, 11160261, 100442349, 903981141]
    assert [N.state_count("haltbarkeit", m=m, lead=0) for m in range(2, 7)] == [9, 81, 729, 6561, 59049]
    assert [N.state_count("haltbarkeit", m=m, lead=1) for m in range(2, 5)] == [81, 729, 6561]
    assert [N.state_count("zwei_stufen", lr=a, lw=b) for a, b in ((1, 1), (1, 2), (2, 2))] == [1064, 9576, 86184]
    assert N.state_count("lieferzeit", L=5, Q=5) == 21 * 6**4 == 27216
    with pytest.raises(ValueError):
        N.state_count("unbekannt")


def test_formats():
    assert N.format_count(100442349) == "100.442.349"
    assert N.format_bytes(N.table_bytes(137781)) == "1,1 MB"
    assert N.format_bytes(N.table_bytes(100442349)) == "0,8 GB"
    assert N.format_bytes(168) == "168 Byte"
    assert N.state_pairs(21) == 189


def test_state_index_is_x_major_and_pipeline_digits_follow():
    assert N.state_index(0, (), 8) == 0
    assert N.state_index(2, (3, 4), 8) == 2 * 81 + 3 * 9 + 4


def test_max_order_and_reach_on_the_mini_instance(mini):
    _, pol, _, _ = mini.rvi(tol=1e-12)
    pi, _ = mini.stationary(pol, tol=1e-13)
    assert A.max_order(pi, pol) == 1
    assert A.reach_count(pi) == 2                                   # unter „immer 1“ bleibt die Kette in (1, 1) und (2, 1) (je Anteil ½)
    assert mini.edge_mass(pi) == pytest.approx(pi[2].sum())


# ---- Simulation -----------------------------------------------------------------------------------------------------------------------------

def test_simulation_by_hand_constant_demand_one_and_order_one():
    """Nachfrage immer 1, Bestellung immer 1, L = 1: Periode 0 kostet 9 (Bestand 0, ein Stück fehlt), danach Bestand 1 und Kosten 0."""
    stream = np.ones((11, 3), dtype=np.int16)
    policy = lambda x, pipe: np.ones(len(x), dtype=np.int64)  # noqa: E731
    assert S.simulate(1, policy, stream, burn=0) == pytest.approx([9.0 / 11] * 3)
    assert S.simulate(1, policy, stream, burn=1) == pytest.approx([0.0] * 3)


def test_demand_stream_shape_range_and_mean():
    st = S.demand_stream(5, 400, 50)
    assert st.shape == (400, 50) and st.min() >= 0 and st.max() <= C.DMAX
    assert st.mean() == pytest.approx(3.0, abs=0.1)
    assert (S.demand_stream(5, 400, 50) == st).all()                # gleicher Zufalls-Seed, gleicher Strom


def test_rule_functions_and_table_policy_agree():
    m = A.LostSales(L=3)
    rng = np.random.default_rng(1)
    x = rng.integers(0, m.X, 50)
    pipe = rng.integers(0, m.A, (50, 2))
    tab = S.table_policy(m, m.policy_capped(15, 4))
    assert (tab(x, pipe) == S.capped(15, 4)(x, pipe)).all()
    assert (S.table_policy(m, m.policy_base_stock(15))(x, pipe) == S.base_stock(15)(x, pipe)).all()


def test_paired_difference_by_hand():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
    mean, se = S.mean_and_se(a, b)                                  # Differenzen 0, 1, 2: Mittel 1, Standardabweichung 1
    assert mean == pytest.approx(1.0) and se == pytest.approx(1 / np.sqrt(3))

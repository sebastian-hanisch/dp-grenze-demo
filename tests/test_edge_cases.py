"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Lieferzeit, Bestellgrenze 0, keine Nachfrage, Gleichstände, Schwellen."""

import numpy as np
import pytest

import dpg_algorithm as A
import dpg_constants as C
import dpg_evaluation as E
import dpg_sim as S


def test_lead_time_one_runs_through_dp_evaluation_and_simulation():
    """L = 1 hat keine Pipeline (Pn = 1); Zweige `L >= 2` dürfen nicht greifen."""
    m = A.LostSales(L=1)
    g, pol, _, _ = m.rvi()
    assert pol.shape == (21, 1) and m.evaluate(pol)[0] == pytest.approx(g, abs=1e-9)
    mean, _ = S.mean_and_se(S.simulate(1, S.table_policy(m, pol), S.demand_stream(1, 300, 200), burn=50))
    assert mean == pytest.approx(g, abs=0.15)


def test_exactly_two_places_of_pipeline_and_lead_time_two():
    m = A.LostSales(L=3, Q=1, XM=2, pd=[0.25, 0.5, 0.25])
    assert m.Pn == 4 and m.M == 2 and m.S == 12
    assert m.rvi(tol=1e-12)[0] > 0


def test_order_limit_zero_means_no_ordering_and_the_cost_of_an_empty_shelf():
    m = A.LostSales(L=1, Q=0, XM=3, pd=[0.5, 0.5])
    g, pol, _, _ = m.rvi(tol=1e-12)
    assert (pol == 0).all() and g == pytest.approx(9.0 * 0.5)       # dauerhaft Bestand 0: 9 · E[D] = 4,5


def test_no_demand_costs_nothing_with_no_orders():
    m = A.LostSales(L=2, Q=2, XM=3, pd=[1.0])
    g, pol, _, _ = m.rvi(tol=1e-12)
    assert g == pytest.approx(0.0, abs=1e-12) and (pol[0] == 0).all()


def test_ties_choose_the_smaller_order():
    """Alle Kosten null (h = p = 0): jede Bestellmenge ist gleich gut, die dokumentierte Regel nimmt die kleinste."""
    m = A.LostSales(L=1, Q=2, XM=3, pd=[0.5, 0.5], hold=0.0, penalty=0.0)
    g, pol, _, _ = m.rvi(tol=1e-12)
    assert g == 0.0 and (pol == 0).all()


def test_rule_targets_at_the_edges():
    m = A.LostSales(L=2, Q=8, XM=20)
    assert (m.policy_base_stock(0) == 0).all()                      # Ziel 0: nie bestellen
    assert m.policy_base_stock(1000).min() == 8                     # sehr hohes Ziel: immer die Bestellgrenze
    assert (m.policy_capped(1000, 0) == 0).all() and (m.policy_capped(1000, 100) == 8).all()
    assert (m.policy_constant(0) == 0).all()


def test_boundaries_are_inclusive_in_the_state_space():
    m = A.LostSales(L=2, Q=8, XM=20)
    assert m.X == 21 and m.A == 9 and m.ip.max() == 20 + 8          # Bestand 0..20 und Bestellung 0..8 gehören zum Zustandsraum


def test_simulation_with_one_counted_period_and_two_chains():
    stream = np.ones((3, 2), dtype=np.int16)
    v = S.simulate(1, lambda x, pipe: np.zeros(len(x), dtype=np.int64), stream, burn=2)
    assert v.tolist() == [9.0, 9.0]                                  # eine gezählte Periode: Bestand 0, Nachfrage 1, ein Stück fehlt


def test_hand_state_counts_start_at_one_period():
    rows = E.hand_state_counts()
    assert rows[0] == dict(L=1, states=3, terms=18) and rows[-1] == dict(L=5, states=48, terms=288)


def test_view_constants_and_presets_cover_both_views():
    assert len(C.VIEWS) == 2 and set(p["view"] for p in C.PRESETS.values()) <= {0, 1}

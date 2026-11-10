"""Auswertung der Lernläufe (dpg_stats) an einer kleinen, von Hand gerechneten Fassung der Daten.

Bezug DP = 100, Capped Base-Stock = 101 (Lücke 1,0 %), Base-Stock = 105. Zwei Zufalls-Seeds, Messpunkte bei den Updates 0, 50, 100, 150, 200, 250, 500, 1000, 3000.
Lücken in % (Seed 0): 200, 100, 10, 4, 3, 2, 1,5, 0,5, 0,2; (Seed 1): 200, 100, 10, 4, 3, 2, 1,5, 1,2, 2,5.
"""

import numpy as np
import pytest

import dpg_constants as C
import dpg_stats as ST

UPDATES = [0, 50, 100, 150, 200, 250, 500, 1000, 3000]
G0 = [300, 200, 110, 104, 103, 102, 101.5, 100.5, 100.2]
G1 = [300, 200, 110, 104, 103, 102, 101.5, 101.2, 102.5]


def _runs(g0, g1, se=0.0):
    mk = lambda g, sek: dict(updates=UPDATES, g=g, se=[se] * len(UPDATES), mode=["exakt"] * len(UPDATES), sek=sek)  # noqa: E731
    return {"0": mk(g0, 100), "1": mk(g1, 200)}


CURVES = {"runs": {"2": _runs(G0, G1)}}
REFS = {"L2": dict(g_dp=100.0, g_cap=101.0, g_bs=105.0, art="exakt")}


def test_gap_percent_and_gap_vs_reference_by_hand():
    assert ST.gap_percent(105.0, 100.0) == pytest.approx(5.0)
    assert ST.gap_vs_reference(105.0, 100.0) == (pytest.approx(5.0), None)                 # exakt bewertet: kein z
    pct, z = ST.gap_vs_reference(105.0, 100.0, se=1.0, se_ref=1.0)
    assert pct == pytest.approx(5.0) and z == pytest.approx(5.0 / np.sqrt(2.0))


def test_reference_cost_prefers_the_dp_and_falls_back_to_capped():
    assert ST.reference_cost(REFS["L2"]) == 100.0
    assert ST.reference_cost(dict(g_dp=None, g_cap=7.0)) == 7.0


def test_run_gaps_pick_and_mean_of_the_last_five():
    updates, seeds, mat = ST.run_gaps(2, curves=CURVES, refs=REFS)
    assert updates == UPDATES and seeds == ["0", "1"]
    assert mat[0] == pytest.approx([200, 100, 10, 4, 3, 2, 1.5, 0.5, 0.2])
    _, _, mat5 = ST.run_gaps(2, view=C.VIEW_MEAN5, curves=CURVES, refs=REFS)
    assert np.isnan(mat5[:, :4]).all()                                                     # die ersten vier Messpunkte haben noch kein Mittel
    assert mat5[0, 4] == pytest.approx((200 + 100 + 10 + 4 + 3) / 5)                       # Messpunkte 0 bis 4
    assert mat5[0, 8] == pytest.approx((3 + 2 + 1.5 + 0.5 + 0.2) / 5)                      # 1,44
    assert mat5[1, 8] == pytest.approx((3 + 2 + 1.5 + 1.2 + 2.5) / 5)                      # 2,04


def test_curve_stats_by_hand():
    s = ST.curve_stats(2, curves=CURVES, refs=REFS)
    assert s["capped_gap"] == pytest.approx(1.0) and s["base_stock_gap"] == pytest.approx(5.0)
    assert s["final"] == pytest.approx([0.2, 2.5])
    assert s["final_median"] == pytest.approx(1.35) and s["final_min"] == pytest.approx(0.2) and s["final_max"] == pytest.approx(2.5)
    assert s["final_sd"] == pytest.approx(np.std([0.2, 2.5], ddof=1))
    assert (s["below_capped"], s["below_2pct"]) == (1, 1)                                  # Seed 0 liegt unter 1,0 % und unter 2 %, Seed 1 mit 2,5 % nicht
    assert s["below_capped_mean5"] == 0                                                    # Mittel 1,44 und 2,04: beide über 1,0 %
    assert s["late_points"] == 2                                                           # Updates 1000 und 3000
    assert s["late_frac_below_2pct"] == pytest.approx(3 / 4) and s["late_frac_below_capped"] == pytest.approx(2 / 4)
    assert s["late_sd_median"] == pytest.approx(np.median([np.std([0.5, 0.2], ddof=1), np.std([1.2, 2.5], ddof=1)]))
    assert s["first_under_2pct_median"] == 500 * C.PERIODS_PER_STEP and s["first_under_2pct_never"] == 0
    assert s["sek_median"] == 150 and (s["sek_min"], s["sek_max"]) == (100, 200)


def test_curve_stats_without_dp_compares_against_capped_in_standard_errors():
    """L = 8: kein Optimum, Bezug ist Capped (100). Läufe: 99 (z = −1,41), 101,5 (z = +2,12), 100,0 (z = 0) mit Standardfehler 0,5 bei Netz und Regel."""
    refs = {"L2": dict(g_dp=None, g_cap=100.0, se_cap=0.5, g_bs=105.0, art="sim")}
    g = lambda last: G0[:-1] + [last]  # noqa: E731
    runs = {s: dict(updates=UPDATES, g=g(v), se=[0.5] * 9, mode=["sim"] * 9, sek=10) for s, v in (("0", 99.0), ("1", 101.5), ("2", 100.0))}
    s = ST.curve_stats(2, curves={"runs": {"2": runs}}, refs=refs)
    assert s["z"] == pytest.approx([-1 / np.hypot(0.5, 0.5), 1.5 / np.hypot(0.5, 0.5), 0.0])
    assert (s["clearly_better"], s["clearly_worse"], s["better_point"]) == (0, 1, 1)
    assert s["capped_gap"] == pytest.approx(0.0)


def test_min_difference_to_dp_only_counts_exactly_evaluated_points():
    runs = _runs(G0, G1)
    runs["0"]["mode"][2] = "sim"                                                           # Messpunkt per Simulation: zählt nicht
    runs["0"]["g"][2] = 90.0                                                               # läge unter dem Optimum, wäre aber ein Simulationswert
    assert ST.min_difference_to_dp({"runs": {"2": runs}}, REFS) == pytest.approx(0.2)      # kleinste exakte Differenz: Seed 0, letzter Punkt
    assert ST.min_difference_to_dp({"runs": {"2": _runs(G0, G1)}}, {"L2": dict(REFS["L2"], art="sim")}) == np.inf


def test_data_files_load_and_cover_all_runs():
    curves, refs = ST.load_curves(), ST.load_refs()
    assert sorted(curves["runs"], key=int) == [str(L) for L in C.LEARN_L]
    assert [len(curves["runs"][str(L)]) for L in C.LEARN_L] == [10, 10, 10, 5, 5, 5, 5]
    assert all(f"L{L}" in refs for L in range(C.L_MIN, C.L_MAX + 1))

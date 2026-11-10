"""Jede Zahl der README steht hier (Tabelle „Zahl → Test“ in der README; Band oder exakt, wie im Test).

Quellen der Zahlen: LIVE = in diesem Test gerechnet (Zustandszahlen, DP bis L = 5, exakte Auswertung der Regeln); DATEN = eingefrorene Messungen in
dpg_data/refs.json (Bezugswerte, Zeiten, DP bei L = 6 und 7, Regeln per Simulation bei L = 7 und 8) und dpg_data/curves.json (50 Lernläufe). Gelernte
Zahlen werden nie neu gerechnet und nie exakt geprüft, nur als eingefrorene Daten gelesen. DP-Zeiten stehen nur als Ordnung, nicht als Sekundenzahl.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import dpg_algorithm as A
import dpg_constants as C
import dpg_counts as N
import dpg_evaluation as E
import dpg_stats as ST

ROOT = Path(__file__).resolve().parent.parent
REFS = ST.load_refs()
CURVES = ST.load_curves()
STATS = {L: ST.curve_stats(L) for L in C.LEARN_L}


def gap(L, rule):
    ref = REFS[f"L{L}"]
    return float(ST.gap_percent(ref[rule], ST.reference_cost(ref)))


# ---- Zustandszahlen ---------------------------------------------------------------------------------------------------------------------------

def test_k01_state_counts_by_formula():
    """K01: 21 / 189 / 1.701 / 15.309 / 137.781 / 1.240.029 / 11.160.261 für L = 1 bis 7, 100.442.349 bei L = 8, 903.981.141 bei L = 9; Faktor 9 je Periode."""
    values = [N.lost_sales_states(L) for L in range(1, 10)]
    assert values == [21, 189, 1701, 15309, 137781, 1240029, 11160261, 100442349, 903981141]
    assert all(b == 9 * a for a, b in zip(values, values[1:]))
    assert [N.format_bytes(N.table_bytes(v)) for v in values[4:9]] == ["1,1 MB", "9,9 MB", "89,3 MB", "0,8 GB", "7,2 GB"]


def test_k02_the_model_has_as_many_states_as_the_formula():
    for L in range(1, 6):
        assert A.LostSales(L=L).S == N.lost_sales_states(L)


def test_k03_perishable_and_two_stage_counts():
    """K03: Haltbarkeit 9 bis 59.049 (m = 2 bis 6, Lieferzeit 0), m = 7: 531.441, m = 8: 4.782.969; zwei Stufen 1.064 / 9.576 / 86.184 / 775.656 / 6.980.904."""
    assert [N.perishable_states(m, 0) for m in range(2, 9)] == [9, 81, 729, 6561, 59049, 531441, 4782969]
    assert [N.perishable_states(m, 1) for m in range(2, 6)] == [81, 729, 6561, 59049]
    assert [N.two_stage_states(a, b) for a, b in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))] == [1064, 9576, 86184, 775656, 6980904]


# ---- DP: Kosten, Rand, Bestellgrenze, erreichbare Zustände -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dps():
    out = {}
    for L in range(1, 6):
        m = A.LostSales(L=L)
        g, pol, _, _ = m.rvi()
        pi, _ = m.stationary(pol)
        out[L] = dict(model=m, g=g, pol=pol, pi=pi)
    return out


def test_k04_dp_costs_per_period(dps):
    """K04: durchschnittliche Kosten je Periode der DP: 4,2521 / 4,7493 / 5,0784 / 5,3050 / 5,4682 (L = 1 bis 5, live), 5,5940 und 5,6978 (L = 6, 7, Daten)."""
    assert [round(dps[L]["g"], 4) for L in range(1, 6)] == [4.2521, 4.7493, 5.0784, 5.3050, 5.4682]
    assert [round(REFS[f"L{L}"]["g_dp"], 4) for L in (6, 7)] == [5.5940, 5.6978]
    for L in range(1, 6):
        assert dps[L]["g"] == pytest.approx(REFS[f"L{L}"]["g_dp"], abs=1e-9)             # die Daten stammen aus demselben Code
    assert all(REFS[f"L{L}"]["g_dp"] < REFS[f"L{L + 1}"]["g_dp"] for L in range(1, 7))      # mehr Lieferzeit kostet mehr


@pytest.mark.slow
def test_k04b_dp_at_lead_time_six_is_reproduced_live():
    g = A.LostSales(L=6).rvi()[0]
    assert g == pytest.approx(REFS["L6"]["g_dp"], abs=1e-8)


def test_k05_edge_mass_is_below_1e4_with_limit_20_and_not_with_16(dps):
    """K05: Randmasse bei Bestandsgrenze 20: unter 1e-20 (L = 1 bis 4), 2,1e-6 (L = 5), 2,6e-5 (L = 6), 9,1e-5 (L = 7); bei Grenze 16 und L = 4: 2,6e-4."""
    assert all(dps[L]["model"].edge_mass(dps[L]["pi"]) < 1e-20 for L in range(1, 5))
    assert round(dps[5]["model"].edge_mass(dps[5]["pi"]) / 1e-6, 1) == 2.1
    assert round(REFS["L6"]["edge"] / 1e-5, 1) == 2.6 and round(REFS["L7"]["edge"] / 1e-5, 1) == 9.1
    assert all(dps[L]["model"].edge_mass(dps[L]["pi"]) < C.EDGE_MASS_LIMIT for L in range(1, 6)) and REFS["L7"]["edge"] < C.EDGE_MASS_LIMIT
    m16 = A.LostSales(L=4, XM=16)
    pi16, _ = m16.stationary(m16.rvi()[1])
    assert round(m16.edge_mass(pi16) / 1e-4, 1) == 2.6                                   # über der Marke 1e-4: mit Grenze 16 reicht der Rand nicht


def test_k06_order_limit_5_gives_the_same_cost_with_fewer_states(dps):
    """K06: bei L = 1 bis 5 bestellt die DP höchstens 5 Stück; Q = 5 gibt dieselben Kosten (Abweichung unter 1e-9) bei 21 · 6^(L−1) Zuständen; Q = 4 nicht."""
    for L in range(1, 6):
        assert A.max_order(dps[L]["pi"], dps[L]["pol"]) == 5, L
        m5 = A.LostSales(L=L, Q=5)
        assert m5.S == 21 * 6 ** (L - 1)
        assert m5.rvi()[0] == pytest.approx(dps[L]["g"], abs=1e-9), L
        assert REFS[f"L{L}"]["q5"]["states_q5"] == m5.S
    assert A.LostSales(L=1, Q=4).rvi()[0] > dps[1]["g"] + 1e-3                             # Zweigprobe: mit Q = 4 greift die Grenze
    assert N.lost_sales_states(5, Q=5) == 27216


def test_k07_reachable_states(dps):
    """K07: unter der DP-Politik kommen 10 / 60 / 313 / 1.188 / 4.785 Zustände vor (L = 1 bis 5, live), 15.426 und 41.428 (L = 6, 7); bei L = 7 sind das 0,4 % der Tabelle; Wachstum je Periode Faktor 2,7 bis 6,0 statt 9."""
    assert [A.reach_count(dps[L]["pi"]) for L in range(1, 6)] == [10, 60, 313, 1188, 4785]
    assert (REFS["L6"]["reach"], REFS["L7"]["reach"]) == (15426, 41428)
    assert round(100 * REFS["L7"]["reach"] / REFS["L7"]["states"], 1) == 0.4
    ratios = [REFS[f"L{L + 1}"]["reach"] / REFS[f"L{L}"]["reach"] for L in range(1, 7)]
    assert (round(min(ratios), 1), round(max(ratios), 1)) == (2.7, 6.0)                     # Faktor 6,0 / 5,2 / 3,8 / 4,0 / 3,2 / 2,7 je Periode statt 9
    assert all(r < 9 for r in ratios)


def test_k08_dp_time_grows_and_lead_time_eight_is_not_in_the_data():
    """K08: DP-Zeit steigt mit L (Ordnung, keine Sekundenzahl); bei L = 8 sind es rund 0,8 GB je Tabelle, die DP ist in der App und in den Daten nicht gerechnet (die einmalige Messung des Prüfers außerhalb der App steht in K24)."""
    t = {L: REFS[f"L{L}"]["t_dp"] for L in range(1, 8)}
    assert t[1] < t[2] < t[3] < t[4] < t[5] < t[6] < t[7]
    assert 0.2 < t[5] < 3 and 2 < t[6] < 15 and 20 < t[7] < 120                              # Größenordnung: rund 1 s bei L = 5, einige Sekunden bei L = 6, rund eine Minute bei L = 7
    assert t[7] > 3 * t[6] and t[6] > 3 * t[5]                                                # jede Periode mehr vervielfacht die Zeit
    assert REFS["L8"]["g_dp"] is None and N.format_bytes(N.table_bytes(N.lost_sales_states(8))) == "0,8 GB"


# ---- Regeln und Maßstab -----------------------------------------------------------------------------------------------------------------------

def test_k09_rule_gaps_to_the_dp():
    """K09: Capped Base-Stock +0,10 / +0,69 / +1,63 / +0,82 / +0,36 / +0,37 / +0,39 % und Base-Stock +1,75 / +3,35 / +5,03 / +6,44 / +7,63 / +8,94 / +9,90 % (L = 1 bis 7)."""
    cap = [round(gap(L, "g_cap"), 2) for L in range(1, 8)]
    bs = [round(gap(L, "g_bs"), 2) for L in range(1, 8)]
    assert cap == [0.10, 0.69, 1.63, 0.82, 0.36, 0.37, 0.39]
    assert bs == [1.75, 3.35, 5.03, 6.44, 7.63, 8.94, 9.90]
    assert round(min(cap), 1) == 0.1 and round(max(cap), 1) == 1.6                           # „zwischen 0,1 und 1,6 %“
    assert all(REFS[f"L{L}"]["art"] == "exakt" for L in range(1, 7)) and REFS["L7"]["art"] == "sim"


def test_k10_rule_parameters_reproduce_the_stored_cost_live(dps):
    """K10: die Parameter der Regeln (Ziel S, Deckel) aus den Daten geben bei L = 1 bis 3 live dieselben Kosten (exakte Auswertung)."""
    for L in (1, 2, 3):
        ref, m = REFS[f"L{L}"], dps[L]["model"]
        assert m.evaluate(m.policy_base_stock(ref["S_bs"]))[0] == pytest.approx(ref["g_bs"], abs=1e-9)
        assert m.evaluate(m.policy_capped(ref["S_cap"], ref["cap"]))[0] == pytest.approx(ref["g_cap"], abs=1e-9)
    assert A.LostSales(L=2).best_capped(range(5, 36), range(1, 9))[1:] == (12, 4)            # das Gitter findet S = 12, Deckel 4 wieder
    assert {REFS[f"L{L}"]["cap"] for L in range(1, 9)} == {3, 4}                              # „Deckel 3 bis 4 Stück“ in der App


def test_k11_beyond_the_dp_base_stock_against_capped():
    """K11: bei L = 8 Base-Stock gegen Capped +9,95 %; gepaart +0,579 ± 0,003 je Periode, Standardfehler je Wert 0,0056 bis 0,0058; gepaarter Fehler kleiner als getrennt."""
    r = REFS["L8"]
    assert round(float(ST.gap_percent(r["g_bs"], r["g_cap"])), 2) == 9.95
    assert round(r["diff_bs_cap"], 3) == 0.579 and round(r["se_diff"], 3) == 0.003
    ses = [REFS[f"L{L}"][k] for L in (7, 8) for k in ("se_bs", "se_cap")]
    assert (round(min(ses), 4), round(max(ses), 4)) == (0.0056, 0.0058)                      # Standardfehler je Wert über beide Lieferzeiten und beide Regeln
    for L in (7, 8):
        ref = REFS[f"L{L}"]
        assert ref["se_diff"] < 0.5 * np.hypot(ref["se_bs"], ref["se_cap"])
    assert (r["S_cap"], r["cap"], r["S_bs"]) == (29, 3, 28)
    assert round(100 * REFS["L7"]["se_cap"] / REFS["L7"]["g_cap"], 1) == 0.1                # „Standardfehler etwa 0,1 % der Kosten“


# ---- Lernläufe --------------------------------------------------------------------------------------------------------------------------------

def test_k12_learning_runs_have_the_stated_shape():
    """K12: 50 Läufe: L = 2 bis 4 mit 10 Zufalls-Seeds, L = 5 bis 8 mit 5; 3.000 Updates à 25.600 Perioden (7,7e7 insgesamt)."""
    assert [STATS[L]["n_seeds"] for L in C.LEARN_L] == [10, 10, 10, 5, 5, 5, 5] and sum(STATS[L]["n_seeds"] for L in C.LEARN_L) == 50
    assert C.PERIODS_PER_STEP == 25600 and f"{C.UPDATES * C.PERIODS_PER_STEP:.1e}" == "7.7e+07"
    assert all(r["updates"][-1] == 3000 for runs in CURVES["runs"].values() for r in runs.values())


def test_k13_final_gaps_to_the_dp():
    """K13: Median der Endstände +0,53 / +0,67 / +1,00 / +0,77 / +0,62 / +0,86 % (L = 2 bis 7); 39 von 40 Läufen (L = 2 bis 6) unter 2 %; größter Endstand +4,71 % (L = 2)."""
    assert [round(STATS[L]["final_median"], 2) for L in range(2, 8)] == [0.53, 0.67, 1.00, 0.77, 0.62, 0.86]
    assert (sum(STATS[L]["below_2pct"] for L in range(2, 7)), sum(STATS[L]["n_seeds"] for L in range(2, 7))) == (39, 40)
    assert round(STATS[2]["final_max"], 2) == 4.71 and max(STATS[L]["final_max"] for L in range(3, 7)) < 2.0
    assert round(STATS[3]["final_min"], 2) == 0.31


def test_k14_learning_against_capped_depends_on_the_lead_time():
    """K14: am Endstand unter Capped in 6 / 8 / 4 von 10 Läufen (L = 2, 3, 4, zusammen 18 von 30), 0 von 5 (L = 5, 6), 1 von 5 (L = 7); Mittel der letzten fünf Messpunkte 4 / 10 / 2 von 10, 0 von 5."""
    assert [STATS[L]["below_capped"] for L in range(2, 8)] == [6, 8, 4, 0, 0, 1]
    assert sum(STATS[L]["below_capped"] for L in (2, 3, 4)) == 18
    assert [STATS[L]["below_capped_mean5"] for L in range(2, 8)] == [4, 10, 2, 0, 0, 0]
    assert round(STATS[3]["capped_gap"], 2) == 1.63 and round(STATS[5]["capped_gap"], 2) == 0.36


def test_k15_scatter_between_seeds_and_within_a_run():
    """K15: Standardabweichung des Endstands zwischen den Zufalls-Seeds 0,32 bis 0,61 Punkte (L = 3 bis 6), 1,36 (L = 2), 0,83 (L = 7); innerhalb eines Laufs ab Update 1.000 im Median 0,37 bis 0,81 (L = 2 bis 6), 1,09 (L = 7)."""
    sd = {L: round(STATS[L]["final_sd"], 2) for L in range(2, 8)}
    assert (sd[3], sd[4], sd[5], sd[6]) == (0.60, 0.35, 0.61, 0.32) and (sd[2], sd[7]) == (1.36, 0.83)
    inner = {L: round(STATS[L]["late_sd_median"], 2) for L in range(2, 8)}
    assert min(inner[L] for L in range(2, 7)) == 0.37 and max(inner[L] for L in range(2, 7)) == 0.81 and inner[7] == 1.09


def test_k16_share_of_late_points_below_2_percent_and_first_crossing():
    """K16: Anteil der Messpunkte ab Update 1.000 unter 2 % Lücke 96 / 93 / 94 / 89 / 92 / 56 % (L = 2 bis 7); erster Messpunkt unter 2 % im Median bei 1,3e7 bis 3,8e7 Perioden."""
    assert [round(100 * STATS[L]["late_frac_below_2pct"]) for L in range(2, 8)] == [96, 93, 94, 89, 92, 56]
    firsts = [STATS[L]["first_under_2pct_median"] for L in range(2, 8)]
    assert min(firsts) == 12.8e6 and max(firsts) == 38.4e6 and all(STATS[L]["first_under_2pct_never"] == 0 for L in range(2, 8))


def test_k17_learned_policy_is_never_below_the_optimum():
    """K17: kleinste Differenz (Kosten des Netzes minus DP) über alle exakt bewerteten Messpunkte aller Läufe: +8e-4, nie negativ."""
    worst = ST.min_difference_to_dp()
    assert worst > 0 and round(worst / 1e-4) == 8


def test_k18_lead_time_eight_counts_with_unpaired_standard_errors():
    """K18: bei L = 8 Endstände gegen Capped −0,17 / −0,08 / −0,05 / +0,32 / +1,48 %; in Standardfehlern (Fehler von Netz und Regel zusammengesetzt, ungepaart) −1,2 / −0,5 / −0,3 / +2,2 / +10,1; 3 von 5 mit geringerer Punktschätzung, 0 erkennbar besser, 2 erkennbar schlechter. Die gepaarten z-Werte (−4,2 / −1,8 / −1,2) stammen aus der Messung des Prüfers und stehen nicht in den Daten."""
    s = STATS[8]
    assert sorted(round(x, 2) for x in s["final"]) == [-0.17, -0.08, -0.05, 0.32, 1.48]
    assert sorted(round(z, 1) for z in s["z"]) == [-1.2, -0.5, -0.3, 2.2, 10.1]
    assert (s["better_point"], s["clearly_better"], s["clearly_worse"]) == (3, 0, 2)


def test_k19_run_times_of_the_learning_runs_and_the_dp():
    """K19: Laufzeit je Lauf im Median 172 bis 219 s je Lieferzeit (L = 2 bis 8), Einzelläufe 147 bis 483 s; bei L = 5 braucht die DP ein Bruchteil davon (mehr als das Fünfzigfache)."""
    assert [STATS[L]["sek_median"] for L in range(2, 9)] == [172, 177.5, 176, 209, 219, 189, 189]
    allsek = [r["sek"] for runs in CURVES["runs"].values() for r in runs.values()]
    assert (min(allsek), max(allsek)) == (147, 483)
    assert STATS[5]["sek_median"] > 50 * REFS["L5"]["t_dp"]


# ---- Handbeispiel und Hilfetexte --------------------------------------------------------------------------------------------------------------

def test_k20_hand_example():
    """K20: Handbeispiel: 6 Zustände; ec = 9 / 2,5 / 1; V3(0,0) = 20,5; g = 1,75 (Aufzählung aller 64 Politiken gleich); Ableitung 13/8 = 1,625; Mini-Zustandszahlen 3 / 6 / 12 / 24 / 48."""
    h = E.hand_example()
    assert len(h["states"]) == 6 and h["n_policies"] == 64
    assert h["ec"] == pytest.approx([9.0, 2.5, 1.0])
    assert h["values"][2][0, 0] == pytest.approx(20.5) and h["costs_action"] == pytest.approx((27.0, 20.5))
    assert h["g"] == pytest.approx(1.75, abs=1e-9) and h["best_g"] == pytest.approx(1.75, abs=1e-9)
    assert h["gradient_exact"][0] == pytest.approx(1.625) == pytest.approx(h["gradient_reinforce"][0])
    assert [r["states"] for r in E.hand_state_counts()] == [3, 6, 12, 24, 48]


def test_k21_preset_help_texts_state_measured_numbers():
    """K21: die Hilfetexte der Presets nennen 189 Zustände / 0,5 % (L = 2), 8 von 10 (L = 3), 137.781 / „Kein Lauf von 5“ (L = 5), rund 100 Millionen / 0,8 GB / Capped rund +0,9 % (L = 8, K24)."""
    h = C.PRESET_HELP
    assert "189 Zustände" in h["Kleiner Fall (L = 2)"] and N.lost_sales_states(2) == 189 and "0,5 %" in h["Kleiner Fall (L = 2)"] and round(STATS[2]["final_median"], 1) == 0.5
    assert "8 von 10" in h["Netz meist unter Capped (L = 3)"] and STATS[3]["below_capped"] == 8 and STATS[3]["n_seeds"] == 10
    assert "137.781" in h["Netz nicht unter Capped (L = 5)"] and "Kein Lauf von 5" in h["Netz nicht unter Capped (L = 5)"] and STATS[5]["below_capped"] == 0 and STATS[5]["n_seeds"] == 5
    assert "100 Millionen" in h["Ohne DP in der App (L = 8)"] and "0,8 GB" in h["Ohne DP in der App (L = 8)"] and 1e8 < N.lost_sales_states(8) < 1.01e8
    assert "+0,9 %" in h["Ohne DP in der App (L = 8)"]


def test_k24_dp_at_lead_time_eight_is_a_single_measurement_outside_the_app():
    """K24: die DP bei L = 8 ist in den Daten nicht enthalten; die README nennt die einmalige Messung des Prüfers (g = 5,782247, 522 s, 5,95 GB Spitzenspeicher, Rechner mit 32 GB). Capped Base-Stock (Ziel 29, Deckel 3) liegt dazu auf Strom 777 bei +0,67 % (aus den Daten), +0,92 % auf einem großen unabhängigen Strom (nur Messung des Prüfers)."""
    from dpg_texts import ASSUMPTIONS_MARKDOWN as T
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ref = REFS["L8"]
    assert ref["g_dp"] is None and (ref["S_cap"], ref["cap"]) == (29, 3)
    g_dp_measured = 5.782247                                           # Messung des Prüfers, nicht aus den Daten
    assert round(100 * (ref["g_cap"] / g_dp_measured - 1), 2) == 0.67
    for needle in ("g = 5,782247", "522 s", "5,95 GB", "32 GB", "+0,92 %", "+0,67 %", "rund +0,9 %", "Streamlit Cloud sind nicht gemessen"):
        assert needle in T and needle in readme, needle
    assert "Speicher ist die Grenze" not in readme and "Rechner mit mehr Speicher" not in readme and "kein Optimum bekannt" not in readme


def test_k22_repository_is_small():
    """K22: Gesamtgröße des Repos unter 3 MB (ohne .git, Caches), die Daten unter 100 KB."""
    skip = {".git", "__pycache__", ".pytest_cache", "venv", ".ruff_cache"}
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip))
    assert total < 3_000_000
    assert sum(p.stat().st_size for p in (ROOT / "dpg_data").glob("*.json")) < 100_000


def test_k23_assumptions_numbers_match_the_data():
    """K23: die Zahlen im Annahmen-Text (Randmasse, Q = 5, Spannen) sind dieselben wie oben; hier nur die Wörter, die an Daten hängen."""
    from dpg_texts import ASSUMPTIONS_MARKDOWN as T
    for needle in ("2,1e-6", "2,6e-5", "9,1e-5", "2,6e-4", "21 · 6^(L−1)", "0,1 und 1,6 %", "10 Zufalls-Seeds für L = 2 bis 4", "5 Zufalls-Seeds für L = 5 bis 8", "25.600"):
        assert needle in T, needle
    assert json.loads((ROOT / "dpg_data" / "curves.json").read_text(encoding="utf-8"))["meta"]["hyperparameter"]["hidden"] == 32
    assert re.search(r"Lernrate 3e-3", T) and C.LEARNING_RATE == 3e-3 and C.HIDDEN == 32

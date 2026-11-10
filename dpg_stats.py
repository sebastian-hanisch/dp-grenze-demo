"""Auswertung der vorgerechneten Lernläufe (dpg_data/curves.json) und der Bezugswerte (dpg_data/refs.json). Nur numpy.

Lücke = 100 * (Kosten / Bezug - 1) in Prozent. Bezug ist bei L <= 7 die DP (das Optimum), bei L = 8 Capped Base-Stock (die DP ist dort weder in der App noch in den Daten).
Ab L = 7 sind auch die Kosten der Regeln und des Netzes Simulationswerte (Standardfehler rund 0,1 %), bis L = 6 exakt.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import dpg_constants as C

DATA = Path(__file__).resolve().parent / "dpg_data"


@lru_cache(maxsize=1)
def load_curves() -> dict:
    return json.loads((DATA / "curves.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_refs() -> dict:
    return json.loads((DATA / "refs.json").read_text(encoding="utf-8"))


def reference_cost(ref: dict) -> float:
    """Bezug der Lücke: die DP (Optimum), wo sie gerechnet ist, sonst Capped Base-Stock."""
    return ref["g_dp"] if ref.get("g_dp") is not None else ref["g_cap"]


def gap_percent(g, base):
    return 100.0 * (np.asarray(g) / base - 1.0)


def gap_vs_reference(g: float, ref_cost: float, se: float = 0.0, se_ref: float = 0.0) -> tuple[float, float | None]:
    """Differenz zur Referenz in Prozent der Referenz und (wenn Standardfehler bekannt) in Standardfehlern der Differenz.

    Einheit „Lücke gegen DP oder gegen Capped“: bei exakt bewerteten Werten (se = 0) gibt es kein z (None)."""
    diff = g - ref_cost
    pct = 100.0 * diff / ref_cost
    s = float(np.hypot(se, se_ref))
    return pct, (diff / s if s > 0 else None)


def run_gaps(L: int, *, view: str = C.VIEW_PICK, curves=None, refs=None):
    """(Updates, Seeds, Matrix der Lücken in Prozent (Seeds x Messpunkte)) für Lieferzeit L.

    view = Abgriff: der Messpunkt selbst. view = Mittel der letzten fünf Messpunkte: Mittel der Lücken der fünf letzten Messpunkte bis
    einschließlich diesem; die ersten vier Messpunkte haben noch kein solches Mittel (nan)."""
    curves = curves or load_curves()
    refs = refs or load_refs()
    base = reference_cost(refs[f"L{L}"])
    runs = curves["runs"][str(L)]
    seeds = sorted(runs, key=int)
    updates = runs[seeds[0]]["updates"]
    assert all(runs[s]["updates"] == updates for s in seeds)
    mat = np.array([gap_percent(runs[s]["g"], base) for s in seeds])
    if view == C.VIEW_MEAN5:
        smooth = np.full_like(mat, np.nan)
        for j in range(4, mat.shape[1]):
            smooth[:, j] = mat[:, j - 4 : j + 1].mean(axis=1)
        mat = smooth
    return updates, seeds, mat


def curve_stats(L: int, *, curves=None, refs=None) -> dict:
    """Kennzahlen der Lernläufe einer Lieferzeit: Endstände, Streuung, Anteil unter 2 % und unter Capped, Abgriff gegen Mittel.

    final: Lücke je Zufalls-Seed bei Update 3.000 (Abgriff). below_capped: Zahl der Läufe, deren Lücke kleiner ist als die von Capped Base-Stock
    (Punktschätzung; bei L = 7 und 8 mit Rauschen, siehe z). late_*: Messpunkte ab Update 1.000."""
    curves = curves or load_curves()
    refs = refs or load_refs()
    ref = refs[f"L{L}"]
    base = reference_cost(ref)
    cap_gap = float(gap_percent(ref["g_cap"], base))
    updates, seeds, mat = run_gaps(L, curves=curves, refs=refs)
    _, _, mat5 = run_gaps(L, view=C.VIEW_MEAN5, curves=curves, refs=refs)
    final = mat[:, -1]
    late = [i for i, u in enumerate(updates) if u >= C.LATE_UPDATE]
    first_under = []
    for row in mat:
        hit = [updates[i] for i in range(len(updates)) if row[i] < C.GAP_LIMIT_PERCENT]
        first_under.append(hit[0] * C.PERIODS_PER_STEP if hit else np.nan)
    stats = dict(
        L=L, n_seeds=len(seeds), seeds=seeds, base=base, capped_gap=cap_gap, base_stock_gap=float(gap_percent(ref["g_bs"], base)),
        final=final.tolist(), final_median=float(np.median(final)), final_min=float(final.min()), final_max=float(final.max()),
        final_mean=float(final.mean()), final_sd=float(final.std(ddof=1)),
        below_capped=int((final < cap_gap).sum()), below_2pct=int((final < C.GAP_LIMIT_PERCENT).sum()),
        mean5_final=mat5[:, -1].tolist(), below_capped_mean5=int((mat5[:, -1] < cap_gap).sum()),
        late_points=len(late), late_frac_below_2pct=float((mat[:, late] < C.GAP_LIMIT_PERCENT).mean()),
        late_frac_below_capped=float((mat[:, late] < cap_gap).mean()),
        late_sd_median=float(np.median(mat[:, late].std(axis=1, ddof=1))), late_sd_max=float(mat[:, late].std(axis=1, ddof=1).max()),
        first_under_2pct_median=float(np.nanmedian(first_under)) if not np.all(np.isnan(first_under)) else None,
        first_under_2pct_never=int(np.isnan(first_under).sum()),
        updates=updates, median_curve=np.median(mat, axis=0).tolist(),
        sek_median=float(np.median([curves["runs"][str(L)][s]["sek"] for s in seeds])),
        sek_min=float(min(curves["runs"][str(L)][s]["sek"] for s in seeds)), sek_max=float(max(curves["runs"][str(L)][s]["sek"] for s in seeds)),
    )
    if ref.get("g_dp") is None:               # L = 8: Vergleich mit Capped in Standardfehlern
        zs = []
        for s in seeds:
            run = curves["runs"][str(L)][s]
            _, z = gap_vs_reference(run["g"][-1], ref["g_cap"], run["se"][-1], ref["se_cap"])
            zs.append(z)
        stats.update(z=zs, clearly_better=int(sum(z < -C.SE_BAND for z in zs)), clearly_worse=int(sum(z > C.SE_BAND for z in zs)),
                     better_point=int(sum(z < 0 for z in zs)))
    return stats


def min_difference_to_dp(curves=None, refs=None) -> float:
    """Orakel „nie unter dem Optimum“: kleinste Differenz (Kosten des Netzes minus DP) über alle exakt bewerteten Messpunkte (muss >= 0 sein)."""
    curves = curves or load_curves()
    refs = refs or load_refs()
    worst = np.inf
    for L, runs in curves["runs"].items():
        ref = refs[f"L{L}"]
        if ref.get("g_dp") is None or ref["art"] != "exakt":
            continue
        for run in runs.values():
            for g, mode in zip(run["g"], run["mode"]):
                if mode == "exakt":
                    worst = min(worst, g - ref["g_dp"])
    return float(worst)

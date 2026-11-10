"""Bezugswerte je Lieferzeit L: DP, beste Regeln, Zeiten, Randmasse, erreichbare Zustände -> dpg_data/refs.json.

Aufruf (nicht Teil der CI, rund 6 Minuten auf einem Arbeitsplatzrechner, L = 6 und 7 brauchen am meisten Speicher und Zeit):
    python scripts/precompute_refs.py [L ...]        # ohne Angabe: L = 1 bis 8

Bis L = 6: exakte Auswertung der Regeln (stationäre Verteilung), Gitter der Parameter wie unten. Ab L = 7: die DP ist noch gerechnet (L = 7),
die Regeln dagegen per Simulation mit gemeinsamen Zufallsströmen (Parameter auf Strom 101 abgestimmt, Kosten auf Strom 777 bewertet,
2.000 Ketten à 1.500 Perioden, 300 Einschwingperioden). L = 8: keine DP (Speicher), nur die Regeln. Zeiten sind lastbehaftet und gelten für
den Rechner, auf dem das Skript lief (Eintrag `rechner`): Größenordnung, nie eine feste Sekundenzahl.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import dpg_algorithm as A  # noqa: E402
import dpg_constants as C  # noqa: E402
import dpg_counts as N  # noqa: E402
import dpg_sim as S  # noqa: E402

OUT = ROOT / "dpg_data" / "refs.json"


def grids(L):
    if L <= 4:
        return range(5, 36), range(1, 9)
    c = 3 * L + 6
    if L == 5:
        return range(c - 5, c + 6), range(2, 6)
    return range(c - 3, c + 4), range(2, 5)


def exact(L):
    t0 = time.perf_counter()
    m = A.LostSales(L=L)
    t_setup = time.perf_counter() - t0
    g, pol, it, t_dp = A.timed_rvi(m)
    pi, _ = m.stationary(pol)
    s_range, caps = grids(L)
    gb, sb = m.best_base_stock(s_range)
    gc, sc, cap = m.best_capped(s_range, caps)
    gk, qk = min((m.evaluate(m.policy_constant(q))[0], q) for q in (range(0, 9) if L <= 5 else (2, 3, 4)))
    return dict(L=L, states=m.S, g_dp=g, g_bs=gb, S_bs=sb, g_cap=gc, S_cap=sc, cap=cap, g_const=gk, q_const=qk,
               edge=m.edge_mass(pi), order_cap=m.order_cap_mass(pi, pol), reach=A.reach_count(pi), t_dp=t_dp, it_dp=it, t_setup=t_setup, art="exakt")


def simulated(L):
    tune = S.demand_stream(C.SIM_TUNE_SEED, C.SIM_PERIODS, C.SIM_CHAINS)
    ev = S.demand_stream(C.DEFAULT_SIM_SEED, C.SIM_PERIODS, C.SIM_CHAINS)
    s_range, caps = grids(L)
    best_b = min((S.simulate(L, S.base_stock(s), tune).mean(), s) for s in s_range)
    best_c = min((S.simulate(L, S.capped(s, cap), tune).mean(), s, cap) for cap in caps for s in s_range)
    vb = S.simulate(L, S.base_stock(best_b[1]), ev)
    vc = S.simulate(L, S.capped(best_c[1], best_c[2]), ev)
    vk, qk = min(((S.simulate(L, lambda x, pipe, q=q: np.full(x.shape, q), ev), q) for q in range(2, 5)), key=lambda t: t[0].mean())
    mb, sb = S.mean_and_se(vb)
    mc, sc = S.mean_and_se(vc)
    dm, ds = S.mean_and_se(vb, vc)
    rec = dict(L=L, states=N.lost_sales_states(L), g_dp=None, g_bs=mb, se_bs=sb, S_bs=best_b[1], g_cap=mc, se_cap=sc, S_cap=best_c[1], cap=best_c[2],
               g_const=float(vk.mean()), q_const=qk, diff_bs_cap=dm, se_diff=ds, art="sim")
    if L == 7:                                    # die DP rechnet bis hier (Tensor-Fassung, rund 11 Millionen Zustände)
        m = A.LostSales(L=7)
        g, pol, it, t_dp = A.timed_rvi(m)
        rec.update(g_dp=g, t_dp=t_dp, it_dp=it)
        pi, _ = m.stationary(pol)
        rec.update(edge=m.edge_mass(pi), order_cap=m.order_cap_mass(pi, pol), reach=A.reach_count(pi))
    return rec


def q5_check(L):
    """Gleiche Kosten mit Bestellgrenze Q = 5 statt 8 (die DP bestellt höchstens 5)? Zustände 21 * 6^(L-1)."""
    g8 = A.LostSales(L=L).rvi()[0]
    m5 = A.LostSales(L=L, Q=5)
    return dict(g_q8=g8, g_q5=m5.rvi()[0], states_q5=m5.S)


def main(argv):
    refs = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    Ls = [int(a) for a in argv] or list(range(1, 9))
    for L in Ls:
        t0 = time.time()
        rec = exact(L) if L <= 6 else simulated(L)
        if L <= 5:
            rec["q5"] = q5_check(L)
        refs[f"L{L}"] = rec
        refs["meta"] = dict(rechner=f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, numpy {np.__version__}",
                            hinweis="Zeiten lastbehaftet (andere Prozesse liefen), nur Größenordnung; ein Kern je Prozess (OMP_NUM_THREADS=1)")
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"L = {L}: fertig in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Kennzahlen, Verdikte (reine Funktionen: Art, Text) und die Rechnungen hinter den Abschnitten der App."""

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import dpg_algorithm as A
import dpg_constants as C
import dpg_counts as N
import dpg_learn as LN
import dpg_sim as S
import dpg_stats as ST


def de(x: float, nd: int = 1, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma; sign=True mit Vorzeichen (+ oder −)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def seconds_text(t: float) -> str:
    """Zeit als Größenordnung: unter 0,05 s als „weniger als 0,05 s“, sonst „rund“ mit einer Nachkommastelle (ab 10 s ohne)."""
    if t < 0.05:
        return "weniger als 0,05 s"
    return f"rund {de(t, 1)} s" if t < 10 else f"rund {t:.0f} s"


@dataclass(frozen=True)
class Settings:
    l: int = C.DEFAULT_L
    view: str = C.VIEW_PICK
    sim_seed: int = C.DEFAULT_SIM_SEED


# ---- Verdikte ------------------------------------------------------------------------------------------------------------------------------

def verdict_growth(L: int, states: int, *, Q: int = C.Q_MAX) -> tuple[str, str]:
    """Wachstum der Zustandszahl bei Lieferzeit L: ab etwa 0,5 GB je Wertetabelle eine Warnung."""
    size = N.table_bytes(states)
    text = (f"Bei Lieferzeit L = {L} hat die Wertetabelle **{N.format_count(states)} Zustände** ({N.format_bytes(size)} je Tabelle). "
            f"Jede zusätzliche Periode Lieferzeit multipliziert die Zahl mit Q + 1 = {Q + 1}.")
    return ("warning" if size >= 5e8 else "info"), text


def verdict_dp_time(L: int, states: int, seconds: float | None, source: str) -> tuple[str, str]:
    """Rechenzeit der DP: source = 'live' (hier gerechnet), 'vorgerechnet' (Messung des Messrechners) oder 'keine' (nicht gerechnet)."""
    if source == "keine" or seconds is None:
        text = (f"Bei L = {L} sind es {N.format_count(states)} Zustände ({N.format_bytes(N.table_bytes(states))} je Wertetabelle, die Rechnung hält mehrere); "
                "die DP wurde hier nicht gerechnet: Rechenzeit und Speicher sind für eine App aufwendig, unmöglich ist die Rechnung nicht. ")
        if L == 8:
            text += "Einmalige Messung außerhalb der App (Prüfer): 522 s, 5,95 GB Spitzenspeicher (siehe Annahmen). "
        return "warning", text + "Rechenzeit und Speicher auf Streamlit Cloud sind nicht gemessen."
    where = "in dieser Sitzung gerechnet" if source == "live" else "vorgerechnet auf dem Messrechner"
    return "info", f"Die DP braucht bei L = {L} **{seconds_text(seconds)}** ({where}; Größenordnung, abhängig von Rechner, Last und Programmierung)."


def verdict_learning(stats: dict | None, L: int) -> tuple[str, str]:
    """Verdikt zum gelernten Netz: Median der Endstände, Capped Base-Stock und Zählung am Endstand; bei L = 8 in Standardfehlern gegen Capped."""
    if stats is None:
        return "info", f"Bei L = {L} gibt es keine Lernläufe: der Zustand hat nur {N.format_count(N.lost_sales_states(L))} Werte, die DP rechnet ihn sofort."
    n = stats["n_seeds"]
    if "z" in stats:
        lower = "keiner" if stats["clearly_better"] == 0 else str(stats["clearly_better"])
        text = (f"Bei L = {L} rechnet die App die DP nicht; Maßstab ist Capped Base-Stock. Von {n} Läufen liegt am Endstand {lower} erkennbar darunter "
                f"({stats['clearly_better']} von {n} mit Abstand über 2 Standardfehlern, Fehler von Netz und Regel zusammengesetzt; der gepaarte Fehler auf demselben Strom ist kleiner, "
                f"so dass ein Lauf mit gepaartem Fehler erkennbar besser sein kann), {stats['clearly_worse']} liegen erkennbar darüber "
                f"({stats['better_point']} von {n} haben als Punktschätzung geringere Kosten).")
        return ("warning" if stats["clearly_better"] == 0 and stats["clearly_worse"] > 0 else "info"), text
    text = (f"Bei L = {L} liegt das Netz im Median der Endstände bei {de(stats['final_median'], 1, True)} % über der DP "
            f"(Bereich {de(stats['final_min'], 1, True)} bis {de(stats['final_max'], 1, True)} %), Capped Base-Stock bei {de(stats['capped_gap'], 1, True)} %. "
            f"Am Endstand liegen {stats['below_capped']} von {n} Läufen unter Capped Base-Stock.")
    return ("success" if 2 * stats["below_capped"] > n else "info"), text


def verdict_benchmark(rows: dict) -> tuple[str, str]:
    """Verdikt zur Simulation mit gemeinsamen Zufallsströmen: Differenz Base-Stock minus Capped und beide Standardfehler (gepaart, getrennt)."""
    d, se_paired, se_unpaired = rows["diff"], rows["se_paired"], rows["se_unpaired"]
    more = "mehr" if d >= 0 else "weniger"
    text = (f"Base-Stock kostet {de(abs(d), 3)} je Periode {more} als Capped Base-Stock (Standardfehler {de(se_paired, 3)} bei gemeinsamem Strom, "
            f"{de(se_unpaired, 3)} bei getrennten Strömen")
    if se_paired > 0:
        text += f"; der gemeinsame Strom macht den Fehler {de(se_unpaired / se_paired, 1)}-mal kleiner"
    return "info", text + ")."


def verdict_order_cap(g_q5: float, g_q8: float, states_q5: int, states_q8: int) -> tuple[str, str]:
    """Bestellgrenze Q = 5 gegen Q = 8: gleiche Kosten (Unterschied unter 1e-9) heißt, die größere Grenze wird nie gebraucht."""
    if abs(g_q5 - g_q8) < 1e-9:
        return "success", (f"Mit Q = 5 statt Q = 8 ergeben sich dieselben Kosten ({de(g_q8, 4)}), aber nur {N.format_count(states_q5)} statt "
                           f"{N.format_count(states_q8)} Zustände: der Faktor hängt an der gewählten Bestellgrenze, nicht an der Nachfrage.")
    return "warning", (f"Mit Q = 5 statt Q = 8 steigen die Kosten von {de(g_q8, 4)} auf {de(g_q5, 4)}: die Bestellgrenze greift hier schon.")


# ---- Rechnungen -----------------------------------------------------------------------------------------------------------------------------

def live_dp(L: int) -> dict:
    """DP für Lieferzeit L <= 5 in dieser Sitzung: Kosten, Politik-Kennzahlen, Zeit der Wertiteration."""
    m = A.LostSales(L=L)
    g, pol, it, t = A.timed_rvi(m)
    pi, _ = m.stationary(pol)
    return dict(L=L, states=m.S, g=g, seconds=t, iterations=it, reach=A.reach_count(pi), edge=m.edge_mass(pi), model=m, policy=pol)


def order_cap_table(L: int, caps=(3, 4, 5, 6, 8), g_default: float | None = None) -> list[dict]:
    """Bestellgrenze Q gegen Zustandszahl und DP-Kosten, live (L <= 5). Ist die DP für Q = 8 schon gerechnet, kann ihr g mitgegeben werden (g_default)."""
    rows = []
    for Q in caps:
        if Q == C.Q_MAX and g_default is not None:
            rows.append(dict(Q=Q, states=N.lost_sales_states(L, Q=Q), g=g_default))
            continue
        m = A.LostSales(L=L, Q=Q)
        rows.append(dict(Q=Q, states=m.S, g=m.rvi()[0]))
    base = next(r["g"] for r in rows if r["Q"] == C.Q_MAX)
    for r in rows:
        r["diff"] = r["g"] - base
    return rows


def simulate_rules(L: int, seed: int, refs: dict, dp: dict | None) -> dict:
    """Base-Stock, Capped Base-Stock (Parameter aus refs) und, wenn die DP gerechnet ist, ihre Politik auf EINEM Strom mit Zufalls-Seed `seed`.

    Gibt Mittel und Standardfehler je Politik sowie die gepaarte Differenz Base-Stock minus Capped (gemeinsamer Strom) und deren Standardfehler bei
    getrennten Strömen zurück."""
    ref = refs[f"L{L}"]
    stream = S.demand_stream(seed, C.SIM_PERIODS, C.SIM_CHAINS)
    vb = S.simulate(L, S.base_stock(ref["S_bs"]), stream)
    vc = S.simulate(L, S.capped(ref["S_cap"], ref["cap"]), stream)
    rows = {"Base-Stock": S.mean_and_se(vb), "Capped Base-Stock": S.mean_and_se(vc)}
    if dp is not None:
        rows["DP-Politik"] = S.mean_and_se(S.simulate(L, S.table_policy(dp["model"], dp["policy"]), stream))
    diff, se_paired = S.mean_and_se(vb, vc)
    return dict(rows=rows, diff=diff, se_paired=se_paired, se_unpaired=float(np.hypot(rows["Base-Stock"][1], rows["Capped Base-Stock"][1])),
                params=dict(S_bs=ref["S_bs"], S_cap=ref["S_cap"], cap=ref["cap"]))


# ---- Handbeispiel ---------------------------------------------------------------------------------------------------------------------------

HAND_PD = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))


def hand_model() -> A.LostSales:
    """Mini-Instanz: Lieferzeit 2, Bestellmenge 0 oder 1, Bestand höchstens 2, Nachfrage 0/1/2 mit 1/4, 1/2, 1/4, h = 1, p = 9."""
    return A.LostSales(L=2, Q=1, XM=2, pd=[float(p) for p in HAND_PD], hold=1.0, penalty=9.0)


def hand_example() -> dict:
    """Alle Zahlen des Handbeispiels: erwartete Periodenkosten, Rückwärtsrechnung (3 Perioden), Durchschnittskosten, Aufzählung aller 64 Politiken."""
    m = hand_model()
    values, actions = m.finite_horizon(3)
    states = [(x, p1) for x in range(3) for p1 in range(2)]        # Zeilen (x, p1) in der Reihenfolge der Tabelle (X, Pn)
    g, pol, _, _ = m.rvi(tol=1e-12)
    best = (np.inf, None)
    for acts in itertools.product(range(2), repeat=len(states)):
        tab = np.array(acts, dtype=np.int16).reshape(3, 2)
        cost, _ = m.evaluate(tab, tol=1e-12)
        if cost < best[0] - 1e-12:
            best = (cost, acts)
    always_one, _ = m.evaluate(m.policy_constant(1), tol=1e-12)
    c0 = m.ec[0] + values[1][0, 0]                                   # Kosten der Aktion 0 in Zustand (0, 0) bei drei Perioden Restlaufzeit
    c1 = m.ec[0] + values[1][0, 1]                                   # Aktion 1: danach ist die Bestellung p1 = 1
    pi = [0.5, 0.5]
    return dict(ec=m.ec.tolist(), values=values, actions=actions, states=states, g=g, best_g=best[0], always_one=always_one,
                costs_action=(float(c0), float(c1)), gradient_exact=LN.score_gradient_hand(pi, [c0, c1]),
                gradient_reinforce=LN.reinforce_expectation(pi, [c0, c1]), n_policies=2 ** len(states))


def hand_state_counts(Ls=(1, 2, 3, 4, 5)) -> list[dict]:
    """Zustandszahl und Aufwand je Wertiteration der Mini-Instanz (Q = 1, XM = 2, 3 Nachfragewerte): Faktor Q + 1 = 2 je Periode Lieferzeit."""
    return [dict(L=L, states=N.lost_sales_states(L, Q=1, XM=2), terms=N.lost_sales_states(L, Q=1, XM=2) * 2 * 3) for L in Ls]


def learning_stats(L: int):
    """Kennzahlen der Lernläufe oder None, wenn es für L keine gibt."""
    return ST.curve_stats(L) if L in C.LEARN_L else None

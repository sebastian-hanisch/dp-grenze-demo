"""Wo die exakte Rechnung aufgibt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zustandszahl, Rechenzeit und Speicher der exakten Rückwärtsrechnung (DP) bei einem Bestandsproblem mit Lieferzeit, dazu die vorgerechneten
Lernkurven eines Politikgradient-Netzes und der Maßstab, wenn die DP nicht mehr rechnet. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dpg_constants as C
import dpg_counts as N
import dpg_evaluation as E
import dpg_stats as ST
import dpg_visualization as V
from dpg_layout import dunkles_design, kopf_css
from dpg_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from dpg_texts import ASSUMPTIONS_MARKDOWN

st.set_page_config(page_title="Wo die exakte Rechnung aufgibt – Sebastian Hanisch", layout="wide")


@st.cache_resource(show_spinner=False)
def _live_dp_full(L):
    return E.live_dp(L)


def _live_dp(L):
    return {k: v for k, v in _live_dp_full(L).items() if k not in ("model", "policy")}


@st.cache_data(show_spinner=False)
def _order_caps(L, g_q8):
    return E.order_cap_table(L, g_default=g_q8)


@st.cache_data(show_spinner=False)
def _simulate(L, seed):
    dp = _live_dp_full(L) if L <= C.LIVE_DP_MAX_L else None
    return E.simulate_rules(L, seed, ST.load_refs(), dp)


@st.cache_data(show_spinner=False)
def _hand():
    return E.hand_example()


def pct(x, nd=1):
    return E.de(x, nd, True) + " %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dpg_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Wo die exakte Rechnung aufgibt")
st.markdown(
    """
Die **exakte Rückwärtsrechnung** (dynamische Programmierung, DP) gibt jedem Zustand einen Wert und liefert so die beste Bestellregel. Bei **Lieferzeit** gehören die
**Bestellungen unterwegs** zum Zustand, und bei verlorener Nachfrage multipliziert jede Periode Lieferzeit die Zahl der Zustände mit der Bestellgrenze plus 1. Die Demo misst,
**wo die DP aufgibt**, und was ein per **Politikgradient gelerntes Netz** dann leistet: gemessen gegen die DP, solange die App sie hat (bis L = 7), danach gegen eine einfache Regel
(**Capped Base-Stock**). Das Beispiel ist in der ganzen Reihe „Bestandspolitiken“ dasselbe: ein Artikel, Poisson-Nachfrage mit Mittel 3, verlorene Nachfrage.

Weiter unten: die Abschnitte **Zustandszahl**, **Rechenzeit und Speicher**, **Gelerntes Netz**, **Maßstab**, **Handbeispiel** und **Wo die Annahmen enden**, dazu die Expander
**⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Rückwärtsrechnung – und woran sie scheitert", expanded=True):
    st.markdown(
        """
1. **Zustand:** der Bestand und alle Bestellungen, die noch unterwegs sind. Bei Lieferzeit L = 3 sind das der Bestand und zwei offene Bestellungen.
2. **Wert je Zustand:** die DP rechnet für jeden Zustand aus, wie teuer es auf Dauer im Mittel wird, wenn man jetzt eine bestimmte Menge bestellt und danach immer das Beste tut
   (**Wertiteration**: die Tabelle wird so lange verbessert, bis sie sich nicht mehr ändert). Die beste Politik bestellt in jedem Zustand die Menge mit den geringsten Kosten.
3. **Warum die Tabelle wächst:** zu jedem Bestand gehört jede mögliche Kombination offener Bestellungen. Mit höchstens Q = 8 Stück je Bestellung hat jede offene Bestellung 9 Werte (0 bis 8); jede Periode Lieferzeit mehr multipliziert die Zahl der Zustände mit 9.
4. **Die Alternative:** ein **Netz** lernt aus Simulation. Es bestellt zufällig nach eigenen Wahrscheinlichkeiten, merkt sich die abgezinsten Kosten der nächsten Perioden (Abschnitte von 100 Perioden) und verschiebt die Wahrscheinlichkeiten zu Bestellungen, die billiger waren als erwartet
   (**Politikgradient**, Verfahren REINFORCE mit gelerntem Basiswert). Es braucht keine Tabelle über alle Zustände, aber Millionen simulierter Perioden, und es trifft das Optimum nicht genau.
5. **Zwei einfache Regeln als Maßstab:** **Base-Stock** bestellt bis zu einer Zielhöhe für Bestand plus Unterwegs; **Capped Base-Stock** tut dasselbe, bestellt aber je Periode höchstens einen Deckel (hier 3 bis 4 Stück).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Beispiel**")
    L = st.slider("Lieferzeit L (Perioden)", *bounds("l_slider"), key="l_slider",
                  help="Wie viele Perioden eine Bestellung bis zur Ankunft braucht. Jede Periode mehr multipliziert die Zahl der Zustände mit 9.")
    st.markdown("**Lernkurven**")
    view_index = st.radio("Anzeige der Lernkurven", options=[0, 1], format_func=lambda i: C.VIEWS[i], key="view_radio",
                          help="Abgriff: die Kosten der Politik des Netzes zum Messpunkt. Mittel: der Durchschnitt der fünf letzten Messpunkte, das glättet das Wandern um das Optimum.")
    st.markdown("**Simulation (Maßstab)**")
    sim_seed = st.number_input("Zufalls-Seed des Nachfragestroms", *bounds("sim_seed_input"), key="sim_seed_input", step=1,
                               help="Alle Regeln sehen denselben Nachfragestrom; ein anderer Zufalls-Seed gibt einen anderen Strom.")
    st.button("🎲 Neuen Nachfragestrom ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"l_slider": int(L), "view_radio": int(view_index), "sim_seed_input": int(sim_seed)})

settings = E.Settings(l=int(L), view=C.VIEWS[int(view_index)], sim_seed=int(sim_seed))
refs = ST.load_refs()
ref = refs[f"L{settings.l}"]
states = N.lost_sales_states(settings.l)
stats = E.learning_stats(settings.l)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

if settings.l <= C.LIVE_DP_MAX_L:
    with st.spinner("Rechnet die DP für diese Lieferzeit (bis L = 4 Bruchteile einer Sekunde, bei L = 5 auf dem Messrechner etwa eine Sekunde) …"):
        dp = _live_dp(settings.l)
    dp_seconds, dp_source = dp["seconds"], "live"
elif ref.get("t_dp") is not None:
    dp, dp_seconds, dp_source = None, ref["t_dp"], "vorgerechnet"
else:
    dp, dp_seconds, dp_source = None, None, "keine"

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Zustände", N.format_count(states), delta=f"× {C.Q_MAX + 1} je Periode Lieferzeit", delta_color="off")
m2.metric("Wertetabelle (eine)", N.format_bytes(N.table_bytes(states)), delta="8 Byte je Zustand", delta_color="off")
m3.metric("Rechenzeit der DP", E.seconds_text(dp_seconds) if dp_seconds is not None else "nicht gerechnet",
          delta={"live": "in dieser Sitzung", "vorgerechnet": "vorgerechnet (Messrechner)", "keine": "hier nicht gerechnet"}[dp_source], delta_color="off")
if stats is None:
    m4.metric("Gelerntes Netz", "keine Lernläufe", delta="Zustand zu klein", delta_color="off")
elif "z" in stats:
    m4.metric("Netz gegen Capped (Median)", pct(stats["final_median"], 2), delta="Maßstab: Capped Base-Stock", delta_color="off")
else:
    m4.metric("Netz, Lücke zur DP (Median)", pct(stats["final_median"], 1), delta=f"Capped Base-Stock: {pct(stats['capped_gap'], 1)}", delta_color="off")
st.caption("Lücke = Abstand der durchschnittlichen Kosten je Periode zur DP in Prozent der DP-Kosten; bei L = 8 zu Capped Base-Stock, weil die App dort keine DP hat.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wo gibt die exakte Rechnung auf, und was bleibt dann?")
for kind, text in (E.verdict_growth(settings.l, states), E.verdict_dp_time(settings.l, states, dp_seconds, dp_source), E.verdict_learning(stats, settings.l)):
    getattr(st, kind)(text)

st.markdown("---")

tab_count, tab_time, tab_net, tab_bench, tab_hand, tab_limits = st.tabs(
    ["📊 Zustandszahl", "⏱️ Rechenzeit und Speicher", "🧠 Gelerntes Netz", "📏 Maßstab", "✏️ Handbeispiel", "🚧 Wo die Annahmen enden"]
)

# --- Zustandszahl --------------------------------------------------------------------------------------------------------------------------

with tab_count:
    st.subheader("🔬 Experiment: Wie schnell wächst die Tabelle mit der Lieferzeit?")
    st.caption("Exakte Zählung: (Bestandsgrenze + 1) · (Q + 1)^(L − 1) bei verlorener Nachfrage, Bestandsgrenze 20.")
    st.plotly_chart(V.build_state_growth(settings.l), width="stretch", key="chart_growth")
    st.dataframe(
        [dict(L=Lx, Zustände=N.format_count(N.lost_sales_states(Lx)), **{"Zustand-Aktion-Paare": N.format_count(N.state_pairs(N.lost_sales_states(Lx)))},
              **{"Wertetabelle (eine)": N.format_bytes(N.table_bytes(N.lost_sales_states(Lx)))}) for Lx in range(1, 10)],
        hide_index=True, width="stretch",
    )

    st.subheader("🔬 Experiment: Hängt der Faktor 9 an der Nachfrage oder an der Bestellgrenze?")
    st.caption("Die DP wird mit verschiedenen Bestellgrenzen Q live gerechnet (für die Lieferzeit der Seitenleiste, bis L = 5).")
    if settings.l <= C.LIVE_DP_MAX_L:
        with st.spinner("Rechnet die DP für fünf Bestellgrenzen …"):
            rows = _order_caps(settings.l, dp['g'])
        st.dataframe(
            [{"Bestellgrenze Q": r["Q"], "Zustände": N.format_count(r["states"]), "DP-Kosten je Periode": E.de(r["g"], 4),
              "Abstand zu Q = 8": E.de(r["diff"], 4, True)} for r in rows],
            hide_index=True, width="stretch",
        )
        r5 = next(r for r in rows if r["Q"] == 5)
        r8 = next(r for r in rows if r["Q"] == C.Q_MAX)
        kind, text = E.verdict_order_cap(r5["g"], r8["g"], r5["states"], r8["states"])
        getattr(st, kind)(text)
    else:
        st.info("Für L ≥ 6 rechnet die App diese Tabelle nicht live. Für L = 1 bis 5 ergibt Q = 5 dieselben Kosten wie Q = 8.")

    st.subheader("🔬 Experiment: Zwei weitere Wachstumsrichtungen (nur gezählt)")
    st.markdown(
        "**Haltbarkeit** (verderbliche Ware, Bestand je Restlebensdauer, höchstens 8 Stück je Altersklasse) und **zwei Stufen in Reihe** (Händler und Lager, jede mit eigener Lieferzeit) "
        "wachsen nach derselben Regel: jede zusätzliche Periode multipliziert mit 9. Die Tabellen zeigen die Zustandszahl nach Formel; die DP wird für diese beiden Richtungen hier nicht gerechnet."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Haltbarkeit m (Restlebensdauer in Perioden)**")
        st.dataframe(
            [{"m": m, "Lieferzeit 0": N.format_count(N.perishable_states(m, 0)), "Lieferzeit 1": N.format_count(N.perishable_states(m, 1))} for m in range(2, 9)],
            hide_index=True, width="stretch",
        )
    with c2:
        st.markdown("**Zwei Stufen: Lieferzeit Händler / Lager**")
        st.dataframe(
            [{"Händler": lr, **{f"Lager {lw}": N.format_count(N.two_stage_states(lr, lw)) for lw in (1, 2, 3)}} for lr in (1, 2, 3)],
            hide_index=True, width="stretch",
        )
    st.caption("Zwei Stufen: 56 Werte Händlerbestand · 19 Werte Lagerbestand · 9^(Lieferzeit Händler + Lieferzeit Lager − 2); der Händlerbestandsbereich entspricht dem Aufbau der Demo „Lager und Händler“.")

# --- Rechenzeit und Speicher ---------------------------------------------------------------------------------------------------------------

with tab_time:
    st.subheader("🔬 Experiment: Setzt die Rechenzeit oder der Speicher die Grenze?")
    st.markdown(
        "Die Wertetabelle wächst mit Faktor 9 je Periode. Die **Rechenzeit der DP** ist auf einem Arbeitsplatzrechner kein früher Anschlag (bei L = 7 rund eine Minute); bei L = 8 sind es rund 0,8 GB je Tabelle, "
        "und die App rechnet die DP dort nicht. Eine einmalige Messung außerhalb der App (Prüfer, Rechner mit 32 GB) brauchte 522 s und 5,95 GB Spitzenspeicher: die DP ist bei L = 8 möglich, aber für eine App aufwendig "
        "(Streamlit Cloud nicht gemessen). Zeiten sind Größenordnungen dieser Umsetzung (siehe Annahmen), nicht Garantien."
    )
    time_rows = []
    for Lx in range(1, 10):
        rx = refs.get(f"L{Lx}")
        s_x = N.lost_sales_states(Lx)
        if rx and rx.get("t_dp") is not None:
            t_text, reach = E.seconds_text(rx["t_dp"]), rx.get("reach")
        else:
            t_text, reach = "nicht gerechnet", None
        time_rows.append({"L": Lx, "Zustände": N.format_count(s_x), "Wertetabelle (eine)": N.format_bytes(N.table_bytes(s_x)), "DP-Zeit (Messrechner)": t_text,
                          "Iterationen": rx.get("it_dp") if rx and rx.get("it_dp") else None,
                          "unter der DP-Politik erreichbar": N.format_count(reach) if reach is not None else "–",
                          "Anteil erreichbar": (E.de(100 * reach / s_x, 2) + " %") if reach is not None else "–"})
    st.dataframe(time_rows, hide_index=True, width="stretch")
    st.caption("Zeiten: Wertiteration allein (ohne Aufbau), ein Rechenkern, Messrechner; L = 8 und 9 in der App nicht gerechnet (L = 8 einmalig außerhalb gemessen, siehe Annahmen). Iterationen: Schritte bis zur Änderung unter 1e-9.")
    if dp is not None:
        st.success(f"In dieser Sitzung gerechnet: L = {settings.l}, {N.format_count(dp['states'])} Zustände, {E.seconds_text(dp['seconds'])} in {dp['iterations']} Iterationen; "
                   f"unter der DP-Politik kommen {N.format_count(dp['reach'])} Zustände vor ({E.de(100 * dp['reach'] / dp['states'], 2)} %).")
    reach_ratios = [refs[f"L{Lx + 1}"]["reach"] / refs[f"L{Lx}"]["reach"] for Lx in range(1, 7)]
    st.plotly_chart(V.build_reach(refs), width="stretch", key="chart_reach")
    st.markdown(
        "**Erreichbar** heißt: Zustände, die unter der besten Politik mit mehr als 1e-9 Anteil der Zeit vorkommen. Ihre Zahl wächst je Periode Lieferzeit nur um den Faktor "
        f"{E.de(min(reach_ratios), 1)} bis {E.de(max(reach_ratios), 1)} statt 9 (Größenordnungen: sie hängen an der Schwelle 1e-9; bei L = 5 ergeben 1e-6 / 1e-9 / 1e-11 2.633 / 4.785 / 5.451 Zustände). "
        "Wer die beste Politik schon kennt, braucht die Tabelle also nicht; man kennt sie vorher nicht."
    )

# --- Gelerntes Netz ------------------------------------------------------------------------------------------------------------------------

with tab_net:
    st.subheader("🔬 Experiment: Wie nah kommt ein gelerntes Netz an die DP, und wie stark streut das?")
    st.markdown(
        "Das Netz sieht nur Simulationen. Pro Lieferzeit liegen **vorgerechnete Lernläufe** mit verschiedenen **Zufalls-Seeds** vor (dpg_data/curves.json; die App rechnet sie nicht: "
        "ein Lauf braucht Minuten). Gemessen wird in Abständen von 50 bis 500 Updates die Güte der Politik, die das Netz gerade wählen würde (**Abgriff**). Jeder Lauf: 3.000 Updates à 25.600 simulierten Perioden."
    )
    if stats is None:
        st.info(f"Für L = {settings.l} gibt es keine Lernläufe: Die DP rechnet {N.format_count(states)} Zustände sofort.")
    else:
        st.plotly_chart(V.build_learning(settings.l, settings.view), width="stretch", key="chart_learning")
        st.caption("Dünne Linien: je ein Zufalls-Seed. Anfangswerte (über 100 %) liegen außerhalb der festen Achse. Anzeige in der Seitenleiste umschaltbar (Abgriff oder Mittel der letzten fünf Messpunkte).")
        a, b, c = st.columns(3)
        n = stats["n_seeds"]
        a.metric("Läufe (Zufalls-Seeds)", n)
        a_label = "Endstand: Median" if "z" not in stats else "Endstand gegen Capped: Median"
        b.metric(a_label, pct(stats["final_median"], 2), delta=f"Bereich {E.de(stats['final_min'], 1, True)} bis {E.de(stats['final_max'], 1, True)} %", delta_color="off")
        c.metric("Laufzeit je Lauf (Median)", f"{stats['sek_median'] / 60:.1f} min".replace(".", ","), delta="Messrechner, andere Prozesse liefen mit", delta_color="off")
        st.plotly_chart(V.build_final_gaps(settings.l, stats), width="stretch", key="chart_final")
        if "z" in stats:
            st.markdown(
                "**Gegen Capped Base-Stock in Standardfehlern** (Netz und Regel auf demselben Strom bewertet, Fehler beider zusammengesetzt, also ohne Paarung und eher zu groß): " + ", ".join(E.de(z, 1, True) for z in stats["z"])
                + f". Erkennbar besser heißt hier: Abstand größer als {C.SE_BAND:.0f} Standardfehler. Der gepaarte Fehler auf demselben Strom ist kleiner (Messung des Prüfers: rund 0,0023 statt 0,0085); mit ihm wäre der beste Lauf erkennbar besser (z = −4,2)."
            )
        else:
            st.markdown(
                f"**Streuung.** Zwischen den Zufalls-Seeds beträgt die Standardabweichung des Endstands {E.de(stats['final_sd'], 2)} Prozentpunkte; innerhalb eines Laufs schwankt der Abgriff ab Update {C.LATE_UPDATE:,}".replace(",", ".")
                + f" mit einer Standardabweichung von im Median {E.de(stats['late_sd_median'], 2)} Punkten. Beides liegt in derselben Größenordnung; ein Teil der Unterschiede am Endstand dürfte deshalb vom Zeitpunkt des Abgriffs stammen (Deutung, nicht getrennt gemessen). "
                f"Ab Update {C.LATE_UPDATE:,}".replace(",", ".") + f" liegen {E.de(100 * stats['late_frac_below_2pct'], 0)} % aller Messpunkte unter {C.GAP_LIMIT_PERCENT:.0f} % Lücke "
                f"und {E.de(100 * stats['late_frac_below_capped'], 0)} % unter der Lücke von Capped Base-Stock."
            )
        st.markdown(f"**Mittel der letzten fünf Messpunkte:** {stats['below_capped_mean5']} von {n} Läufen liegen unter Capped Base-Stock (Abgriff am Endstand: {stats['below_capped']} von {n}).")
        kind, text = E.verdict_learning(stats, settings.l)
        getattr(st, kind)(text)
    st.markdown("**Über alle Lieferzeiten** (Endstand, Abgriff bei Update 3.000):")
    all_rows = []
    for Lx in C.LEARN_L:
        sx = E.learning_stats(Lx)
        all_rows.append({"L": Lx, "Läufe": sx["n_seeds"], "Median": pct(sx["final_median"], 2), "Bereich": f"{E.de(sx['final_min'], 2, True)} bis {E.de(sx['final_max'], 2, True)} %",
                         "Capped Base-Stock": pct(sx["capped_gap"], 2) if "z" not in sx else "Maßstab",
                         "unter Capped (Endstand)": f"{sx['below_capped']} von {sx['n_seeds']}" if "z" not in sx else f"{sx['better_point']} von {sx['n_seeds']} (Punktschätzung), {sx['clearly_better']} erkennbar"})
    st.dataframe(all_rows, hide_index=True, width="stretch")
    st.caption("Bezug: Lücke zur DP in Prozent der DP-Kosten (L = 2 bis 7), L = 8: Abstand zu Capped Base-Stock. Ab L = 7 sind Netz und Regeln Simulationswerte (Standardfehler etwa 0,1 %; gegen die exakte DP mit einem gemeinsamen Versatz von bis zu etwa 0,2 Punkten nach unten, Messung des Prüfers, siehe Annahmen).")

# --- Maßstab -------------------------------------------------------------------------------------------------------------------------------

with tab_bench:
    st.subheader("🔬 Experiment: Womit misst man die Lücke, wenn die DP nicht mehr rechnet?")
    st.markdown(
        "1. **Solange die App die DP hat, gegen sie** (hier bis L = 7; bei L = 8 gibt es nur eine einmalige Messung außerhalb der App).\n"
        "2. **Danach gegen eine einfache Regel**, deren Parameter auf einem Zufallsstrom abgestimmt und auf einem anderen bewertet werden; sonst ist der Vergleich zu ihren Gunsten verzerrt.\n"
        "3. **Gemeinsame Zufallsströme** für alle verglichenen Politiken: Die Streuung einer Differenz wird viel kleiner.\n"
        "4. **Eine Lücke gegen eine einfache Regel unterschätzt die Lücke zum Optimum** um die eigene Lücke dieser Regel."
    )
    st.plotly_chart(V.build_benchmark_gaps(refs), width="stretch", key="chart_bench")
    st.dataframe(
        [{"L": Lx, "Capped Base-Stock (Ziel S, Deckel)": f"{refs[f'L{Lx}']['S_cap']}, {refs[f'L{Lx}']['cap']}", "Base-Stock (Ziel S)": refs[f"L{Lx}"]["S_bs"],
          "Bewertung": "exakt" if refs[f"L{Lx}"]["art"] == "exakt" else "Simulation"} for Lx in range(1, 9)],
        hide_index=True, width="stretch",
    )
    st.caption("Balken: Lücke zur DP in % der DP-Kosten (L = 1 bis 7), bei L = 8 Base-Stock gegen Capped Base-Stock. Parameter: bestes Gitter, exakt bewertet bis L = 6, ab L = 7 per Simulation auf Strom 101 abgestimmt und auf Strom 777 bewertet.")
    st.markdown(f"**Live: dieselbe Frage mit einem eigenen Zufallsstrom** (Lieferzeit und Zufalls-Seed in der Seitenleiste, {C.SIM_CHAINS:,} Ketten × {C.SIM_PERIODS:,} Perioden, {C.SIM_BURN} Einschwingperioden).".replace(",", "."))
    with st.spinner("Simuliert Base-Stock, Capped Base-Stock" + (" und die DP-Politik" if settings.l <= C.LIVE_DP_MAX_L else "") + " auf einem gemeinsamen Strom (wenige Sekunden) …"):
        sim = _simulate(settings.l, settings.sim_seed)
    base_mean = sim["rows"]["Capped Base-Stock"][0]
    st.dataframe(
        [{"Politik": name, "Kosten je Periode": E.de(m, 3), "Standardfehler": E.de(se, 3), "Abstand zu Capped (%)": E.de(100 * (m - base_mean) / base_mean, 2, True)}
         for name, (m, se) in sim["rows"].items()],
        hide_index=True, width="stretch",
    )
    st.caption(f"Parameter aus der Tabelle oben: Base-Stock Ziel {sim['params']['S_bs']}, Capped Base-Stock Ziel {sim['params']['S_cap']} mit Deckel {sim['params']['cap']}. "
               "Standardfehler: Streuung der Kosten über die Ketten geteilt durch die Wurzel der Kettenzahl.")
    kind, text = E.verdict_benchmark(sim)
    getattr(st, kind)(text)
    if settings.l > C.LIVE_DP_MAX_L:
        st.info("Ohne DP-Politik: bei L > 5 rechnet die App die DP nicht live. Die Lücke der Regeln zum Optimum steht für L = 1 bis 7 oben (vorgerechnet); bei L = 8 gibt es sie nur aus einer einmaligen Messung außerhalb der App (Capped Base-Stock rund +0,9 %, siehe Annahmen).")

# --- Handbeispiel --------------------------------------------------------------------------------------------------------------------------

with tab_hand:
    st.subheader("✏️ Handbeispiel: sechs Zustände, von Hand nachrechenbar")
    h = _hand()
    st.markdown(
        "**Mini-Instanz:** Lieferzeit 2, Bestellmenge 0 oder 1, Bestand höchstens 2, Nachfrage 0, 1 oder 2 mit Wahrscheinlichkeit ¼, ½, ¼, Lagerkosten 1 je Stück, Fehlmengenkosten 9 je Stück, verlorene Nachfrage. "
        "Zustand (x, p1): Bestand x und die Bestellung p1 von vor einer Periode, also 3 · 2 = **6 Zustände**."
    )
    st.markdown(
        f"**1. Erwartete Periodenkosten** je Bestand: Bestand 0: {E.de(h['ec'][0], 1)}, Bestand 1: {E.de(h['ec'][1], 1)}, Bestand 2: {E.de(h['ec'][2], 1)} "
        "(Bestand 1: ¼ · 1 Lagerkosten für ein Reststück bei Nachfrage 0 plus ¼ · 9 Fehlmengenkosten bei Nachfrage 2)."
    )
    names = [f"({x}, {p1})" for x, p1 in h["states"]]
    st.markdown("**2. Rückwärtsrechnung** über 1, 2 und 3 Perioden Restlaufzeit (Endwert 0), Wert V je Zustand (x, p1) und die beste Bestellmenge bei 3 Perioden:")
    st.dataframe(
        [{"Zustand (x, p1)": nm, "V1": E.de(float(h["values"][0].reshape(-1)[i]), 3), "V2": E.de(float(h["values"][1].reshape(-1)[i]), 3),
          "V3": E.de(float(h["values"][2].reshape(-1)[i]), 3), "beste Menge bei 3 Perioden": int(h["actions"][2].reshape(-1)[i])} for i, nm in enumerate(names)],
        hide_index=True, width="stretch",
    )
    st.markdown(
        f"Zustand (0, 0): Bestellung 0 kostet {E.de(h['costs_action'][0], 1)}, Bestellung 1 kostet {E.de(h['costs_action'][1], 1)} bei drei Perioden: **bestellen**."
    )
    st.markdown(
        f"**3. Durchschnittskosten:** „immer 1 bestellen“ hält den Bestand in {{1, 2}}; Übergänge 1 → 2 und 2 → 1 haben je Wahrscheinlichkeit ¼, die Kette bleibt also je zur Hälfte in 1 und 2: "
        f"½ · 2,5 + ½ · 1 = **{E.de(h['always_one'], 2)}**. Aufzählung aller {h['n_policies']} Politiken: bestes g = {E.de(h['best_g'], 2)}; relative Wertiteration: {E.de(h['g'], 2)}."
    )
    ge, gr = h["gradient_exact"][0], h["gradient_reinforce"][0]
    st.markdown(
        f"**4. Politikgradient:** im Zustand (0, 0) zwei Bestellmengen mit gleicher Wahrscheinlichkeit ½ (Softmax mit θ = 0); Kosten {E.de(h['costs_action'][0], 1)} und {E.de(h['costs_action'][1], 1)}. "
        f"Die Ableitung der erwarteten Kosten nach θ₀ ist ½ · (27 − 23,75) = {E.de(ge, 3, True)}; der Mittelwert der REINFORCE-Schätzung ist {E.de(gr, 3, True)}: gleich. "
        "Das Netz folgt der Gegenrichtung und bestellt häufiger die billigere Menge."
    )
    st.markdown("**Wachstum der Mini-Instanz:** Faktor Q + 1 = 2 je Periode Lieferzeit.")
    st.dataframe(
        [{"L": r["L"], "Zustände": r["states"], "Rechenaufwand je Wertiteration (Zustände · 2 Mengen · 3 Nachfragewerte)": r["terms"]} for r in E.hand_state_counts()],
        hide_index=True, width="stretch",
    )

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

with tab_limits:
    st.subheader("🚧 Wo die Annahmen enden")
    st.markdown(
        "| Annahme | Was passiert, wenn sie verletzt ist | Nachweis |\n|---|---|---|\n"
        "| Bestellgrenze Q = 8 | Mit Q = 5 dieselben Kosten, 21 · 6^(L−1) statt 21 · 9^(L−1) Zustände: die Grenze der DP verschiebt sich um etwa eine Periode Lieferzeit | Experiment „Faktor 9“ (live, L = 1 bis 5) |\n"
        "| Bestandsgrenze 20 | Mit 16 wäre die Randmasse bei L = 4 schon 2,6e-4; bei L = 8 nicht gemessen, mit mehr Randmasse zu rechnen | Test, Annahmen |\n"
        "| Ein Lernverfahren, eine Einstellung | Aussagen über „Lernen“ gelten für diese Einstellung | vorgerechnete Läufe |\n"
        "| Capped Base-Stock als Maßstab | bei L = 1 bis 7 +0,1 bis +1,6 % über der DP; bei L = 8 nach einer einmaligen Messung außerhalb der App rund +0,9 % | Annahmen |\n"
    )
    st.markdown(
        "**Lehre:** „Das Netz schlägt Capped Base-Stock“ ist weder immer wahr noch immer falsch. Am Endstand liegt das Netz bei den Lieferzeiten L = 2, 3, 4 in "
        + ", ".join(f"{E.learning_stats(Lx)['below_capped']} von {E.learning_stats(Lx)['n_seeds']}" for Lx in (2, 3, 4))
        + " Läufen unter Capped Base-Stock, bei L = 5 und 6 in "
        + f"{E.learning_stats(5)['below_capped']} von {E.learning_stats(5)['n_seeds']} und {E.learning_stats(6)['below_capped']} von {E.learning_stats(6)['n_seeds']}, bei L = 7 in "
        + f"{E.learning_stats(7)['below_capped']} von {E.learning_stats(7)['n_seeds']} (Simulation). Das Vorzeichen des Vergleichs hängt also von der Lieferzeit ab, und je nach Abgriff (Endstand oder Mittel der letzten fünf Messpunkte) ändert sich die Zählung. "
        "Eine Aussage über „Lernen“ braucht deshalb Verfahren, Einstellung, Lieferzeit, Zahl der Zufalls-Seeds und die Art des Abgriffs."
    )

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Nachfrageverteilung, Kosten und Lieferzeit. *Gesucht:* eine Politik (Bestellmenge je Zustand), die die durchschnittlichen Kosten je Periode minimiert, und die Zahl der Zustände, die ihre exakte Berechnung aufzählen muss.

| Symbol | Bedeutung |
|---|---|
| $x$ | Bestand zu Beginn der Periode, $0 \le x \le X_M = 20$ |
| $p_1, \dots, p_{L-1}$ | Bestellungen unterwegs ($p_1$ kommt in der nächsten Periode an) |
| $s = (x, p_1, \dots, p_{L-1})$ | Zustand |
| $a$ | Bestellmenge, $0 \le a \le Q = 8$ |
| $D$ | Nachfrage je Periode, Poisson mit Mittel 3 (bei 14 abgeschnitten) |
| $h, p$ | Lagerkosten 1 je Stück, Fehlmengenkosten 9 je fehlendem Stück |
| $g$ | durchschnittliche Kosten je Periode |
| $h(s)$ | relativer Wert des Zustands (Wertetabelle) |

**Kosten einer Periode** bei Bestand $x$ (verlorene Nachfrage):

$$\bar c(x) = h\,\mathbb E\big[(x-D)^+\big] + p\,\mathbb E\big[(D-x)^+\big]$$

**Übergang:** der Bestand wächst um die Ankunft $p_1$, die Pipeline rückt auf, die neue Bestellung $a$ wird ihr letztes Glied:

$$x' = \min\big((x-D)^+ + p_1,\; X_M\big), \qquad (p_1, \dots, p_{L-1}) \to (p_2, \dots, p_{L-1}, a)$$

**Optimalitätsgleichung** (durchschnittliche Kosten, Wertiteration bis zur Konstanz):

$$g + h(s) = \min_{0 \le a \le Q}\Big[\, \bar c(x) + \sum_{d} \Pr(D=d)\; h\big(s'(s, a, d)\big)\Big]$$

**Zahl der Zustände** bei Lieferzeit $L$:

$$|\mathcal S| = (X_M + 1)\,(Q+1)^{L-1}$$

**Haltbarkeit** (Altersklassen $m$, je höchstens $C$ Stück): $(C+1)^{m-1}$ Zustände bei Lieferzeit 0 und $(C+1)^{m}$ bei Lieferzeit 1.

**Zwei Stufen:** $56 \cdot 19 \cdot (Q+1)^{L_r + L_w - 2}$ Zustände bei Lieferzeit $L_r$ des Händlers und $L_w$ des Lagers.

**Politikgradient** (Kosten $G_t$ bis zum Ende, Basiswert $b(s_t)$ nur vom Zustand abhängig, Netz $\pi_\theta$):

$$\nabla_\theta J(\theta) = \mathbb E\Big[\sum_t \big(G_t - b(s_t)\big)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big]$$

**Softmax im Handbeispiel** (Kosten $c_a$, mittlere Kosten $\bar c = \sum_b \pi_b c_b$):

$$\frac{\partial}{\partial \theta_a} \sum_b \pi_b c_b = \pi_a\,(c_a - \bar c)$$

**Standardfehler** einer Differenz auf gemeinsamem Strom (Differenz $\delta_i$ je Kette, $N$ Ketten):

$$\mathrm{SE}(\bar\delta) = \frac{s_\delta}{\sqrt{N}}$$

**Literatur.** Capped Base-Stock (Minimum aus Base-Stock-Bestellung und Deckel): Definition nach van Jaarsveld und Arts (arXiv 2101.07519, Abschnitt 7.2); Xin (2021, Operations Research 69(1)) zeigt nach dem Abstract (Volltext nicht gelesen), dass es Capped-Base-Stock-Parameter gibt, mit denen die Regel für wachsende Lieferzeit asymptotisch optimal ist. Zum gelernten Netz gegen Heuristiken: Gijsbrechts u. a., SSRN-Arbeitsfassung (Juli 2021), nicht die gedruckte Fassung.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html)."
)

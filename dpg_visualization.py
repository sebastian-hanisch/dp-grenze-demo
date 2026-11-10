"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import warnings

import numpy as np
import plotly.graph_objects as go

import dpg_constants as C
import dpg_counts as N
import dpg_stats as ST


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_state_growth(selected_L: int):
    """Zustandszahl über der Lieferzeit (logarithmische Achse) für Q = 8 und Q = 5, mit der Speichermarke 1 GB je Tabelle."""
    Ls = list(range(1, 10))
    fig = go.Figure()
    for Q, color, name in ((C.Q_MAX, C.COLOR_BASE, "Bestellgrenze Q = 8"), (5, C.COLOR_DP, "Bestellgrenze Q = 5 (reicht für dieselben Kosten)")):
        fig.add_trace(go.Scatter(x=Ls, y=[N.lost_sales_states(L, Q=Q) for L in Ls], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 hovertemplate="L = %{x}: %{y:,.0f} Zustände<extra></extra>"))
    fig.add_hline(y=1e9 / C.BYTES_PER_VALUE, line_dash="dash", line_color="gray", annotation_text="1 GB je Wertetabelle", annotation_position="bottom right")
    fig.add_vline(x=selected_L, line_color=C.COLOR_CAPPED, line_width=2, annotation_text=f"L = {selected_L}", annotation_position="top")
    fig.update_xaxes(title_text="Lieferzeit L (Perioden)", dtick=1)
    fig.update_yaxes(title_text="Zustände (logarithmisch)", type="log")
    return _base(fig)


def build_reach(refs: dict):
    """Zustände der Tabelle gegen die Zustände, die unter der DP-Politik überhaupt vorkommen (logarithmische Achse), L = 1 bis 7."""
    Ls = [L for L in range(1, 8) if refs[f"L{L}"].get("reach") is not None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ls, y=[refs[f"L{L}"]["states"] for L in Ls], mode="lines+markers", name="alle Zustände der Tabelle", line=dict(color=C.COLOR_BASE, width=3)))
    fig.add_trace(go.Scatter(x=Ls, y=[refs[f"L{L}"]["reach"] for L in Ls], mode="lines+markers", name="unter der DP-Politik erreichbar",
                             line=dict(color=C.COLOR_DP, width=3)))
    fig.update_xaxes(title_text="Lieferzeit L (Perioden)", dtick=1)
    fig.update_yaxes(title_text="Zustände (logarithmisch)", type="log")
    return _base(fig)


Y_RANGE = (-1.0, 12.0)


def build_learning(L: int, view: str):
    """Lernkurven: Lücke in % über den simulierten Perioden; Band Minimum bis Maximum, Median, einzelne Läufe, Linien der Regeln.

    Die Anfangswerte (100 bis 380 %) liegen außerhalb der festen Achse (−1 bis 12 %)."""
    refs = ST.load_refs()
    updates, seeds, mat = ST.run_gaps(L, view=view)
    periods = [u * C.PERIODS_PER_STEP for u in updates]
    ref = refs[f"L{L}"]
    base = ST.reference_cost(ref)
    fig = go.Figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # Spalten ohne Wert (die ersten vier Messpunkte beim Mittel) sind ganz nan
        lo, hi, med = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0), np.nanmedian(mat, axis=0)
    ok = [i for i in range(len(periods)) if not np.isnan(med[i])]
    xs = [periods[i] for i in ok]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[hi[i] for i in ok] + [lo[i] for i in ok][::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                             hoverinfo="skip", name="Minimum bis Maximum über die Zufalls-Seeds"))
    for s, row in zip(seeds, mat):
        fig.add_trace(go.Scatter(x=[periods[i] for i in ok], y=[row[i] for i in ok], mode="lines", line=dict(color=C.COLOR_RUN, width=1), showlegend=False,
                                 hovertemplate=f"Zufalls-Seed {s}: %{{y:.2f}} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[med[i] for i in ok], mode="lines+markers", line=dict(color=C.COLOR_NET, width=3), name="Median über die Zufalls-Seeds",
                             hovertemplate="Median: %{y:.2f} %<extra></extra>"))
    cap_gap = float(ST.gap_percent(ref["g_cap"], base))
    bs_gap = float(ST.gap_percent(ref["g_bs"], base))
    if ref.get("g_dp") is not None:
        fig.add_hline(y=0, line_color=C.COLOR_DP, line_width=2, annotation_text="DP (Optimum)", annotation_position="bottom right")
    fig.add_hline(y=cap_gap, line_color=C.COLOR_CAPPED, line_dash="dash", line_width=2,
                  annotation_text="Capped Base-Stock" + (" (Maßstab)" if ref.get("g_dp") is None else ""), annotation_position="top right")
    fig.add_hline(y=bs_gap, line_color=C.COLOR_BASE, line_dash="dot", line_width=2, annotation_text="Base-Stock", annotation_position="top right")
    fig.update_xaxes(title_text="simulierte Perioden", range=[0, C.UPDATES * C.PERIODS_PER_STEP * 1.02])
    fig.update_yaxes(title_text=("Lücke zur DP (%)" if ref.get("g_dp") is not None else "Abstand zu Capped Base-Stock (%)"), range=list(Y_RANGE))
    return _base(fig, 420)


def build_final_gaps(L: int, stats: dict):
    """Endstand je Zufalls-Seed (Abgriff und Mittel der letzten fünf Messpunkte) gegen die Linie von Capped Base-Stock."""
    seeds = stats["seeds"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Seed {s}" for s in seeds], y=stats["final"], name="Abgriff bei Update 3.000", marker_color=C.COLOR_NET))
    fig.add_trace(go.Bar(x=[f"Seed {s}" for s in seeds], y=stats["mean5_final"], name="Mittel der letzten fünf Messpunkte", marker_color="rgba(76,120,168,0.45)"))
    fig.add_hline(y=stats["capped_gap"], line_color=C.COLOR_CAPPED, line_dash="dash", line_width=2, annotation_text="Capped Base-Stock", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=("Lücke zur DP (%)" if "z" not in stats else "Abstand zu Capped (%)"))
    return _base(fig, 330)


def build_benchmark_gaps(refs: dict):
    """Lücke von Capped Base-Stock und Base-Stock je Lieferzeit gegen die DP (L = 1 bis 7) bzw. Base-Stock gegen Capped (L = 8)."""
    Ls = list(range(1, 9))
    cap, bs = [], []
    for L in Ls:
        ref = refs[f"L{L}"]
        base = ST.reference_cost(ref)
        cap.append(float(ST.gap_percent(ref["g_cap"], base)))
        bs.append(float(ST.gap_percent(ref["g_bs"], base)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=Ls, y=cap, name="Capped Base-Stock", marker_color=C.COLOR_CAPPED))
    fig.add_trace(go.Bar(x=Ls, y=bs, name="Base-Stock", marker_color=C.COLOR_BASE))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Lieferzeit L (Perioden)", dtick=1)
    fig.update_yaxes(title_text="Lücke zur DP (%); bei L = 8 zu Capped")
    return _base(fig, 330)

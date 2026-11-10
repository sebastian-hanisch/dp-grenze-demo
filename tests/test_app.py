"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ---- Inhalt der App ---------------------------------------------------------------------------------------------------------------------------

TAB_LABELS = ["📊 Zustandszahl", "⏱️ Rechenzeit und Speicher", "🧠 Gelerntes Netz", "📏 Maßstab", "✏️ Handbeispiel", "🚧 Wo die Annahmen enden"]


def _run_with(**state) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    for key, value in state.items():
        at.session_state[key] = value
    at.run(timeout=TIMEOUT)
    assert not at.exception, (state, [str(e) for e in at.exception])
    return at


def test_every_tab_and_expander_is_present():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == TAB_LABELS
    labels = [e.label for e in at.expander]
    assert "⚠️ Annahmen, Grenzen und bekannte Probleme" in labels and "📐 Mathematische Formulierung" in labels


def test_every_lead_time_renders_with_metrics_and_verdicts():
    """L = 1 bis 8 mit der Ansicht „Abgriff“ und „Mittel der letzten fünf Messpunkte“: keine Ausnahme, vier Kennzahlen, die drei Verdikte der Kernfrage."""
    for L in range(1, 9):
        at = _run_with(l_slider=L, view_radio=L % 2)
        shown = [m.label for m in at.metric]
        assert shown[:4] == ["Zustände", "Wertetabelle (eine)", "Rechenzeit der DP", shown[3]], L
        texts = [e.value for kind in ("info", "success", "warning") for e in getattr(at, kind)]
        assert any(f"L = {L}" in t and "Zustände" in t for t in texts), L          # Verdikt zum Wachstum
        assert any(t.startswith(f"Bei L = {L}") or f"bei L = {L}" in t for t in texts), L


def test_lead_time_eight_shows_the_dp_as_not_computed_and_one_has_no_runs():
    at = _run_with(l_slider=8)
    assert at.metric[2].value == "nicht gerechnet"
    assert any("die DP wurde hier nicht gerechnet" in e.value for e in at.warning)
    at = _run_with(l_slider=1)
    assert at.metric[3].value == "keine Lernläufe"
    assert any("keine Lernläufe" in e.value for e in at.info)


def test_simulation_seed_changes_the_live_benchmark():
    a = _run_with(l_slider=2, sim_seed_input=1)
    b = _run_with(l_slider=2, sim_seed_input=2)
    rows = lambda at: [str(e.value) for e in at.dataframe]  # noqa: E731
    assert rows(a) != rows(b)


def test_hand_example_numbers_are_on_the_page():
    at = _run_with()
    text = " ".join(m.value for m in at.markdown)
    assert "6 Zustände" in text and "1,75" in text and "1,625" in text and "20,5" in text

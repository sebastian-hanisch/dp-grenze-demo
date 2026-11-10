"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink mit Begrenzung."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

import dpg_constants as C
import dpg_evaluation as ev
import dpg_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for preset in C.PRESETS.values():
        assert C.L_MIN <= preset["l"] <= C.L_MAX and preset["view"] in (0, 1) and 0 <= preset["sim_seed"] <= C.SIM_SEED_MAX


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(l=first["l"], view=C.VIEWS[first["view"]], sim_seed=first["sim_seed"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_name_the_lead_time_they_set():
    for name, preset in C.PRESETS.items():
        assert f"L = {preset['l']}" in name


def test_permalink_is_clipped_and_invalid_values_are_ignored():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["l"] = "99"                  # über der Obergrenze: auf 8 begrenzt
    at.query_params["ansicht"] = "7"             # keine erlaubte Auswahl: ignoriert, Standard bleibt
    at.query_params["strom"] = "abc"             # keine Zahl: ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["l_slider"] == C.L_MAX
    assert at.session_state["view_radio"] == 0 and at.session_state["sim_seed_input"] == C.DEFAULT_SIM_SEED


def test_permalink_reflects_the_settings_in_the_address():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.slider(key="l_slider").set_value(3).run()
    assert at.query_params["l"] == ["3"] or at.query_params["l"] == "3"

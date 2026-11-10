"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""


import dpg_evaluation as E


def test_de_and_seconds_text():
    assert E.de(0.5, 1, True) == "+0,5" and E.de(-0.05, 2, True) == "−0,05" and E.de(4.7493, 4) == "4,7493"
    assert E.seconds_text(0.03) == "weniger als 0,05 s"
    assert E.seconds_text(0.8) == "rund 0,8 s"
    assert E.seconds_text(48.9) == "rund 49 s"


def test_verdict_growth_names_states_size_and_factor_and_warns_from_half_a_gigabyte():
    kind, text = E.verdict_growth(5, 137781)
    assert kind == "info" and "L = 5" in text and "**137.781 Zustände**" in text and "(1,1 MB je Tabelle)" in text and "Q + 1 = 9" in text
    assert E.verdict_growth(8, 100442349)[0] == "warning"
    assert E.verdict_growth(8, 62_500_000)[0] == "warning"          # genau 0,5 GB (8 Byte je Zustand): inklusive Schwelle
    assert E.verdict_growth(8, 62_499_999)[0] == "info"
    assert "Q + 1 = 6" in E.verdict_growth(5, 27216, Q=5)[1]


def test_verdict_dp_time_for_the_three_sources():
    kind, text = E.verdict_dp_time(2, 189, 0.03, "live")
    assert kind == "info" and "bei L = 2 **weniger als 0,05 s**" in text and "in dieser Sitzung gerechnet" in text
    kind, text = E.verdict_dp_time(7, 11160261, 48.9, "vorgerechnet")
    assert kind == "info" and "**rund 49 s**" in text and "vorgerechnet auf dem Messrechner" in text
    kind, text = E.verdict_dp_time(8, 100442349, None, "keine")
    assert kind == "warning" and "100.442.349 Zustände" in text and "0,8 GB" in text and "Speicher" in text
    assert "unmöglich ist die Rechnung nicht" in text and "522 s, 5,95 GB" in text and "Streamlit Cloud sind nicht gemessen" in text
    assert "der Speicherbedarf ist der Grund" not in text
    assert "522 s" not in E.verdict_dp_time(9, 903981141, None, "keine")[1]          # die Messung gilt nur für L = 8
    assert E.verdict_dp_time(8, 100442349, 5.0, "keine")[0] == "warning"      # ohne Quelle „keine“ zählt die Zeit nicht


def _stats(**kw):
    base = dict(n_seeds=10, final_median=0.53, final_min=0.2, final_max=4.71, capped_gap=0.69, below_capped=6)
    return {**base, **kw}


def test_verdict_learning_with_dp_reference():
    kind, text = E.verdict_learning(_stats(), 2)
    assert kind == "success"
    assert "Bei L = 2" in text and "bei +0,5 % über der DP" in text and "Bereich +0,2 bis +4,7 %" in text
    assert "Capped Base-Stock bei +0,7 %" in text and "6 von 10 Läufen unter Capped Base-Stock" in text
    assert E.verdict_learning(_stats(below_capped=5), 2)[0] == "info"          # genau die Hälfte ist keine Mehrheit
    assert E.verdict_learning(_stats(below_capped=6), 2)[0] == "success"
    assert E.verdict_learning(_stats(below_capped=0), 5)[0] == "info"


def test_verdict_learning_against_capped_in_standard_errors():
    s = _stats(n_seeds=5, z=[10.1, -0.5, -1.2, -0.3, 2.2], clearly_better=0, clearly_worse=2, better_point=3)
    kind, text = E.verdict_learning(s, 8)
    assert kind == "warning"
    assert "rechnet die App die DP nicht" in text and "keiner erkennbar darunter (0 von 5 mit Abstand über 2 Standardfehlern, Fehler von Netz und Regel zusammengesetzt;" in text
    assert "gepaarte Fehler auf demselben Strom ist kleiner" in text
    assert "2 liegen erkennbar darüber" in text and "3 von 5 haben als Punktschätzung geringere Kosten" in text
    assert E.verdict_learning({**s, "clearly_worse": 0}, 8)[0] == "info"
    assert E.verdict_learning({**s, "clearly_better": 1, "clearly_worse": 2}, 8)[0] == "info"
    kind, text = E.verdict_learning({**s, "clearly_better": 1, "clearly_worse": 2}, 8)
    assert "1 erkennbar darunter" in text and "keiner erkennbar" not in text           # Regression: der Text sagte „keiner“, auch wenn ein Lauf erkennbar besser war


def test_verdict_learning_without_runs():
    kind, text = E.verdict_learning(None, 1)
    assert kind == "info" and "keine Lernläufe" in text and "21 Werte" in text


def test_verdict_benchmark_names_difference_and_both_standard_errors():
    kind, text = E.verdict_benchmark(dict(diff=0.128, se_paired=0.001, se_unpaired=0.005))
    assert kind == "info"
    assert "0,128 je Periode mehr als Capped Base-Stock" in text and "Standardfehler 0,001 bei gemeinsamem Strom, 0,005 bei getrennten Strömen" in text
    assert "5,0-mal kleiner" in text
    assert "0,050 je Periode weniger" in E.verdict_benchmark(dict(diff=-0.05, se_paired=0.01, se_unpaired=0.02))[1]
    assert "mal kleiner" not in E.verdict_benchmark(dict(diff=0.1, se_paired=0.0, se_unpaired=0.0))[1]        # Standardfehler null: kein Verhältnis


def test_verdict_order_cap_same_cost_and_cap_that_binds():
    kind, text = E.verdict_order_cap(4.7493, 4.7493, 126, 189)
    assert kind == "success" and "dieselben Kosten (4,7493)" in text and "126 statt 189 Zustände" in text
    kind, text = E.verdict_order_cap(4.8, 4.7493, 100, 189)
    assert kind == "warning" and "von 4,7493 auf 4,8000" in text
    assert E.verdict_order_cap(5.0 + 1e-10, 5.0, 1, 2)[0] == "success"       # unter 1e-9 gilt als gleich
    assert E.verdict_order_cap(5.0 + 1e-8, 5.0, 1, 2)[0] == "warning"

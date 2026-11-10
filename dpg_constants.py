"""Konstanten: Regler-Grenzen, feste Annahmen des Beispiels, Farben, Lernläufe und Presets.

Das Beispiel ist in allen Demos der Reihe „Bestandspolitiken“ dasselbe: ein Artikel, Poisson-Nachfrage mit Mittel 3 je Periode,
h = 1 je Stück und Periode Lagerkosten, p = 9 je fehlendem Stück (verlorene Nachfrage, kein Rückstand), Bestellkosten 0,
Bestellung höchstens Q = 8 Stück je Periode, Lieferzeit L Perioden. Gerechnet wird mit durchschnittlichen Kosten je Periode.
"""

MEAN, HOLD, PENALTY = 3.0, 1.0, 9.0          # Nachfragemittel, Lagerkosten je Stück und Periode, Fehlmengenkosten je Stück
Q_MAX = 8                                    # Bestellgrenze: höchstens so viele Stück je Bestellung
XM = 20                                      # Bestandsgrenze: Bestand über 20 wird abgeschnitten (Randmasse bis L = 7 unter 1e-4, siehe README)
DMAX = 14                                    # Nachfrage wird bei 14 abgeschnitten (Rest der Wahrscheinlichkeit liegt auf 14)
EDGE_MASS_LIMIT = 1e-4                       # gedruckte Randmasse (Anteil der Zeit bei Bestand = XM) muss darunter liegen
REACH_THRESHOLD = 1e-9                       # „erreichbar“: Zustand mit Anteil der Zeit über dieser Schwelle unter der DP-Politik

L_MIN, L_MAX, DEFAULT_L = 1, 8, 2            # Lieferzeit in Perioden
LIVE_DP_MAX_L = 5                            # bis hierher rechnet die App die DP selbst (mit Spinner ab L = 5)
LEARN_L = tuple(range(2, 9))                 # Lieferzeiten mit vorgerechneten Lernläufen
DP_EXACT_MAX_L = 7                           # darüber ist die DP weder in der App noch in den vorgerechneten Daten (Rechenzeit und Speicher, siehe README)
BYTES_PER_VALUE = 8                          # eine Tabelle der Werte: 8 Byte je Zustand

SIM_CHAINS, SIM_PERIODS, SIM_BURN = 2000, 1500, 300   # Simulation: Ketten, Perioden, Einschwingperioden (nicht gezählt)
SIM_TUNE_SEED, DEFAULT_SIM_SEED = 101, 777            # Parameter der Regeln auf Strom 101 abgestimmt, bewertet auf dem Strom der Seitenleiste
SIM_SEED_MAX = 999999
SE_BAND = 2.0                                         # „erkennbar“ heißt: Differenz größer als 2 Standardfehler

# Lernverfahren (Politikgradient mit gelerntem Basiswert, Adam), fest für alle vorgerechneten Läufe
HIDDEN, CHAINS_PER_UPDATE, PERIODS_PER_UPDATE = 32, 256, 100
GAMMA, ENTROPY, LEARNING_RATE = 0.98, 0.01, 3e-3
UPDATES = 3000
PERIODS_PER_STEP = CHAINS_PER_UPDATE * PERIODS_PER_UPDATE   # 25.600 simulierte Perioden je Update
LATE_UPDATE = 1000                                          # „späte“ Messpunkte: ab Update 1000
GAP_LIMIT_PERCENT = 2.0                                     # „nahe an der DP“: Lücke unter 2 %

VIEW_PICK, VIEW_MEAN5 = "Abgriff (ein Messpunkt)", "Mittel der letzten fünf Messpunkte"
VIEWS = (VIEW_PICK, VIEW_MEAN5)

COLOR_DP = "#2F6B65"
COLOR_CAPPED = "#d98324"
COLOR_BASE = "#9a3b3b"
COLOR_NET = "#4c78a8"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_RUN = "rgba(76,120,168,0.30)"

PRESETS = {
    "Kleiner Fall (L = 2)": {"l": 2, "view": 0, "sim_seed": DEFAULT_SIM_SEED},
    "Netz meist unter Capped (L = 3)": {"l": 3, "view": 0, "sim_seed": DEFAULT_SIM_SEED},
    "Netz nicht unter Capped (L = 5)": {"l": 5, "view": 0, "sim_seed": DEFAULT_SIM_SEED},
    "Ohne DP in der App (L = 8)": {"l": 8, "view": 0, "sim_seed": DEFAULT_SIM_SEED},
}
PRESET_HELP = {
    "Kleiner Fall (L = 2)": "Lieferzeit 2: 189 Zustände, die DP braucht Bruchteile einer Sekunde; das gelernte Netz liegt im Median der Endstände 0,5 % über der DP.",
    "Netz meist unter Capped (L = 3)": "Lieferzeit 3: Am Endstand liegt das Netz in 8 von 10 Läufen unter Capped Base-Stock (der Regel mit gedeckelter Bestellmenge).",
    "Netz nicht unter Capped (L = 5)": "Lieferzeit 5: 137.781 Zustände. Kein Lauf von 5 liegt am Endstand unter Capped Base-Stock (mit nur 5 Läufen ein grobes Bild).",
    "Ohne DP in der App (L = 8)": "Lieferzeit 8: rund 100 Millionen Zustände, die Wertetabelle braucht etwa 0,8 GB. Die App rechnet die DP hier nicht und vergleicht mit Capped Base-Stock (einmalige Messung außerhalb der App: Capped rund +0,9 % über dem Optimum).",
}

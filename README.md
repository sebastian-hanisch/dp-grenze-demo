# Wo die exakte Rechnung aufgibt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-dp-grenze-demo.streamlit.app/)**

Sechste Demo der Linie **Bestandspolitiken** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Dasselbe Beispiel wie in der ganzen Linie: ein Artikel, Poisson-Nachfrage mit Mittel 3 je Periode, **verlorene Nachfrage**, Lagerkosten 1 und Fehlmengenkosten 9 je Stück, Bestellung höchstens 8 Stück, Lieferzeit L Perioden. Die Vorgänger-Demos bestimmen die beste Bestellregel mit der exakten Rückwärtsrechnung (dynamische Programmierung, DP), solange sich der Zustand aufzählen lässt. Diese Demo misst, **wo das aufhört**, und was ein per Politikgradient gelerntes Netz an dieser Stelle leistet. Die Kante zu den Vorgängern ist ein `fix` mit negativem Befund: die Schwäche (die Tabelle über alle Zustände) wird nicht behoben, sondern vermessen, samt der Alternative.

## Warum dieses Problem

**Kernfrage: Wo gibt die exakte Rechnung auf, und was bleibt dann?** Bei Lieferzeit gehören die Bestellungen unterwegs zum Zustand. Die Wertetabelle der DP wächst bei verlorener Nachfrage um den Faktor Bestellgrenze + 1 = 9 je Periode Lieferzeit. Die **Rechenzeit** ist auf einem Arbeitsplatzrechner dabei kein früher Anschlag: L = 5 (137.781 Zustände) rechnet in rund einer Sekunde (Messrechner), L = 7 (11.160.261 Zustände) in rund einer Minute. Bei L = 8 sind es 100.442.349 Zustände, rund 0,8 GB je Wertetabelle (die Rechnung hält mehrere). Die App rechnet L = 8 nicht; eine einmalige Messung außerhalb der App (Prüfer, Rechner mit 32 GB) rechnete die DP dort in 522 s bei 5,95 GB Spitzenspeicher: möglich, aber aufwendig, die Grenzen von Streamlit Cloud sind nicht gemessen. Ein per **Politikgradient** gelerntes Netz braucht dafür Minuten und liegt im Median der Endstände 0,5 bis rund 1 % über den DP-Kosten (L = 2 bis 7; bei L = 7 mit Simulationsversatz, siehe Ehrliche Grenzen), erreicht aber nicht verlässlich die Güte von **Capped Base-Stock** (bestelle bis zu einem Zielbestand, höchstens einen Deckel je Periode): ob es ihn schlägt, hängt von der Lieferzeit, vom Zufalls-Seed und vom Zeitpunkt des Abgriffs ab. Gilt für ein Verfahren, eine Einstellung und dieses Beispiel (siehe Ehrliche Grenzen).

## Modell

Zustand (x, p1, …, p_{L−1}): Bestand x (0 bis 20) und die Bestellungen unterwegs (je 0 bis 8). Je Periode: Entscheidung a, Nachfrage D (Poisson, Mittel 3, bei 14 abgeschnitten), Kosten h·(x−D)⁺ + p·(D−x)⁺, dann x′ = min((x−D)⁺ + p1, 20) und die Pipeline rückt auf. Zielgröße: durchschnittliche Kosten je Periode. Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App. Zustandszahl: (20 + 1)·(8 + 1)^(L−1). Für **Haltbarkeit** (verderbliche Ware, Bestand je Restlebensdauer) und **zwei Stufen in Reihe** wird nur gezählt, nicht gerechnet.

## Methodik

- **DP (`dpg_algorithm.py`):** relative Wertiteration. Die Nachfrage wirkt nur auf (Bestand, p1), der Rest der Pipeline schiebt sich: je Bestellmenge ein dichtes Matrixprodukt statt einer Übergangsmatrix. Dieselbe Struktur wertet feste Regeln exakt aus (stationäre Verteilung).
- **Regeln:** Base-Stock (bestelle bis zur Bestandsposition S) und Capped Base-Stock (zusätzlich höchstens ein Deckel je Periode); Parameter per Gitter, bis L = 6 exakt bewertet, bei L = 7 und 8 per Simulation (abgestimmt auf Zufallsstrom 101, bewertet auf Strom 777).
- **Simulation mit gemeinsamen Zufallsströmen (`dpg_sim.py`):** alle verglichenen Politiken sehen denselben Nachfragestrom; die Streuung einer Differenz wird klein.
- **Netz (`dpg_learn.py`):** Strategie-Netz und Kritiker mit je 32 tanh-Einheiten, Politikgradient (REINFORCE) mit gelerntem Basiswert, Adam. Die 50 Lernläufe sind **vorgerechnet** (`dpg_data/curves.json`); die App liest sie nur.
- **Live in der App:** Zustandszahlen, DP bis L = 5 (bei L = 5 mit Spinner), exakte Bewertung, Bestellgrenzen-Experiment, Simulation (2.000 Ketten × 1.500 Perioden), Handbeispiel. **Vorgerechnet:** DP bei L = 6 und 7, Regeln bei L = 7 und 8, alle Lernläufe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Poisson-Nachfrage Mittel 3, h = 1, p = 9, verlorene Nachfrage, Bestellgrenze 8, Bestandsgrenze 20, durchschnittliche Kosten je Periode; Lücke in % der DP-Kosten (bei L = 8: gegen Capped Base-Stock). Spalte Test: Nachweis in `tests/test_claims.py` (K-Nummer, Beschreibung im Docstring des Tests).

| Frage | Befund | Test |
|---|---|---|
| Zustandszahl bei verlorener Nachfrage (L = 1 bis 9) | 21 / 189 / 1.701 / 15.309 / 137.781 / 1.240.029 / 11.160.261 / 100.442.349 / 903.981.141, je Periode mal 9; eine Wertetabelle (8 Byte je Zustand) 1,1 MB (L = 5), 9,9 MB, 89,3 MB, 0,8 GB (L = 8), 7,2 GB (L = 9) | K01, K02 |
| Zustandszahl bei Haltbarkeit und zwei Stufen | Haltbarkeit (Lieferzeit 0, m = 2 bis 8): 9 bis 4.782.969; zwei Stufen: 1.064 / 9.576 / 86.184 / 775.656 / 6.980.904 (nur Formel, nicht gerechnet) | K03 |
| DP-Kosten je Periode | 4,2521 / 4,7493 / 5,0784 / 5,3050 / 5,4682 (L = 1 bis 5, im Test gerechnet), 5,5940 und 5,6978 (L = 6, 7, Daten) | K04 |
| Bestandsgrenze 20 reicht bis L = 7 (Marke 1e-4) | Randmasse unter 1e-20 (L = 1 bis 4), 2,1e-6 (L = 5), 2,6e-5 (L = 6), 9,1e-5 (L = 7), bis L = 7 unter 1e-4; mit Grenze 16 bei L = 4 schon 2,6e-4; bei L = 8 nicht gemessen, mit mehr Randmasse zu rechnen | K05 |
| Der Faktor 9 hängt an der Bestellgrenze | bei L = 1 bis 5 bestellt die DP höchstens 5 Stück; Q = 5 gibt dieselben Kosten (Abweichung unter 1e-9) bei 21·6^(L−1) Zuständen (27.216 statt 137.781 bei L = 5); mit Q = 4 steigen die Kosten | K06 |
| Erreichbare Zustände unter der DP-Politik | 10 / 60 / 313 / 1.188 / 4.785 (L = 1 bis 5), 15.426 und 41.428 (L = 6, 7); bei L = 7 sind das 0,4 % der Tabelle; Wachstum je Periode Faktor 2,7 bis 6,0 statt 9; Schwelle 1e-9: bei L = 5 ergeben die Schwellen 1e-6 / 1e-9 / 1e-11 2.633 / 4.785 / 5.451 Zustände, die Zahlen sind also Größenordnungen (Nachrechnung des Prüfers) | K07 |
| Rechenzeit der DP | DP-Zeit steigt mit L: rund eine Sekunde bei L = 5, einige Sekunden bei L = 6, rund eine Minute bei L = 7 (Messrechner, Größenordnung); bei L = 8 in der App nicht gerechnet (rund 0,8 GB je Wertetabelle); einmalige Messung außerhalb der App (Prüfer, Rechner mit 32 GB): 522 s, 5,95 GB Spitzenspeicher | K08 |
| Lücke der einfachen Regeln zur DP (L = 1 bis 7) | Capped Base-Stock +0,10 / +0,69 / +1,63 / +0,82 / +0,36 / +0,37 / +0,39 %; Base-Stock +1,75 / +3,35 / +5,03 / +6,44 / +7,63 / +8,94 / +9,90 % (L = 7: Simulation auf Strom 777, Standardfehler rund 0,1 %; auf einem unabhängigen Strom mit 40.000 Ketten +0,62 % und +10,18 %, Messung des Prüfers, nicht in den Daten) | K09, K10 |
| Regeln bei L = 8 | Base-Stock +9,95 % gegen Capped; gepaarte Differenz +0,579 ± 0,003 je Periode, Standardfehler je Wert 0,0056 bis 0,0058 (L = 7, 8), der Standardfehler der gepaarten Differenz auf gemeinsamem Zufallsstrom ist weniger als halb so groß wie der aus beiden Einzelfehlern zusammengesetzte | K11 |
| Lernläufe | 50 Läufe: 10 Zufalls-Seeds für L = 2 bis 4, 5 für L = 5 bis 8, je 3.000 Updates à 25.600 Perioden | K12 |
| Wie nah das Netz an die DP kommt | Median der Endstände +0,53 / +0,67 / +1,00 / +0,77 / +0,62 / +0,86 % (L = 2 bis 7); 39 von 40 Läufen (L = 2 bis 6) unter 2 %; größter Endstand +4,71 % (L = 2); der Wert für L = 7 ist eine Simulation mit Versatz nach unten (bis rund 0,2 Punkte, eher rund +1,1 %) | K13 |
| Schlägt das Netz Capped Base-Stock? | am Endstand in 6 / 8 / 4 von 10 Läufen (L = 2, 3, 4; zusammen 18 von 30), in 0 von 5 (L = 5 und 6), in 1 von 5 (L = 7, Simulation); gemessen am Mittel der letzten fünf Messpunkte 4 / 10 / 2 von 10 und 0 von 5; Punktzählung ohne Standardfehler, und das wechselnde Vorzeichen kommt auch von Capped selbst (Lücke zur DP bei L = 3 mit +1,63 % am größten) | K14 |
| Streuung | Standardabweichung des Endstands zwischen den Zufalls-Seeds 0,32 bis 0,61 Punkte (L = 3 bis 6; 1,36 bei L = 2, 0,83 bei L = 7); innerhalb eines Laufs ab Update 1.000 im Median 0,37 bis 0,81 Punkte (L = 2 bis 6; 1,09 bei L = 7) | K15 |
| Anteil guter Messpunkte | ab Update 1.000 liegen 96 / 93 / 94 / 89 / 92 / 56 % der Messpunkte unter 2 % Lücke (L = 2 bis 7); erster Messpunkt unter 2 % im Median bei 1,3e7 bis 3,8e7 Perioden (Raster der Messpunkte, nicht der genaue Zeitpunkt); L = 7 mit nur 25 Messpunkten und Simulationsversatz nach unten | K16 |
| Nie unter dem Optimum | kleinste Differenz (Kosten des Netzes minus DP) über alle exakt bewerteten Messpunkte: +8e-4, nie negativ | K17 |
| L = 8: Netz gegen Capped | Endstände gegen Capped −0,17 / −0,08 / −0,05 / +0,32 / +1,48 %, in Standardfehlern −1,2 / −0,5 / −0,3 / +2,2 / +10,1 (Fehler von Netz und Regel zusammengesetzt, ohne Paarung): mit diesen Fehlern ist kein Lauf erkennbar besser, zwei erkennbar schlechter; mit dem gepaarten Fehler auf demselben Strom (rund 0,0023 bis 0,0024 statt 0,0085, Messung des Prüfers, nicht in den Daten) wäre der beste Lauf mit z = −4,2 erkennbar besser, die beiden anderen mit −1,8 und −1,2 nicht | K18 |
| DP bei L = 8 (einmalig außerhalb der App) | Messung des Prüfers: g = 5,782247, 522 s, 5,95 GB Spitzenspeicher, Rechner mit 32 GB; Capped Base-Stock liegt darüber um +0,67 % (Strom 777, aus den Daten gerechnet) bzw. +0,92 % (Strom mit 40.000 Ketten, Messung des Prüfers) | K24 |
| Laufzeit | je Lauf im Median 172 bis 219 s je Lieferzeit (L = 2 bis 8), Einzelläufe 147 bis 483 s (Messrechner, andere Prozesse liefen); bei L = 5 braucht die DP weniger als ein Fünfzigstel davon | K19 |
| Handbeispiel | 6 Zustände; erwartete Periodenkosten 9 / 2,5 / 1; V3(0,0) = 20,5; g = 1,75 (alle 64 Politiken aufgezählt: gleich); Ableitung 13/8 = 1,625 | K20 |
| Hilfetexte der Presets | nennen nur Zahlen aus den Zeilen oben | K21 |
| Größe des Repos | unter 3 MB, die Daten unter 100 KB | K22 |
| Annahmen-Text | seine Zahlen stimmen mit den Daten überein | K23 |

**Lehre.** „Das Netz schlägt Capped Base-Stock“ ist weder immer wahr noch immer falsch: das Vorzeichen des Vergleichs wechselt mit der Lieferzeit (6 von 10 bei L = 2, 0 von 5 bei L = 5), und die Zählung hängt davon ab, ob man den Endstand oder das Mittel der letzten fünf Messpunkte nimmt (8 von 10 gegen 10 von 10 bei L = 3). Das wechselnde Vorzeichen kommt auch von Capped selbst (bei L = 3 hat es seine größte Lücke zur DP), und 0 von 5 ist mit nur 5 Läufen vereinbar mit einem Anteil bis rund 45 % (95-%-Schranke); die Lehre beschreibt, sie belegt keine Ursache. Eine Aussage über Lernen braucht Verfahren, Einstellung, Lieferzeit, Zahl der Zufalls-Seeds und die Art des Abgriffs.

## Ehrliche Grenzen

- **Modell.** Ein Artikel, Poisson-Nachfrage mit Mittel 3 je Periode (bei 14 abgeschnitten), Lagerkosten h = 1 je Stück und Periode, Fehlmengenkosten p = 9 je fehlendem Stück, **verlorene Nachfrage** (kein Rückstand), Bestellkosten 0, Bestellgrenze Q = 8, Lieferzeit L Perioden. Gerechnet wird mit durchschnittlichen Kosten je Periode. Alle Aussagen gelten für dieses Beispiel, nicht für andere Nachfragemittel oder Kostenverhältnisse.
- **Bestandsgrenze 20.** Der Bestand wird bei 20 abgeschnitten. Der Anteil der Zeit am Rand (Bestand = 20) unter der DP-Politik liegt bei L = 1 bis 4 unter 1e-20, bei L = 5 bei 2,1e-6, bei L = 6 bei 2,6e-5 und bei L = 7 bei 9,1e-5, also bis L = 7 unter der gesetzten Marke 1e-4 (von L = 5 bis 7 wuchs er je Periode um den Faktor 3,5 bis 12). Mit der Grenze 16 läge er schon bei L = 4 mit 2,6e-4 über dieser Marke. Für L = 8 wurde die Randmasse nicht gemessen, auch nicht bei der einmaligen DP-Messung (siehe unten); dort ist mit mehr Randmasse zu rechnen.
- **Bestellgrenze Q = 8 ist größer als nötig.** Bei L = 1 bis 5 bestellt die DP-Politik in allen Zuständen, die mit mehr als 1e-9 Anteil der Zeit vorkommen, höchstens 5 Stück. Mit Q = 5 ergeben sich bei L = 1 bis 5 dieselben Kosten, aber nur 21 · 6^(L−1) statt 21 · 9^(L−1) Zustände. Der Faktor „je Periode Lieferzeit mal 9“ ist also eine Modellwahl, kein Naturgesetz: wer die nötige Bestellgrenze kennt, verschiebt die Grenze der DP um etwa eine Periode Lieferzeit, sie verschwindet nicht.
- **Rechenzeiten** sind Messungen dieser Umsetzung (nutzt die Pipeline-Struktur, numpy, ein Rechenkern je Prozess) auf einem Arbeitsplatzrechner mit 16 Threads und 32 GB Arbeitsspeicher, während andere Prozesse liefen: nur Größenordnungen. Mit anderer Programmierung verschiebt sich die Grenze; eine andere Fassung wurde hier nicht gemessen. Rechenzeit und Speicher auf Streamlit Cloud sind nicht gemessen.
- **Zustände, die kein Optimum brauchen.** Unter der DP-Politik kommt nur ein kleiner Teil der Zustände vor. Wer die Politik schon kennt, braucht die Tabelle nicht, aber man kennt sie vorher nicht.
- **Lernläufe.** Ein Verfahren (Politikgradient mit gelerntem Basiswert, Adam), eine Einstellung (32 versteckte Einheiten, Lernrate 3e-3, 3.000 Updates à 25.600 simulierten Perioden), 10 Zufalls-Seeds für L = 2 bis 4 und 5 Zufalls-Seeds für L = 5 bis 8. Jede Aussage über „Lernen“ ist eine Aussage über dieses abgestimmte Verfahren. Das Netz wandert bei konstanter Lernrate um das Optimum (die Ursache wurde nicht getrennt gemessen): der Abgriff zu einem Messpunkt streut auch innerhalb eines Laufs. Die Läufe sind vorgerechnet und werden hier nur gelesen.
- **Bewertung des Netzes.** Bis L = 6 wird die gierige Politik des Netzes exakt bewertet (bei L = 6 bis Update 500 per Simulation), ab L = 7 per Simulation mit Standardfehler von etwa 0,1 % der Kosten. Bei L = 7 und 8 sind auch die Regeln Simulationswerte (Parameter auf einem Zufallsstrom abgestimmt, auf einem anderen bewertet). Alle Simulationswerte teilen denselben Zufallsstrom 777; gegen die exakte DP haben sie deshalb einen gemeinsamen Versatz. Eine Messung des Prüfers bei L = 6 und 7 (nicht in den Daten) fand Strom 777 rund 0,013 je Periode (0,23 %) unter dem Erwartungswert; bei L = 7 liegen die Lücken zur DP daher eher rund 0,2 Punkte höher (Capped Base-Stock auf einem unabhängigen Strom mit 40.000 Ketten +0,62 % statt +0,39 %, Base-Stock +10,18 % statt +9,90 %). Im Vergleich der Politiken untereinander wirkt der Versatz auf alle gleichgerichtet.
- **DP bei L = 8.** Die App rechnet sie nicht, und die vorgerechneten Daten enthalten sie nicht: rund 0,8 GB je Wertetabelle (die Rechnung hält mehrere), dazu Rechenzeit von Minuten; für eine App ist das aufwendig, unmöglich ist es nicht. Rechenzeit und Speicher auf Streamlit Cloud sind nicht gemessen. Eine einmalige Messung des Prüfers außerhalb der App (Code dieser Demo, Rechner mit 16 Threads und 32 GB, ein Rechenkern) rechnete die DP bei L = 8: g = 5,782247 nach 72 Iterationen, 522 s, 5,95 GB Spitzenspeicher; kein Test der Demo prüft diese Zahlen. In der App ist Capped Base-Stock bei L = 8 der Maßstab. Nach dieser Messung liegt Capped Base-Stock (Ziel 29, Deckel 3) bei L = 8 rund +0,9 % über dem Optimum (+0,92 % auf einem unabhängigen Strom mit 40.000 Ketten, +0,67 % auf Strom 777). Eine Lücke gegen eine einfache Regel unterschätzt die Lücke zum Optimum um die eigene Lücke dieser Regel (bei L = 1 bis 7 zwischen 0,1 und 1,6 %).
- **Literatur.** Gijsbrechts u. a. (nur die SSRN-Arbeitsfassung vom Juli 2021 gelesen, nicht die gedruckte Fassung) berichten für ihr Verfahren (A3C) bei Lost Sales die Güte der besten Heuristiken, nicht mehr. Ihre Instanzen (Nachfragemittel 5, Lieferzeit 2 bis 4) und ihr Verfahren sind andere: ihre Zahlen sind mit denen dieser Demo nicht vergleichbar. Capped Base-Stock (Xin 2021) ist hier nur nach dem Abstract bekannt: danach gibt es Capped-Base-Stock-Parameter, mit denen die Regel für wachsende Lieferzeit asymptotisch optimal ist; das stützt „gute einfache Regel“, nicht „beste Regel“, und andere Regeln (etwa van Jaarsveld und Arts, arXiv 2101.07519) wurden hier nicht verglichen. Dass sie hier nur 0,1 bis 1,6 % über der DP liegt (L = 1 bis 7), gilt für dieses Beispiel mit Mittel 3; für andere Nachfragemittel wurde das hier nicht gemessen.

## Verifikation

- **Kern:** Einheiten an der Mini-Instanz von Hand gerechnet (`tests/test_algorithm.py`): erwartete Periodenkosten, Rückwärtsrechnung, relative Wertiteration, Regeln, Zählung, Simulation.
- **Orakel (`tests/test_oracle_dpg.py`, unabhängige Rechenwege):** Wertiteration gegen Politik-Iteration auf einer dichten, durch Aufzählung aller Zustandstupel aufgebauten Kette; exakte Auswertung gegen ein lineares Gleichungssystem; Simulation gegen die exakte Auswertung (Band 4 Standardfehler); Rückwärtsschritt des Netzes gegen endliche Differenzen über alle 216 Pfade einer Mini-Instanz, mit Zweigprobe (ein aktionsabhängiger Basiswert verfälscht den Gradienten); Zählformeln gegen Aufzählung; Handbeispiel in Brüchen. Der Orakeltest prüft den Rückwärtsschritt des Netzes; Diskontierung, Entropieterm, Normierung des Vorteils und Adam sind kein erwartungstreuer Gradient und nur durch die Lernkurve geprüft.
- **Zweigproben:** Capped und Base-Stock geben verschiedene Kosten; Q = 4 gibt andere Kosten als Q = 8, Q = 5 gleiche.
- **Verdikte** als reine Funktionen mit Schwellen und jeder Zahl im Text (`tests/test_verdicts.py`); Randfälle (`tests/test_edge_cases.py`); Auswertung der Lernläufe an handgerechneten Daten (`tests/test_stats.py`).
- **Oberfläche:** die App wurde einmal im Browser angesehen (Seite „Gelerntes Netz“, L = 3); AppTest für jede Lieferzeit (1 bis 8), beide Ansichten, jeden Button, jeden Regler an den Grenzen, Permalink mit Begrenzung, Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben), Footer.
- **Wiederholbarkeit der vorgerechneten Zahlen:** `scripts/precompute_refs.py` rechnet die Bezugswerte neu (rund 6 Minuten), `scripts/precompute_curves.py` die Lernläufe (rund 3 bis 4 Minuten je Lauf), `scripts/build_curves_json.py` fasst sie zusammen. Auf demselben Rechner mit denselben Python- und numpy-Versionen gab ein neu gerechneter Lernlauf (L = 2, Zufalls-Seed 0, Updates 0 bis 250) dieselben Zahlen wie der gespeicherte (Abweichung 0,0); über Versionen und Rechner ist das nicht garantiert, deshalb prüft kein Test eine gelernte Zahl exakt. Die Daten stammen aus Vormessungen zu dieser Demo (gleicher Code).
- 119 Tests, Laufzeit rund 30 s (gemessen, geringe Auslastung des Rechners), unter Last länger (`python -m pytest tests -q`); ein Test (DP bei L = 6, einige Sekunden) hat die Marke `slow`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Kennzahlen, Kernfrage, sechs Abschnitte, Annahmen, Mathematik) |
| `dpg_algorithm.py` | Kern: DP (Wertiteration, Rückwärtsrechnung), exakte Auswertung, Regeln als Tabellen |
| `dpg_constants.py` | Beispiel, Regler-Grenzen, Lernverfahren, Farben, Presets |
| `dpg_counts.py` | Zustandszahlen für drei Wachstumsrichtungen, Formate |
| `dpg_sim.py` | Simulation mit gemeinsamen Zufallsströmen, Regeln als Funktionen |
| `dpg_learn.py` | Netz, Politikgradient, Gradient-Bausteine, Lernlauf (nur für die Skripte und Tests) |
| `dpg_stats.py` | Auswertung der Lernläufe, Lücke gegen DP oder Capped |
| `dpg_evaluation.py` | Verdikte (reine Funktionen), live gerechnete Abschnitte, Handbeispiel |
| `dpg_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dpg_presets.py`, `dpg_layout.py` | Permalink, Presets, Zufalls-Seed-Button; fester Kopf |
| `dpg_texts.py` | Annahmen-Text (wörtlich auch in dieser README) |
| `dpg_data/refs.json`, `dpg_data/curves.json` | vorgerechnete Bezugswerte und 50 Lernläufe |
| `scripts/` | Neuberechnung der vorgerechneten Daten (nicht in der CI) |
| `pytest.ini` | Marke `slow` |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly) und pytest |
| `tests/` | Einheiten, Orakel, Claims, Verdikte, Auswertung, Randfälle, Presets, Layout, App, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Lernen live:** ein Lauf braucht Minuten (Median 172 bis 219 s auf dem Messrechner); Streamlit Cloud ist nicht gemessen. Die Läufe liegen vorgerechnet vor.
- **DP bei L = 8 in der App:** nicht live und nicht vorgerechnet (rund 0,8 GB je Tabelle, mehrere Tabellen gleichzeitig; eine einmalige Messung außerhalb der App brauchte 522 s und 5,95 GB Spitzenspeicher, siehe Ehrliche Grenzen). Rechenzeit und Speicher auf Streamlit Cloud sind nicht gemessen.
- **Andere Lernverfahren** (Evolutionsstrategie, Actor-Critic mit mehreren Arbeitern, Ableiten durch den Simulator) und Abstimmung der Einstellung: hier ein Verfahren, eine Einstellung. Erklärt sind Politikgradient und Basiswert in [policy-gradient-demo](https://github.com/sebastian-hanisch/policy-gradient-demo) und [actor-critic-demo](https://github.com/sebastian-hanisch/actor-critic-demo) (diese Demo wiederholt sie nicht), Wertiteration in [value-iteration-demo](https://github.com/sebastian-hanisch/value-iteration-demo), die Explosion einer Tabelle mit der Kapazität in [dynamic-programming-demo](https://github.com/sebastian-hanisch/dynamic-programming-demo).
- **Mehrere Artikel, Dual-Sourcing, Rückstand:** außerhalb dieser Demo. `off-policy-demo` (Offline-Auswertung einer Regel aus Logdaten) ist ein anderes Thema.
- **DP für verderbliche Ware und zwei Stufen:** nur Zustandszahlen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Bestandspolitiken: von Base-Stock bis Regelwahl](https://sebastianhanisch.net/konzepte-bestandspolitiken.html).

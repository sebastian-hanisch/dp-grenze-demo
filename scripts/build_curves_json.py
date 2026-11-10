"""Die Lernläufe (JSON-Zeilen, Dateien ..._L<L>_XM20_s<Seed>.jsonl in einem Ordner) zu dpg_data/curves.json zusammenfassen.

Aufruf:  python scripts/build_curves_json.py --runs DIR
Nur Läufe mit Endmesspunkt bei Update 3.000 werden übernommen. Werte auf 6 Nachkommastellen gerundet (Kosten je Periode).
Die ersten Läufe der Auslieferung stammen aus den Vormessungen des Stücks (gleicher Code, gleiche Einstellungen; Wiederholung des Laufs
L = 2, Zufalls-Seed 0 mit scripts/precompute_curves.py ergab bis Update 250 dieselben Zahlen).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dpg_constants as C  # noqa: E402

OUT = ROOT / "dpg_data" / "curves.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(args.runs).glob(f"*_L*_XM{C.XM}_s*.jsonl")):
        m = re.search(rf"_L(\d+)_XM{C.XM}_s(\d+)\.jsonl$", path.name)
        rows = [json.loads(z) for z in path.read_text(encoding="utf-8").splitlines() if z.strip()]
        if not rows or rows[-1]["update"] != C.UPDATES:
            continue
        runs.setdefault(m.group(1), {})[m.group(2)] = dict(
            updates=[r["update"] for r in rows],
            g=[round(r["g"], 6) for r in rows],
            se=[round(r["se"], 6) for r in rows],
            mode=[r["mode"] for r in rows],
            sek=rows[-1]["sek"],
        )
    data = dict(
        meta=dict(updates=C.UPDATES, periods_per_update=C.PERIODS_PER_STEP, xm=C.XM, q=C.Q_MAX, verfahren="Politikgradient mit gelerntem Basiswert, Adam",
                  hyperparameter=dict(hidden=C.HIDDEN, chains=C.CHAINS_PER_UPDATE, periods=C.PERIODS_PER_UPDATE, gamma=C.GAMMA,
                                      entropy=C.ENTROPY, learning_rate=C.LEARNING_RATE),
                  hinweis="Laufzeit je Lauf (sek) lastbehaftet: mehrere Prozesse liefen gleichzeitig; nur Größenordnung"),
        runs={L: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for L, v in sorted(runs.items(), key=lambda kv: int(kv[0]))},
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print({L: len(v) for L, v in data["runs"].items()}, f"{OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()

"""Lernläufe (Politikgradient) rechnen: je Lauf eine Datei pg_L<L>_XM20_s<Seed>.jsonl mit einer Zeile je Messpunkt.

Aufruf (nicht Teil der CI; ein Lauf mit 3.000 Updates braucht rund 3 bis 4 Minuten auf einem Arbeitsplatzrechner, alle 50 Läufe rund drei Stunden
nacheinander, bei zwei Prozessen gleichzeitig die Hälfte):
    python scripts/precompute_curves.py --out DIR                      # alle 50 Läufe (L = 2 bis 4: Zufalls-Seeds 0 bis 9; L = 5 bis 8: 0 bis 4)
    python scripts/precompute_curves.py --out DIR --jobs 2:0 3:0-2     # nur diese Läufe (Lieferzeit:Zufalls-Seeds)
    python scripts/precompute_curves.py --out DIR --worker 0 --workers 2   # jeder zweite Auftrag (zwei Prozesse, jeder mit eigenem Aufruf)
Fertige Läufe (letzte Zeile bei Update 3000) werden übersprungen. Danach:
    python scripts/build_curves_json.py --runs DIR                     # schreibt dpg_data/curves.json

Wiederholbarkeit: derselbe Aufruf ergibt auf demselben Rechner mit denselben Python- und numpy-Versionen dieselben Zahlen (geprüft am Lauf
L = 2, Zufalls-Seed 0, Updates 0 bis 250: Abweichung 0,0). Über Versionen und Rechner hinweg ist Gleichheit nicht garantiert; deshalb prüfen die
Tests nie eine gelernte Zahl exakt, nur Bänder.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dpg_constants as C  # noqa: E402
import dpg_learn as LN  # noqa: E402

DEFAULT_JOBS = [(L, s) for L in (2, 3, 4) for s in range(10)] + [(L, s) for L in (5, 6, 7, 8) for s in range(5)]


def parse_jobs(specs):
    jobs = []
    for spec in specs:
        L, seeds = spec.split(":")
        lo, _, hi = seeds.partition("-")
        jobs += [(int(L), s) for s in range(int(lo), int(hi or lo) + 1)]
    return jobs


def finished(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        lines = [json.loads(z) for z in path.read_text(encoding="utf-8").splitlines() if z.strip()]
        return bool(lines) and lines[-1]["update"] == C.UPDATES
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", nargs="*")
    ap.add_argument("--updates", type=int, default=C.UPDATES)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = parse_jobs(args.jobs) if args.jobs else DEFAULT_JOBS
    for i, (L, seed) in enumerate(jobs):
        path = out / f"pg_L{L}_XM{C.XM}_s{seed}.jsonl"
        if i % args.workers != args.worker or finished(path):
            continue
        with path.open("w", encoding="utf-8") as fh:
            def on_point(rec, L=L, seed=seed, fh=fh):
                fh.write(json.dumps(dict(L=L, seed=seed, XM=C.XM, **rec)) + "\n")
                fh.flush()
                print(f"L = {L}, Zufalls-Seed {seed}, Update {rec['update']}: Kosten {rec['g']:.4f}", flush=True)
            LN.train(L=L, seed=seed, updates=args.updates, on_point=on_point)


if __name__ == "__main__":
    main()

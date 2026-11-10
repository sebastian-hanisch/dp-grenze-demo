"""README und App: der Annahmen-Text steht wörtlich in beiden; jeder K-Verweis der Befunde-Tabelle gibt es als Test; kein Platzhalter."""

import re
from pathlib import Path

from dpg_texts import ASSUMPTIONS_MARKDOWN

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_assumptions_text_is_in_the_readme_word_for_word():
    assert ASSUMPTIONS_MARKDOWN.rstrip("\n") in README


def test_app_uses_the_shared_assumptions_text():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from dpg_texts import ASSUMPTIONS_MARKDOWN" in app and "st.markdown(ASSUMPTIONS_MARKDOWN)" in app


def test_every_claim_reference_in_the_readme_has_a_test():
    cited = set(re.findall(r"\bK(\d\d)\b", README))
    tests = (ROOT / "tests" / "test_claims.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"def test_k(\d\d)", tests))
    assert cited and cited <= defined, sorted(cited - defined)
    assert defined <= cited | {"04"}, sorted(defined - cited)                       # K04b gehört zu K04; jeder Test der Datei wird in der README genannt


def test_no_placeholders_left():
    marker = "TO" + "DO"
    for path in list(ROOT.glob("*.py")) + [ROOT / "README.md"] + list((ROOT / "scripts").glob("*.py")):
        assert marker not in path.read_text(encoding="utf-8"), path.name
    assert "ASSUMPTIONS_PLACEHOLDER" not in README

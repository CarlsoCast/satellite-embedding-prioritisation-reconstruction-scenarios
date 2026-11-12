"""Check the public repository structure without requiring the large raw data."""
from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "requirements.txt",
    ROOT / "docs/paper_figures.csv",
    ROOT / "outputs/paper_figures/figure_01.png",
    ROOT / "outputs/paper_figures/figure_10.png",
]


def main() -> None:
    missing = [path for path in REQUIRED if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing repository files:\n" + "\n".join(map(str, missing)))

    scripts = sorted((ROOT / "src").rglob("*.py"))
    if not scripts:
        raise RuntimeError("No Python analysis scripts were found.")
    for script in scripts:
        py_compile.compile(str(script), doraise=True)

    figures = sorted((ROOT / "outputs/paper_figures").glob("figure_*.png"))
    if len(figures) != 9:
        raise RuntimeError(f"Expected 9 manuscript figures, found {len(figures)}.")

    print(f"Repository check passed: {len(scripts)} scripts and {len(figures)} manuscript figures.")


if __name__ == "__main__":
    main()

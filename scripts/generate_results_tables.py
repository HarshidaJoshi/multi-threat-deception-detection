"""
Evaluation module for XAI-Verify
Created for dissertation and Scopus journal evaluation.
Compatible with existing project structure.
Do not modify existing project logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluation_helpers import generate_results_tables_md  # noqa: E402


def main() -> None:
    path = generate_results_tables_md()
    print(f"Generated journal summary table: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""CLI for the frozen Day 17 hold-out evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.final_holdout import run_final_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "day17_final_protocol.yaml",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow reuse of an already completed summary; no new results are run.",
    )
    args = parser.parse_args()
    summary = run_final_holdout(args.protocol, allow_existing=args.allow_existing)
    print(f"Final hold-out status: {summary['status']}")
    print(
        "Summary: "
        + str(Path(summary["protocol"]["output_root"]) / "final_holdout_summary.json")
    )


if __name__ == "__main__":
    main()

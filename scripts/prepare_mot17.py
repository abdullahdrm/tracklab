"""Validate MOT17 data and create a reproducible split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Support direct execution: python scripts/prepare_mot17.py
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tracklab.datasets.mot17 import (
    discover_sequences,
    iter_ground_truth_rows,
)
from tracklab.datasets.splits import (
    load_split_plan,
    validate_split_plan,
)


def resolve_project_path(value: str) -> Path:
    """Resolve relative CLI paths against the repository root."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    """Prefer stable repository-relative paths in the manifest."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MOT17 sequence files and write a split manifest."
    )
    parser.add_argument(
        "--root",
        default="data/raw/MOT17/train",
        help="MOT17 training directory",
    )
    parser.add_argument(
        "--splits",
        default="configs/mot17_splits.yaml",
        help="YAML split definition",
    )
    parser.add_argument(
        "--output",
        default="outputs/mot17/manifest.json",
        help="Output manifest path",
    )
    args = parser.parse_args()

    try:
        data_root = resolve_project_path(args.root)
        split_path = resolve_project_path(args.splits)
        output_path = resolve_project_path(args.output)

        plan = load_split_plan(split_path)
        sequences = discover_sequences(data_root, variant=plan.variant)
        by_name = {sequence.name: sequence for sequence in sequences}

        validate_split_plan(plan, set(by_name))

        split_entries: dict[str, list[dict]] = {}
        split_summary: dict[str, dict[str, int]] = {}

        for split_name in ("tuning", "holdout"):
            entries = []

            for name in plan.sequence_names(split_name):
                sequence = by_name[name]
                annotation_rows = sum(
                    1
                    for _ in iter_ground_truth_rows(
                        sequence.gt_file,
                        seq_length=sequence.seq_length,
                    )
                )

                entries.append(
                    {
                        "sequence": sequence.name,
                        "relative_path": sequence.path.relative_to(
                            data_root
                        ).as_posix(),
                        "frames": sequence.seq_length,
                        "fps": sequence.frame_rate,
                        "width": sequence.width,
                        "height": sequence.height,
                        "image_extension": sequence.image_extension,
                        "raw_annotation_rows": annotation_rows,
                    }
                )

            split_entries[split_name] = entries
            split_summary[split_name] = {
                "sequences": len(entries),
                "frames": sum(item["frames"] for item in entries),
                "raw_annotation_rows": sum(
                    item["raw_annotation_rows"] for item in entries
                ),
            }

        manifest = {
            "schema_version": 1,
            "dataset": "MOT17",
            "variant": plan.variant,
            "dataset_root": display_path(data_root),
            "split_config": display_path(split_path),
            "annotation_format": (
                "Raw MOT17 ground truth: frame,id,left,top,width,height,"
                "mark,class_id,visibility (9 columns). No class filtering "
                "is performed during manifest preparation."
            ),
            "splits": split_entries,
            "summary": split_summary,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(sequences)} {plan.variant} sequences.")
    for split_name, summary in split_summary.items():
        print(
            f"{split_name}: {summary['sequences']} sequences, "
            f"{summary['frames']} frames, "
            f"{summary['raw_annotation_rows']} raw annotation rows"
        )
    print(f"Manifest written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

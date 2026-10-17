"""Controlled real and synthetic MOT17 preprocessing ablation experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import yaml

from evaluation.degradations import (
    DEGRADATION_NAMES,
    DegradationName,
    apply_degradation,
)
from evaluation.mot17_detector_comparison import run_sequence_tracker
from evaluation.trackeval_bridge import evaluate_mot_challenge
from tracklab.config import (
    PipelineConfig,
    PreprocessingConfig,
    load_config,
    resolve_device,
)
from tracklab.datasets.mot17 import read_sequence_info
from tracklab.preprocessing.base import Frame, validate_output_frame
from tracklab.preprocessing.colorspace import ColorSpaceName, ColorSpaceOrder
from tracklab.preprocessing.factory import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_SEQUENCES: tuple[str, ...] = (
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
)
PREPROCESSING_MODES: tuple[str, ...] = ("none", "gamma", "clahe", "denoise")
# Deliberately small, interpretable set: test both orders for two-method
# combinations, plus two orderings of a three-method chain.
REAL_PREPROCESSING_CHAINS: tuple[tuple[str, ...], ...] = (
    ("gamma", "clahe"),
    ("clahe", "gamma"),
    ("denoise", "clahe"),
    ("clahe", "denoise"),
    ("gamma", "clahe", "denoise"),
    ("denoise", "gamma", "clahe"),
)
# Compare CLAHE alone and both gamma/CLAHE orders on one brightness channel.
# The existing "clahe" case is the LAB-L baseline; "gamma_clahe" is the prior
# cross-space chain (BGR gamma followed by LAB-L CLAHE).
REAL_COLORSPACE_CASES: tuple[tuple[str, ColorSpaceName, ColorSpaceOrder], ...] = (
    ("hsv_clahe", "hsv", "clahe"),
    ("ycrcb_clahe", "ycrcb", "clahe"),
    ("lab_gamma_clahe", "lab", "gamma_clahe"),
    ("hsv_gamma_clahe", "hsv", "gamma_clahe"),
    ("ycrcb_gamma_clahe", "ycrcb", "gamma_clahe"),
    ("lab_clahe_gamma", "lab", "clahe_gamma"),
    ("hsv_clahe_gamma", "hsv", "clahe_gamma"),
    ("ycrcb_clahe_gamma", "ycrcb", "clahe_gamma"),
)
SYNTHETIC_DEGRADATIONS: tuple[str, ...] = (
    "low_light",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
)
Stage = Literal["real", "synthetic", "all"]


@dataclass(frozen=True, slots=True)
class AblationCase:
    """One detector/tracker condition within the fixed ablation matrix."""

    stage: Literal["real", "synthetic"]
    case_id: str
    degradation: DegradationName
    preprocessing: str
    chain: tuple[str, ...] = ()
    color_space: ColorSpaceName | None = None
    color_space_order: ColorSpaceOrder | None = None


def build_ablation_cases(stage: Stage) -> tuple[AblationCase, ...]:
    """Build a deterministic experiment matrix without allowing split overrides."""
    if stage not in {"real", "synthetic", "all"}:
        raise ValueError("stage must be 'real', 'synthetic', or 'all'")

    cases: list[AblationCase] = []
    if stage in {"real", "all"}:
        for mode in PREPROCESSING_MODES:
            cases.append(
                AblationCase(
                    stage="real",
                    case_id=mode,
                    degradation="none",
                    preprocessing=mode,
                )
            )
        for chain in REAL_PREPROCESSING_CHAINS:
            cases.append(
                AblationCase(
                    stage="real",
                    case_id="_".join(chain),
                    degradation="none",
                    preprocessing="chain",
                    chain=chain,
                )
            )
        for case_id, color_space, order in REAL_COLORSPACE_CASES:
            cases.append(
                AblationCase(
                    stage="real",
                    case_id=case_id,
                    degradation="none",
                    preprocessing="colorspace",
                    color_space=color_space,
                    color_space_order=order,
                )
            )

    if stage in {"synthetic", "all"}:
        for degradation in SYNTHETIC_DEGRADATIONS:
            for mode in PREPROCESSING_MODES:
                cases.append(
                    AblationCase(
                        stage="synthetic",
                        case_id=f"{degradation}__{mode}",
                        degradation=cast(DegradationName, degradation),
                        preprocessing=mode,
                    )
                )

    return tuple(cases)


def select_ablation_cases(
    stage: Stage,
    case_ids: tuple[str, ...] | None = None,
) -> tuple[AblationCase, ...]:
    """Select a deterministic subset of one stage's allowlisted experiment matrix."""
    available = build_ablation_cases(stage)
    if case_ids is None:
        return available
    requested = tuple(case_ids)
    if not requested:
        raise ValueError("case_ids must not be empty when provided")
    if len(requested) != len(set(requested)):
        raise ValueError("case_ids contains duplicates")
    available_by_id = {case.case_id: case for case in available}
    missing = [case_id for case_id in requested if case_id not in available_by_id]
    if missing:
        raise ValueError(
            "Unknown case ID(s) for stage " + stage + ": " + ", ".join(missing)
        )
    requested_set = set(requested)
    return tuple(case for case in available if case.case_id in requested_set)


def build_frame_transform(
    preprocessing_config: PreprocessingConfig,
    degradation: DegradationName = "none",
    *,
    seed: int = 1729,
) -> Callable[[Frame], Frame]:
    """Apply synthetic degradation first, then the selected preprocessor.

    The seeded random noise changes between frames but is reproducible when the
    same sequence is evaluated again. A separate transform is created per
    sequence so paired preprocessing conditions receive identical noise fields.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if degradation not in DEGRADATION_NAMES:
        raise ValueError(f"Unsupported degradation: {degradation}")

    preprocessor = build_preprocessor(preprocessing_config)
    frame_offset = 0

    def transform(frame: Frame) -> Frame:
        nonlocal frame_offset
        degraded = apply_degradation(frame, degradation, seed=seed + frame_offset)
        frame_offset += 1
        transformed = preprocessor(degraded)
        return validate_output_frame(frame, transformed)

    return transform


def validate_tuning_split(
    tuning_sequences: list[str] | tuple[str, ...],
    holdout_sequences: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate the exact allowlisted tuning split and reject hold-out leakage."""
    tuning = tuple(tuning_sequences)
    holdout = set(holdout_sequences)
    if len(tuning) != len(set(tuning)):
        raise ValueError("Tuning split contains duplicate sequences")
    if set(tuning) != set(TUNING_SEQUENCES):
        raise ValueError(
            "Ablation only permits MOT17 tuning sequences 04, 05, 09, and 11"
        )
    leaked = sorted(set(tuning).intersection(holdout))
    if leaked:
        raise ValueError("Hold-out sequences are forbidden: " + ", ".join(leaked))
    return TUNING_SEQUENCES


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_yaml_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a YAML mapping: {path}")
    return value


def _prepare_runtime(
    eval_config_path: str | Path,
    pipeline_config_path: str | Path,
) -> tuple[PipelineConfig, SimpleNamespace, Path, Path, tuple[str, ...]]:
    evaluation_settings = _load_yaml_mapping(_resolve_project_path(eval_config_path))
    split_settings = evaluation_settings.get("splits")
    if not isinstance(split_settings, dict):
        raise TypeError("Evaluation config must contain tuning/holdout splits")
    tuning_sequences = split_settings.get("tuning")
    holdout_sequences = split_settings.get("holdout")
    if not isinstance(tuning_sequences, list) or not isinstance(
        holdout_sequences, list
    ):
        raise TypeError("Tuning and holdout sequence lists must be configured")
    sequences = validate_tuning_split(tuning_sequences, holdout_sequences)

    if evaluation_settings.get("variant", "FRCNN") != "FRCNN":
        raise ValueError(
            "The ablation runner currently expects the MOT17 FRCNN variant"
        )

    data_root = _resolve_project_path(
        evaluation_settings.get("data_root", "data/raw/MOT17/train")
    )
    if not data_root.is_dir():
        raise FileNotFoundError(f"MOT17 data directory not found: {data_root}")

    pipeline_config = load_config(_resolve_project_path(pipeline_config_path))
    yolo_settings = evaluation_settings.get("yolo", {})
    if not isinstance(yolo_settings, dict):
        raise TypeError("The 'yolo' evaluation settings must be a mapping")

    model_path = str(yolo_settings.get("model_path", "models/yolo11n.pt"))
    resolved_model_path = _resolve_project_path(model_path)
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {resolved_model_path}")

    imgsz = int(yolo_settings.get("imgsz", 960))
    confidence = float(yolo_settings.get("confidence", 0.1))
    iou = float(yolo_settings.get("iou", 0.7))
    if imgsz <= 0 or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("YOLO imgsz must be positive and confidence must be in [0, 1]")
    if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("YOLO IoU must be in [0, 1]")

    device = resolve_device(str(yolo_settings.get("device", "auto")))
    detector_config = SimpleNamespace(
        model_path=model_path,
        imgsz=imgsz,
        inference_conf=confidence,
        iou=iou,
        device=device,
    )
    run_config = SimpleNamespace(
        detector=detector_config,
        tracker=pipeline_config.tracker,
    )
    return (
        pipeline_config,
        run_config,
        data_root,
        resolved_model_path,
        sequences,
    )


def _case_configuration(
    case: AblationCase,
    pipeline_config: PipelineConfig,
    run_config: SimpleNamespace,
    sequences: tuple[str, ...],
    seed: int,
) -> dict:
    preprocessing_data = pipeline_config.preprocessing.model_dump(mode="json")
    preprocessing_data["mode"] = case.preprocessing
    if case.preprocessing == "chain":
        preprocessing_data["chain"] = list(case.chain)
    elif case.preprocessing == "colorspace":
        preprocessing_data["color_space"] = case.color_space
        preprocessing_data["color_space_order"] = case.color_space_order
    return {
        "stage": case.stage,
        "case_id": case.case_id,
        "degradation": case.degradation,
        "preprocessing": preprocessing_data,
        "seed": seed,
        "sequences": list(sequences),
        "yolo": {
            "model_path": run_config.detector.model_path,
            "imgsz": run_config.detector.imgsz,
            "confidence": run_config.detector.inference_conf,
            "iou": run_config.detector.iou,
            "device": run_config.detector.device,
            "weights_size_bytes": Path(run_config.detector.model_path).stat().st_size
            if Path(run_config.detector.model_path).is_absolute()
            else _resolve_project_path(run_config.detector.model_path).stat().st_size,
        },
        "bytetrack": pipeline_config.tracker.model_dump(mode="json"),
    }


def _run_case(
    case: AblationCase,
    *,
    pipeline_config: PipelineConfig,
    run_config: SimpleNamespace,
    data_root: Path,
    output_root: Path,
    sequences: tuple[str, ...],
    seed: int,
    resume: bool,
) -> dict:
    case_root = output_root / case.stage / case.case_id
    report_path = case_root / "report.json"
    tracker_name = "yolo_bytetrack"
    trackers_root = case_root / "trackers"
    preprocessing_data = pipeline_config.preprocessing.model_dump(mode="python")
    preprocessing_data["mode"] = case.preprocessing
    if case.preprocessing == "chain":
        preprocessing_data["chain"] = list(case.chain)
    elif case.preprocessing == "colorspace":
        preprocessing_data["color_space"] = case.color_space
        preprocessing_data["color_space_order"] = case.color_space_order
    preprocessing_config = PreprocessingConfig.model_validate(preprocessing_data)
    identity = _case_configuration(case, pipeline_config, run_config, sequences, seed)

    if resume and report_path.is_file():
        old_report = json.loads(report_path.read_text(encoding="utf-8"))
        if old_report.get("configuration") != identity:
            raise ValueError(
                f"Cannot resume {case.case_id}: its saved configuration differs. "
                "Use the same settings or rerun without --resume."
            )
        report = old_report
        report.setdefault("results", {}).setdefault("per_sequence", {})
    else:
        report = {
            "schema_version": 1,
            "status": "in_progress",
            "stage": case.stage,
            "case_id": case.case_id,
            "configuration": identity,
            "results": {"per_sequence": {}, "combined": None},
        }
        _write_json(report_path, report)

    for sequence_name in sequences:
        sequence = read_sequence_info(data_root / sequence_name)
        prediction_file = trackers_root / tracker_name / "data" / f"{sequence.name}.txt"
        old_result = report["results"]["per_sequence"].get(sequence.name)
        can_resume = (
            resume
            and isinstance(old_result, dict)
            and prediction_file.is_file()
            and old_result.get("prediction_file") == str(prediction_file)
            and isinstance(old_result.get("run"), dict)
        )

        if can_resume:
            run_info = dict(old_result["run"])
            run_info["reused_predictions"] = True
            print(f"[{case.stage}/{case.case_id}] {sequence.name}: reuse predictions")
        else:
            transform = build_frame_transform(
                preprocessing_config,
                case.degradation,
                seed=seed,
            )
            print(
                f"[{case.stage}/{case.case_id}] {sequence.name}: running YOLO + ByteTrack"
            )
            run_info = run_sequence_tracker(
                sequence,
                "yolo",
                run_config,
                prediction_file,
                frame_transform=transform,
                verbose=False,
            )
            run_info["reused_predictions"] = False

        elapsed_seconds = float(run_info.get("elapsed_seconds", 0.0))
        run_info["throughput_fps"] = (
            round(sequence.seq_length / elapsed_seconds, 3)
            if elapsed_seconds > 0
            else None
        )
        metrics = evaluate_mot_challenge(
            data_root,
            trackers_root,
            case_root / "trackeval" / "per_sequence" / sequence.name,
            [sequence.name],
            tracker_name=tracker_name,
            variant="FRCNN",
        )
        report["results"]["per_sequence"][sequence.name] = {
            "run": run_info,
            "metrics": metrics,
            "prediction_file": str(prediction_file),
        }
        _write_json(report_path, report)
        print(
            f"  HOTA={metrics['HOTA']:.4f} MOTA={metrics['MOTA']:.4f} "
            f"IDF1={metrics['IDF1']:.4f} IDSW={metrics['IDSW']:.0f} "
            f"time={run_info.get('elapsed_seconds', 0):.1f}s "
            f"FPS={run_info.get('throughput_fps') or 0.0:.2f}"
        )

    combined = evaluate_mot_challenge(
        data_root,
        trackers_root,
        case_root / "trackeval" / "combined",
        list(sequences),
        tracker_name=tracker_name,
        variant="FRCNN",
    )
    report["results"]["combined"] = combined
    report["elapsed_sum_s"] = round(
        sum(
            float(
                report["results"]["per_sequence"][name]["run"].get(
                    "elapsed_seconds", 0.0
                )
            )
            for name in sequences
        ),
        3,
    )
    report["frames_processed"] = sum(
        int(read_sequence_info(data_root / name).seq_length) for name in sequences
    )
    elapsed = float(report["elapsed_sum_s"])
    report["throughput_fps"] = (
        round(report["frames_processed"] / elapsed, 3) if elapsed > 0 else None
    )
    report["status"] = "complete"
    _write_json(report_path, report)
    return report


def _write_stage_summary(
    stage: Literal["real", "synthetic"],
    cases: tuple[AblationCase, ...],
    output_root: Path,
) -> dict:
    rows = []
    for case in cases:
        report_path = output_root / case.stage / case.case_id / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Case report missing: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"Ablation case is incomplete: {case.case_id}")
        rows.append(
            {
                "case_id": case.case_id,
                "preprocessing": case.preprocessing,
                "degradation": case.degradation,
                "combined": report["results"]["combined"],
                "per_sequence": {
                    sequence: entry["metrics"]
                    for sequence, entry in report["results"]["per_sequence"].items()
                },
                "elapsed_sum_s": report.get("elapsed_sum_s", 0.0),
                "frames_processed": report.get("frames_processed", 0),
                "throughput_fps": report.get("throughput_fps"),
                "report_path": str(report_path),
            }
        )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "stage": stage,
        "sequences": list(TUNING_SEQUENCES),
        "holdout_sequences_evaluated": [],
        "cases": rows,
    }
    _write_json(output_root / stage / "ablation_summary.json", summary)

    print(f"\n===== {stage.upper()} ABLATION SUMMARY =====")
    print(
        "case                         HOTA    DetA    AssA    MOTA    IDF1    IDSW   seconds    FPS"
    )
    for row in rows:
        values = row["combined"]
        fps_value = (
            float(row["throughput_fps"]) if row["throughput_fps"] is not None else 0.0
        )
        print(
            f"{row['case_id']:<28} "
            f"{values['HOTA']:.4f}  {values['DetA']:.4f}  "
            f"{values['AssA']:.4f}  {values['MOTA']:.4f}  "
            f"{values['IDF1']:.4f}  {values['IDSW']:5.0f}  "
            f"{row['elapsed_sum_s']:7.1f}  "
            f"{fps_value:7.2f}"
        )
    print(f"JSON: {output_root / stage / 'ablation_summary.json'}")
    return summary


def run_ablation(
    stage: Stage = "real",
    *,
    eval_config_path: str | Path = "configs/eval.yaml",
    pipeline_config_path: str | Path = "configs/pipeline.yaml",
    output_root: str | Path = "outputs/mot17/ablations",
    seed: int = 1729,
    resume: bool = False,
    case_ids: tuple[str, ...] | None = None,
) -> list[dict]:
    """Run the selected E2/E2b matrix on the four allowlisted tuning sequences."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    (
        pipeline_config,
        run_config,
        data_root,
        model_path,
        sequences,
    ) = _prepare_runtime(eval_config_path, pipeline_config_path)
    output = _resolve_project_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    print(f"YOLO weights: {model_path}")
    print(f"Tuning sequences only: {', '.join(sequences)}")
    print("Hold-out sequences evaluated: none")
    print(
        f"Baseline inference: imgsz={run_config.detector.imgsz}, "
        f"conf={run_config.detector.inference_conf}, "
        f"iou={run_config.detector.iou}, device={run_config.detector.device}"
    )

    cases = select_ablation_cases(stage, case_ids)
    reports: list[dict] = []
    for case in cases:
        print(f"\n===== {case.stage.upper()} / {case.case_id} =====")
        report = _run_case(
            case,
            pipeline_config=pipeline_config,
            run_config=run_config,
            data_root=data_root,
            output_root=output,
            sequences=sequences,
            seed=seed,
            resume=resume,
        )
        reports.append(report)

    for stage_name in ("real", "synthetic"):
        stage_cases = tuple(case for case in cases if case.stage == stage_name)
        if stage_cases:
            _write_stage_summary(
                cast(Literal["real", "synthetic"], stage_name), stage_cases, output
            )

    return reports

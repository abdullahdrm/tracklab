"""Loading and validating TrackLab configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)


class StrictModel(BaseModel):
    """Reject unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DetectorConfig(StrictModel):
    """YOLO detector settings."""

    model_path: str = "models/yolo11n.pt"
    imgsz: PositiveInt = 960
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    classes: list[int] = Field(default_factory=lambda: [0], min_length=1)
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        resolve_device(value)
        return value


class TrackerConfig(StrictModel):
    """ByteTrack settings compatible with the expected YAML schema."""

    tracker_type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.25, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    fuse_score: bool = True


class GammaConfig(StrictModel):
    """Gamma correction settings."""

    value: float = Field(default=0.8, gt=0.0)


class ClaheConfig(StrictModel):
    """CLAHE settings."""

    clip_limit: float = Field(default=2.0, gt=0.0)
    tile_grid: tuple[PositiveInt, PositiveInt] = (8, 8)


class DenoiseConfig(StrictModel):
    """Denoising settings for bilateral and colored non-local means filters."""

    method: Literal["bilateral", "fastNlMeans"] = "bilateral"
    d: PositiveInt = 5
    sigma_color: float = Field(default=50.0, ge=0.0)
    sigma_space: float = Field(default=50.0, ge=0.0)
    h: float = Field(default=10.0, ge=0.0)
    h_color: float = Field(default=10.0, ge=0.0)


class RouterConfig(StrictModel):
    """Optional clip-level preprocessing router."""

    enabled: bool = False


class PreprocessingConfig(StrictModel):
    """Classical image preprocessing settings."""

    mode: Literal["none", "gamma", "clahe", "denoise", "chain", "colorspace"] = "none"
    gamma: GammaConfig = Field(default_factory=GammaConfig)
    clahe: ClaheConfig = Field(default_factory=ClaheConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    chain: list[Literal["gamma", "clahe", "denoise"]] = Field(
        default_factory=lambda: ["gamma", "clahe"]
    )
    color_space: Literal["lab", "hsv", "ycrcb"] = "lab"
    color_space_order: Literal["clahe", "gamma_clahe", "clahe_gamma"] = "gamma_clahe"
    router: RouterConfig = Field(default_factory=RouterConfig)


class LineConfig(StrictModel):
    """A line used for directional pedestrian counting."""

    id: str
    p1: tuple[float, float]
    p2: tuple[float, float]


class TrajectoryConfig(StrictModel):
    """Settings for storing and displaying trajectories."""

    min_len: PositiveInt = 10
    smooth_window: PositiveInt = 5


class AnalyticsConfig(StrictModel):
    """Line counting and trajectory settings."""

    point: Literal["foot", "center"] = "foot"
    lines: list[LineConfig] = Field(default_factory=list)
    min_track_age: PositiveInt = 5
    cooldown_frames: int = Field(default=15, ge=0)
    trajectories: TrajectoryConfig = Field(default_factory=TrajectoryConfig)


class OutputConfig(StrictModel):
    """Video, JSON and quality-report settings."""

    save_video: bool = True
    save_json: bool = True
    include_frames: bool = True
    save_mot_txt: bool = False
    quality_every_n_frames: PositiveInt = 15


class PipelineConfig(StrictModel):
    """Validated configuration consumed by the processing pipeline."""

    detector: DetectorConfig
    tracker: TrackerConfig
    preprocessing: PreprocessingConfig
    analytics: AnalyticsConfig
    output: OutputConfig
    tracker_yaml: Path


def _read_yaml(path: Path) -> dict:
    """Read a YAML file and require a mapping at its root."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError(f"Configuration must contain a YAML mapping: {path}")

    return data


def load_config(path: str | Path) -> PipelineConfig:
    """Load the main YAML file and validate all referenced settings.

    Paths to component YAML files are resolved relative to the main
    configuration file, not the terminal's current working directory.
    """

    config_path = Path(path).expanduser().resolve()
    main_config = _read_yaml(config_path)

    component_models = {
        "detector": DetectorConfig,
        "tracker": TrackerConfig,
        "preprocessing": PreprocessingConfig,
        "analytics": AnalyticsConfig,
    }

    component_data: dict[str, dict] = {}
    component_paths: dict[str, Path] = {}

    for name in component_models:
        reference = main_config.get(name)

        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"'{name}' must reference a YAML filename in {config_path}"
            )

        component_path = (config_path.parent / reference).resolve()
        component_paths[name] = component_path
        component_data[name] = _read_yaml(component_path)

    validated = {
        name: model.model_validate(component_data[name])
        for name, model in component_models.items()
    }

    return PipelineConfig(
        **validated,
        output=OutputConfig.model_validate(main_config.get("output", {})),
        tracker_yaml=component_paths["tracker"],
    )


def resolve_device(name: str) -> str:
    """Resolve 'auto' to MPS, CUDA or CPU, in that priority order."""

    cleaned = name.strip().lower()

    if cleaned == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = getattr(mps_backend, "is_available", None)

        if callable(mps_available) and mps_available():
            return "mps"

        cuda_backend = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda_backend, "is_available", None)

        if callable(cuda_available) and cuda_available():
            return "cuda"

        return "cpu"

    if cleaned in {"mps", "cuda", "cpu"} or (
        cleaned.startswith("cuda:") and cleaned[5:].isdigit()
    ):
        return cleaned

    raise ValueError(f"Unsupported device: {name}")

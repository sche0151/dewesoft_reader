"""Persistent processing preferences; no threshold is assumed on first use."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import tempfile

from dxd_analysis import DetectionConfig


@dataclass(frozen=True)
class ProcessingSettings:
    ct_min_peak_a: float
    minimum_event_ms: float = 3.0
    rms_window_ms: float = 1.0
    start_fraction_of_peak: float = 0.01
    end_fraction_of_peak: float = 0.005
    start_noise_multiplier: float = 8.0
    end_noise_multiplier: float = 4.0
    start_hold_ms: float = 0.25
    end_hold_ms: float = 15.0
    minimum_signal_to_noise: float = 5.0

    def __post_init__(self):
        if isinstance(self.ct_min_peak_a, bool) or not isinstance(self.ct_min_peak_a, (int, float)):
            raise ValueError("CT threshold must be a number in amps")
        if not math.isfinite(self.ct_min_peak_a) or self.ct_min_peak_a < 0:
            raise ValueError("CT threshold must be finite and non-negative")
        self.detection_config()

    def detection_config(self) -> DetectionConfig:
        """Build the exact validated detector configuration shared with the GUI."""
        return DetectionConfig(**{field.name: getattr(self, field.name) for field in fields(DetectionConfig)})


def default_settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "DewesoftReader" / "settings.json"


def load_settings(path: Path) -> ProcessingSettings | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProcessingSettings(ct_min_peak_a=data["ct_min_peak_a"], **{
            field.name: data.get(field.name, field.default) for field in fields(DetectionConfig)})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid settings file {path}: {exc}") from exc


def save_settings(settings: ProcessingSettings, path: Path) -> None:
    """Replace atomically so an interrupted save cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(asdict(settings), f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

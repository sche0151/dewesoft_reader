"""Pure signal analysis shared by the batch runner and future desktop GUI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    # Detection only. Metrics are always calculated from the original samples.
    rms_window_ms: float = 1.0
    start_fraction_of_peak: float = 0.01
    end_fraction_of_peak: float = 0.005
    start_noise_multiplier: float = 8.0
    end_noise_multiplier: float = 4.0
    start_hold_ms: float = 0.25
    end_hold_ms: float = 15.0
    minimum_event_ms: float = 3
    minimum_signal_to_noise: float = 5.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.rms_window_ms <= 0:
            raise ValueError("rms_window_ms must be greater than zero")
        for name in ("start_fraction_of_peak", "end_fraction_of_peak"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be greater than zero and at most 1")
        if self.end_fraction_of_peak > self.start_fraction_of_peak:
            raise ValueError("End peak fraction must not exceed start peak fraction")
        if self.end_noise_multiplier > self.start_noise_multiplier:
            raise ValueError("End noise multiplier must not exceed start noise multiplier")
        if self.minimum_signal_to_noise < 1:
            raise ValueError("minimum_signal_to_noise must be at least 1")


CONFIG = DetectionConfig()


def ct_is_usable(current_a: np.ndarray, threshold_a: float) -> bool:
    """The threshold is in SI amps and shared by all phases; equality qualifies."""
    if not math.isfinite(threshold_a) or threshold_a < 0:
        raise ValueError("CT threshold must be finite and non-negative")
    if current_a.size == 0 or not np.all(np.isfinite(current_a)):
        return False
    peak = float(np.max(np.abs(current_a)))
    # An all-zero channel remains flat even if the user sets the threshold to zero.
    return peak > 0 and peak >= threshold_a


def validate_signal(values: np.ndarray, timestamps: np.ndarray) -> None:
    if values.ndim != 1 or timestamps.ndim != 1 or values.size != timestamps.size or not values.size:
        raise ValueError("Signal and timestamps must be non-empty, matching one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("Signal contains non-finite samples or timestamps")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be strictly increasing")


def moving_rms(values: np.ndarray, window_samples: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float64)
    window_samples = min(values.size, max(1, int(window_samples)))
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    return np.sqrt(np.convolve(np.square(values), kernel, mode="same"))


def _first_sustained(mask: np.ndarray, length: int, start: int = 0) -> Optional[int]:
    length = max(1, int(length))
    if start >= mask.size:
        return None

    segment = mask[start:].astype(np.int8, copy=False)
    if segment.size < length:
        return None

    if length == 1:
        hits = np.flatnonzero(segment)
    else:
        runs = np.convolve(segment, np.ones(length, dtype=np.int64), mode="valid")
        hits = np.flatnonzero(runs == length)

    if hits.size == 0:
        return None
    return start + int(hits[0])


def detect_event(current_a: np.ndarray, sample_rate_hz: float, cfg: DetectionConfig = CONFIG):
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive and finite")
    if current_a.ndim != 1 or not np.all(np.isfinite(current_a)):
        raise ValueError("Current must be a finite one-dimensional signal")
    n = current_a.size
    if n < 2:
        return None, "NO_EVENT_DETECTED"

    window = min(n, max(1, round(cfg.rms_window_ms * sample_rate_hz / 1000.0)))
    envelope = moving_rms(current_a, window)

    # The lower percentile is a robust estimate of the non-event floor even if
    # the file contains a substantial active-current region.
    noise_floor = float(np.percentile(envelope, 10.0))
    peak_envelope = float(np.max(envelope))

    noise_reference = max(noise_floor, np.finfo(float).eps)
    if peak_envelope / noise_reference < cfg.minimum_signal_to_noise:
        return None, "NO_EVENT_DETECTED"

    dynamic_range = max(0.0, peak_envelope - noise_floor)
    start_threshold = max(
        noise_floor * cfg.start_noise_multiplier,
        noise_floor + cfg.start_fraction_of_peak * dynamic_range,
    )
    end_threshold = max(
        noise_floor * cfg.end_noise_multiplier,
        noise_floor + cfg.end_fraction_of_peak * dynamic_range,
    )
    end_threshold = min(end_threshold, start_threshold)

    start_hold = max(1, round(cfg.start_hold_ms * sample_rate_hz / 1000.0))
    end_hold = max(1, round(cfg.end_hold_ms * sample_rate_hz / 1000.0))
    # Round up so a duration between sample boundaries cannot admit a shorter
    # event. The tiny sample tolerance accommodates native timestamp round-off.
    minimum_event = max(1, math.ceil(cfg.minimum_event_ms * sample_rate_hz / 1000.0 - 1e-7))

    events = []
    short_event_found = False
    search_from = 0

    while search_from < n:
        start = _first_sustained(envelope >= start_threshold, start_hold, search_from)
        if start is None:
            break

        end_low = _first_sustained(envelope <= end_threshold, end_hold, start + start_hold)
        if end_low is None:
            end = n
        else:
            end = end_low

        # The centred envelope is only a coarse detector. Refine on original
        # samples rather than adding another half-window of quiet data.
        left = max(search_from, start - window)
        right = min(n, end + window)
        absolute = np.abs(current_a[left:right])
        starts = np.flatnonzero(absolute >= start_threshold)
        ends = np.flatnonzero(absolute >= end_threshold)
        if starts.size and ends.size:
            start = left + int(starts[0])
            end = left + int(ends[-1]) + 1  # exclusive sample boundary
        else:
            start = end

        if end > start and end - start >= minimum_event:
            energy_score = float(np.sum(np.square(current_a[start:end])))
            events.append((start, end, energy_score))
        elif end > start:
            short_event_found = True

        if end_low is None:
            break
        search_from = max(end + 1, end_low + end_hold)

    if not events:
        return None, "EVENT_SHORTER_THAN_MINIMUM" if short_event_found else "NO_EVENT_DETECTED"

    # A normal test should contain one event. If several are present, analyse
    # the event carrying the most I^2 content and flag the row for review.
    start, end, _ = max(events, key=lambda item: item[2])
    status = "MULTIPLE_EVENTS" if len(events) > 1 else "OK"
    return (start, end), status


def sample_widths(timestamps: np.ndarray, sample_rate_hz: float, is_async: bool) -> np.ndarray:
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive and finite")
    if timestamps.size == 0 or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not is_async:
        return np.full(timestamps.size, 1.0 / sample_rate_hz, dtype=np.float64)

    if timestamps.size == 1:
        return np.array([1.0 / sample_rate_hz], dtype=np.float64)

    dt = np.diff(timestamps)
    fallback = float(np.median(dt))
    return np.concatenate([dt, np.array([fallback])])


def calculate_current_metrics(
    current_a: np.ndarray,
    timestamps: np.ndarray,
    is_async: bool,
    sample_rate_hz: float,
    start: int,
    end: int,
):
    validate_signal(current_a, timestamps)
    if not 0 <= start < end <= current_a.size:
        raise ValueError("Invalid current-event bounds")
    values = current_a[start:end]
    times = timestamps[start:end]
    # Use the following timestamp where available, including at the event end.
    widths = sample_widths(timestamps, sample_rate_hz, is_async)[start:end]

    peak_current_a = float(np.max(np.abs(values)))
    i2t_a2s = float(np.sum(np.square(values) * widths))
    duration_s = float(np.sum(widths))
    rms_current_a = float(np.sqrt(i2t_a2s / duration_s))
    start_time_s = float(times[0])
    end_time_s = start_time_s + duration_s if not is_async else float(times[-1] + widths[-1])

    return {
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,
        "peak_current_a": peak_current_a,
        "rms_current_a": rms_current_a,
        "i2t_a2s": i2t_a2s,
    }


def calculate_voltage_rms(
    voltage_v: np.ndarray,
    timestamps: np.ndarray,
    event_start_time_s: float,
    sample_rate_hz: float,
    is_async: bool = False,
) -> float:
    """RMS from the first voltage sample to (excluding) this phase's event start.

    Uniformly sampled data uses ordinary sample RMS. Irregular samples use
    time weights, clipping the last interval at the current-event boundary.
    """
    validate_signal(voltage_v, timestamps)
    if not math.isfinite(event_start_time_s):
        raise ValueError("Event start time must be finite")
    mask = timestamps < event_start_time_s
    values = voltage_v[mask]
    if values.size == 0:
        raise ValueError("NO_PRE_EVENT_VOLTAGE")
    if is_async:
        widths = sample_widths(timestamps, sample_rate_hz, True)[mask]
        widths = np.minimum(widths, event_start_time_s - timestamps[mask])
        return float(np.sqrt(np.sum(np.square(values) * widths) / np.sum(widths)))
    return float(np.sqrt(np.mean(np.square(values))))



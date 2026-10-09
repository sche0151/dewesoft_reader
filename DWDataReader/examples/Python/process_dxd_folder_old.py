from __future__ import annotations

import argparse
import csv
import ctypes
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from DWDataReaderHeader import (
    DWChannel,
    DWChannelProps,
    DWChannelType,
    DWFileInfo,
    DWMeasurementInfo,
    READER_HANDLE,
    check_error,
    decode_bytes,
    load_library,
)

'''
python DWDataReader/examples/Python/process_dxd_folder.py "D:/Eaton_P02478763_252108361" -o results.csv
'''

PHASES = ("Red", "White", "Blue")
EXPECTED_SAMPLE_RATE_HZ = 20_000.0


@dataclass(frozen=True)
class DetectionConfig:
    # Detection only. Metrics are always calculated from the original samples.
    rms_window_ms: float = 1.0
    start_fraction_of_peak: float = 0.01
    end_fraction_of_peak: float = 0.005
    start_noise_multiplier: float = 8.0
    end_noise_multiplier: float = 4.0
    start_hold_ms: float = 0.25
    end_hold_ms: float = 5.0
    minimum_event_ms: float = 1.0
    minimum_signal_to_noise: float = 5.0


CONFIG = DetectionConfig()
lib = None


def channel_name(ch: DWChannel) -> str:
    return decode_bytes(ch.name).strip()


def channel_unit(ch: DWChannel) -> str:
    return decode_bytes(ch.unit).strip()


def _phase_match(name: str, phase: str) -> bool:
    return re.search(rf"\b{re.escape(phase)}\b", name, flags=re.IGNORECASE) is not None


def _is_ct(name: str) -> bool:
    return re.search(r"\bC\s*\.?\s*T\s*\.?", name, flags=re.IGNORECASE) is not None


def _is_tx(name: str) -> bool:
    return re.search(r"\bT\s*\.?\s*X\s*\.?", name, flags=re.IGNORECASE) is not None


def choose_phase_channels(channels: list[DWChannel], phase: str):
    phase_channels = [ch for ch in channels if _phase_match(channel_name(ch), phase)]

    shunts = [ch for ch in phase_channels if re.search(r"\bshunt\b", channel_name(ch), re.I)]
    cts = [ch for ch in phase_channels if _is_ct(channel_name(ch))]
    voltages = [ch for ch in phase_channels if _is_tx(channel_name(ch))]

    warnings = []

    if shunts:
        current = sorted(shunts, key=lambda ch: ch.index)[0]
        current_source = "Shunt"
        if len(shunts) > 1:
            warnings.append("MULTIPLE_SHUNT_CHANNELS")
    elif cts:
        current = sorted(cts, key=lambda ch: ch.index)[0]
        current_source = "C.T."
        if len(cts) > 1:
            warnings.append("MULTIPLE_CT_CHANNELS")
    else:
        current = None
        current_source = ""

    if voltages:
        voltage = sorted(voltages, key=lambda ch: ch.index)[0]
        if len(voltages) > 1:
            warnings.append("MULTIPLE_TX_CHANNELS")
    else:
        voltage = None

    return current, current_source, voltage, warnings


def unit_factor_to_si(unit: str, quantity: str) -> float:
    normalized = unit.strip().replace(" ", "").lower().replace("µ", "u").replace("μ", "u")

    if quantity == "current":
        factors = {
            "a": 1.0,
            "ka": 1_000.0,
            "ma": 1e-3,
            "ua": 1e-6,
        }
    elif quantity == "voltage":
        factors = {
            "v": 1.0,
            "kv": 1_000.0,
            "mv": 1e-3,
            "uv": 1e-6,
        }
    else:
        raise ValueError(f"Unknown quantity {quantity!r}")

    if normalized not in factors:
        raise ValueError(f"Unsupported {quantity} unit {unit!r}")
    return factors[normalized]


def get_channel_type(reader: READER_HANDLE, ch: DWChannel) -> DWChannelType:
    value = ctypes.c_int()
    value_len = ctypes.c_int(ctypes.sizeof(value))
    check_error(
        lib,
        lib.DWIGetChannelProps(
            reader,
            ch.index,
            DWChannelProps.DW_CH_TYPE,
            ctypes.byref(value),
            ctypes.byref(value_len),
        ),
    )
    return DWChannelType(value.value)


def get_channels(reader: READER_HANDLE) -> list[DWChannel]:
    count = ctypes.c_int()
    check_error(lib, lib.DWIGetChannelListCount(reader, ctypes.byref(count)))
    channel_array = (DWChannel * count.value)()
    check_error(lib, lib.DWIGetChannelList(reader, channel_array))
    return list(channel_array)


def read_channel(reader: READER_HANDLE, ch: DWChannel, sample_rate_hz: float):
    if ch.array_size != 1:
        raise ValueError(f"Array channel is not supported: {channel_name(ch)!r} (array_size={ch.array_size})")

    sample_count = ctypes.c_longlong()
    check_error(lib, lib.DWIGetScaledSamplesCount(reader, ch.index, ctypes.byref(sample_count)))

    n = sample_count.value
    if n <= 0:
        raise ValueError(f"Channel has no samples: {channel_name(ch)!r}")

    samples_c = (ctypes.c_double * n)()
    ch_type = get_channel_type(reader, ch)

    timestamps_c = None
    if ch_type == DWChannelType.DW_CH_TYPE_ASYNC:
        timestamps_c = (ctypes.c_double * n)()

    check_error(
        lib,
        lib.DWIGetScaledSamples(reader, ch.index, 0, sample_count, samples_c, timestamps_c),
    )

    samples = np.ctypeslib.as_array(samples_c).astype(np.float64, copy=True)

    if timestamps_c is None:
        timestamps = np.arange(n, dtype=np.float64) / sample_rate_hz
        is_async = False
    else:
        timestamps = np.ctypeslib.as_array(timestamps_c).astype(np.float64, copy=True)
        is_async = True

    return samples, timestamps, is_async


def moving_rms(values: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
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
        runs = np.convolve(segment, np.ones(length, dtype=np.int16), mode="valid")
        hits = np.flatnonzero(runs == length)

    if hits.size == 0:
        return None
    return start + int(hits[0])


def detect_event(current_a: np.ndarray, sample_rate_hz: float, cfg: DetectionConfig = CONFIG):
    n = current_a.size
    if n < 2:
        return None, "NO_EVENT_DETECTED"

    window = max(1, round(cfg.rms_window_ms * sample_rate_hz / 1000.0))
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
    minimum_event = max(1, round(cfg.minimum_event_ms * sample_rate_hz / 1000.0))

    events = []
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

        # np.convolve(..., mode="same") centres the RMS window. Correct the
        # approximate boundary by half a window before calculating metrics.
        start = max(0, start - window // 2)
        end = min(n, end + window // 2)

        if end - start >= minimum_event:
            energy_score = float(np.sum(np.square(current_a[start:end])))
            events.append((start, end, energy_score))

        if end_low is None:
            break
        search_from = max(end + 1, end_low + end_hold)

    if not events:
        return None, "NO_EVENT_DETECTED"

    # A normal test should contain one event. If several are present, analyse
    # the event carrying the most I^2 content and flag the row for review.
    start, end, _ = max(events, key=lambda item: item[2])
    status = "MULTIPLE_EVENTS" if len(events) > 1 else "OK"
    return (start, end), status


def sample_widths(timestamps: np.ndarray, sample_rate_hz: float, is_async: bool) -> np.ndarray:
    if not is_async:
        return np.full(timestamps.size, 1.0 / sample_rate_hz, dtype=np.float64)

    if timestamps.size == 1:
        return np.array([1.0 / sample_rate_hz], dtype=np.float64)

    dt = np.diff(timestamps)
    positive_dt = dt[dt > 0]
    fallback = float(np.median(positive_dt)) if positive_dt.size else 1.0 / sample_rate_hz
    dt = np.where(dt > 0, dt, fallback)
    return np.concatenate([dt, np.array([fallback])])


def calculate_current_metrics(
    current_a: np.ndarray,
    timestamps: np.ndarray,
    is_async: bool,
    sample_rate_hz: float,
    start: int,
    end: int,
):
    values = current_a[start:end]
    times = timestamps[start:end]
    widths = sample_widths(times, sample_rate_hz, is_async)

    peak_current_a = float(np.max(np.abs(values)))
    rms_current_a = float(np.sqrt(np.mean(np.square(values))))
    i2t_a2s = float(np.sum(np.square(values) * widths))
    duration_s = float(np.sum(widths))
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
    start_time_s: float,
    end_time_s: float,
) -> float:
    mask = (timestamps >= start_time_s) & (timestamps < end_time_s)
    values = voltage_v[mask]
    if values.size == 0:
        raise ValueError("No voltage samples fall inside the detected current-event window")
    return float(np.sqrt(np.mean(np.square(values))))


def empty_row(filename: str, phase: str):
    return {
        "File": filename,
        "Phase": phase,
        "Current Source": "",
        "Current Channel": "",
        "Voltage Channel": "",
        "Sample Rate (Hz)": "",
        "Start Time (s)": "",
        "Duration (s)": "",
        "Peak Current (A)": "",
        "RMS Current (A)": "",
        "I^2t (A^2 s)": "",
        "RMS Voltage (V)": "",
        "Status": "",
    }


def process_open_file(reader: READER_HANDLE, path: Path) -> list[dict]:
    measurement_info = DWMeasurementInfo(0, 0, 0, 0)
    check_error(lib, lib.DWIGetMeasurementInfo(reader, ctypes.byref(measurement_info)))
    sample_rate_hz = float(measurement_info.sample_rate)

    channels = get_channels(reader)
    rows = []

    for phase in PHASES:
        row = empty_row(path.name, phase)
        row["Sample Rate (Hz)"] = sample_rate_hz
        statuses = []

        if not math.isclose(sample_rate_hz, EXPECTED_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1.0):
            statuses.append(f"UNEXPECTED_SAMPLE_RATE_{sample_rate_hz:g}Hz")

        current_ch, source, voltage_ch, selection_warnings = choose_phase_channels(channels, phase)
        statuses.extend(selection_warnings)

        if current_ch is None:
            statuses.append("NO_CURRENT_CHANNEL")
            row["Status"] = ";".join(statuses)
            rows.append(row)
            continue

        row["Current Source"] = source
        row["Current Channel"] = channel_name(current_ch)
        if voltage_ch is not None:
            row["Voltage Channel"] = channel_name(voltage_ch)

        try:
            current_raw, current_t, current_async = read_channel(reader, current_ch, sample_rate_hz)
            current_factor = unit_factor_to_si(channel_unit(current_ch), "current")
            current_a = current_raw * current_factor
        except Exception as exc:
            statuses.append(f"CURRENT_READ_ERROR:{exc}")
            row["Status"] = ";".join(statuses)
            rows.append(row)
            continue

        event, event_status = detect_event(current_a, sample_rate_hz)
        if event is None:
            statuses.append(event_status)
            row["Status"] = ";".join(statuses)
            rows.append(row)
            continue

        if event_status != "OK":
            statuses.append(event_status)

        start, end = event
        metrics = calculate_current_metrics(
            current_a,
            current_t,
            current_async,
            sample_rate_hz,
            start,
            end,
        )

        row["Start Time (s)"] = metrics["start_time_s"]
        row["Duration (s)"] = metrics["duration_s"]
        row["Peak Current (A)"] = metrics["peak_current_a"]
        row["RMS Current (A)"] = metrics["rms_current_a"]
        row["I^2t (A^2 s)"] = metrics["i2t_a2s"]

        if voltage_ch is None:
            statuses.append("NO_VOLTAGE_CHANNEL")
        else:
            try:
                voltage_raw, voltage_t, _ = read_channel(reader, voltage_ch, sample_rate_hz)
                voltage_factor = unit_factor_to_si(channel_unit(voltage_ch), "voltage")
                voltage_v = voltage_raw * voltage_factor
                row["RMS Voltage (V)"] = calculate_voltage_rms(
                    voltage_v,
                    voltage_t,
                    metrics["start_time_s"],
                    metrics["end_time_s"],
                )
            except Exception as exc:
                statuses.append(f"VOLTAGE_READ_ERROR:{exc}")

        row["Status"] = ";".join(statuses) if statuses else "OK"
        rows.append(row)

    return rows


def process_file(reader: READER_HANDLE, path: Path) -> list[dict]:
    file_info = DWFileInfo(0, 0, 0)
    opened = False
    try:
        c_filename = ctypes.c_char_p(str(path).encode("utf-8"))
        check_error(lib, lib.DWIOpenDataFile(reader, c_filename, ctypes.byref(file_info)))
        opened = True
        return process_open_file(reader, path)
    finally:
        if opened:
            check_error(lib, lib.DWICloseDataFile(reader))


def find_dxd_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() == ".dxd")


def write_csv(rows: list[dict], output_path: Path):
    fieldnames = list(empty_row("", "").keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global lib

    parser = argparse.ArgumentParser(
        description="Batch-process Dewesoft .dxd files into one phase-per-row summary CSV."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .dxd files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <folder>/dxd_summary.csv)",
    )
    parser.add_argument("--recursive", action="store_true", help="Also search subfolders")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    output_path = args.output.resolve() if args.output else folder / "dxd_summary.csv"
    files = find_dxd_files(folder, args.recursive)
    if not files:
        raise SystemExit(f"No .dxd files found in {folder}")

    lib = load_library()
    reader = READER_HANDLE()
    check_error(lib, lib.DWICreateReader(ctypes.byref(reader)))

    rows = []
    try:
        for number, path in enumerate(files, start=1):
            print(f"[{number}/{len(files)}] {path.name}")
            try:
                rows.extend(process_file(reader, path))
            except Exception as exc:
                # Keep the batch running and leave an auditable failure row for each phase.
                for phase in PHASES:
                    row = empty_row(path.name, phase)
                    row["Status"] = f"FILE_ERROR:{exc}"
                    rows.append(row)
    finally:
        check_error(lib, lib.DWIDestroyReader(reader))

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import ctypes
import math
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from dxd_analysis import (DetectionConfig, calculate_current_metrics, calculate_voltage_rms, detect_event,
                          ct_is_usable, validate_signal)
from dxd_settings import ProcessingSettings, default_settings_path, load_settings, save_settings

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


PHASES = ("Red", "White", "Blue")
EXPECTED_SAMPLE_RATE_HZ = 20_000.0

DETECTION_COLUMNS = {
    "rms_window_ms": "Detection RMS Window (ms)",
    "start_fraction_of_peak": "Start Peak Fraction",
    "end_fraction_of_peak": "End Peak Fraction",
    "start_noise_multiplier": "Start Noise Multiplier",
    "end_noise_multiplier": "End Noise Multiplier",
    "start_hold_ms": "Start Hold (ms)",
    "end_hold_ms": "End Hold (ms)",
    "minimum_event_ms": "Minimum Event Duration (ms)",
    "minimum_signal_to_noise": "Minimum Signal to Noise",
}
lib = None


def settings_columns(settings: ProcessingSettings) -> dict:
    return {"CT Minimum Peak (A)": settings.ct_min_peak_a,
            **{column: getattr(settings, name) for name, column in DETECTION_COLUMNS.items()}}



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

    ct = min(cts, key=lambda ch: ch.index) if cts else None
    shunt = min(shunts, key=lambda ch: ch.index) if shunts else None
    if len(cts) > 1:
        warnings.append("MULTIPLE_CT_CHANNELS")
    if len(shunts) > 1:
        warnings.append("MULTIPLE_SHUNT_CHANNELS")

    if voltages:
        voltage = sorted(voltages, key=lambda ch: ch.index)[0]
        if len(voltages) > 1:
            warnings.append("MULTIPLE_TX_CHANNELS")
    else:
        voltage = None

    return ct, shunt, voltage, warnings


def configure_library(library):
    """Declare the ABI used here, particularly 64-bit sample positions."""
    pointer = ctypes.POINTER
    signatures = {
        "DWICreateReader": [pointer(READER_HANDLE)],
        "DWIDestroyReader": [READER_HANDLE],
        "DWIOpenDataFile": [READER_HANDLE, ctypes.c_char_p, pointer(DWFileInfo)],
        "DWICloseDataFile": [READER_HANDLE],
        "DWIGetMeasurementInfo": [READER_HANDLE, pointer(DWMeasurementInfo)],
        "DWIGetChannelListCount": [READER_HANDLE, pointer(ctypes.c_int)],
        "DWIGetChannelList": [READER_HANDLE, pointer(DWChannel)],
        "DWIGetChannelProps": [READER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                               pointer(ctypes.c_int)],
        "DWIGetScaledSamplesCount": [READER_HANDLE, ctypes.c_int, pointer(ctypes.c_longlong)],
        "DWIGetScaledSamples": [READER_HANDLE, ctypes.c_int, ctypes.c_longlong, ctypes.c_int,
                                pointer(ctypes.c_double), pointer(ctypes.c_double)],
        "DWGetLastStatus": [pointer(ctypes.c_int), ctypes.c_void_p, pointer(ctypes.c_int)],
    }
    for name, args in signatures.items():
        function = getattr(library, name)
        function.argtypes = args
        function.restype = ctypes.c_int
    return library


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
    if n > 2_147_483_647:
        raise ValueError("Channel exceeds the single-read limit; chunked reading is required")

    samples_c = (ctypes.c_double * n)()
    ch_type = get_channel_type(reader, ch)

    if ch_type not in (DWChannelType.DW_CH_TYPE_SYNC, DWChannelType.DW_CH_TYPE_ASYNC):
        raise ValueError(f"Unsupported channel type: {ch_type.name}")
    # Read native timestamps for both types to preserve channel alignment and
    # channels stored at a rate different from the measurement base rate.
    timestamps_c = (ctypes.c_double * n)()

    check_error(
        lib,
        lib.DWIGetScaledSamples(reader, ch.index, 0, n, samples_c, timestamps_c),
    )

    samples = np.ctypeslib.as_array(samples_c).astype(np.float64, copy=True)

    timestamps = np.ctypeslib.as_array(timestamps_c).astype(np.float64, copy=True)
    validate_signal(samples, timestamps)
    is_async = ch_type == DWChannelType.DW_CH_TYPE_ASYNC

    return samples, timestamps, is_async


def empty_row(filename: str, phase: str):
    return {
        "File": filename,
        "Phase": phase,
        "Current Source": "",
        "Current Channel": "",
        "CT Minimum Peak (A)": "",
        **dict.fromkeys(DETECTION_COLUMNS.values(), ""),
        "CT Peak (A)": "",
        "Current Selection Reason": "",
        "Voltage Channel": "",
        "Sample Rate (Hz)": "",
        "Start Time (s)": "",
        "End Time (s)": "",
        "Duration (s)": "",
        "Peak Current (kA)": "",
        "RMS Current (kA)": "",
        "I^2t (A^2 s x10^6)": "",
        "RMS Voltage (V)": "",
        "Voltage RMS Start Time (s)": "",
        "Voltage RMS End Time (s)": "",
        "Status": "",
    }


@dataclass
class PhaseAnalysis:
    """Results and original SI waveforms retained for a future review screen."""
    row: dict
    current_a: np.ndarray | None = None
    current_t: np.ndarray | None = None
    voltage_v: np.ndarray | None = None
    voltage_t: np.ndarray | None = None
    event: tuple[int, int] | None = None


def select_current_channel(ct, shunt, read_current, threshold_a):
    """Read CT first; a qualifying CT is never replaced merely for a missing event."""
    ct_peak = None
    if ct is None:
        reason = "CT_MISSING"
    else:
        try:
            data = read_current(ct)
            ct_peak = float(np.max(np.abs(data[0])))
            if ct_is_usable(data[0], threshold_a):
                return ct, "C.T.", data, "CT_MEETS_THRESHOLD", ct_peak
            reason = "CT_FLAT" if ct_peak == 0 else "CT_BELOW_THRESHOLD"
        except Exception as exc:
            reason = f"CT_READ_ERROR:{exc}"
    if shunt is not None:
        try:
            return shunt, "Shunt", read_current(shunt), reason, ct_peak
        except Exception as exc:
            reason += f";SHUNT_READ_ERROR:{exc}"
    else:
        reason += ";NO_SHUNT_CHANNEL"
    return None, "", None, reason, ct_peak


def analyse_open_file(reader: READER_HANDLE, path: Path, settings: ProcessingSettings) -> list[PhaseAnalysis]:
    measurement_info = DWMeasurementInfo(0, 0, 0, 0)
    check_error(lib, lib.DWIGetMeasurementInfo(reader, ctypes.byref(measurement_info)))
    base_rate = float(measurement_info.sample_rate)
    if not math.isfinite(base_rate) or base_rate <= 0:
        raise ValueError("Invalid measurement sample rate")
    channels = get_channels(reader)
    selections = {phase: choose_phase_channels(channels, phase) for phase in PHASES}

    # Cache candidate signals once, preserving native timestamp offsets between
    # channels. Use a single origin for all phases and both voltage and current.
    cache = {}
    for ct, shunt, voltage, _ in selections.values():
        for ch, quantity in ((ct, "current"), (shunt, "current"), (voltage, "voltage")):
            if ch is None or ch.index in cache:
                continue
            try:
                values, times, is_async = read_channel(reader, ch, base_rate)
                values = values * unit_factor_to_si(channel_unit(ch), quantity)
                validate_signal(values, times)
                cache[ch.index] = (values, times, is_async)
            except Exception as exc:
                cache[ch.index] = exc
    origin = min((data[1][0] for data in cache.values() if not isinstance(data, Exception)), default=0.0)
    cache = {index: data if isinstance(data, Exception) else (data[0], data[1] - origin, data[2])
             for index, data in cache.items()}

    def read_cached(ch):
        data = cache[ch.index]
        if isinstance(data, Exception):
            raise data
        return data

    results = []
    for phase in PHASES:
        row = empty_row(path.name, phase)
        row.update(settings_columns(settings))
        result = PhaseAnalysis(row)
        results.append(result)
        ct, shunt, voltage_ch, warnings = selections[phase]
        statuses = list(warnings)
        current_ch, source, current, reason, ct_peak = select_current_channel(
            ct, shunt, read_cached, settings.ct_min_peak_a)
        row["Current Selection Reason"] = reason
        row["CT Peak (A)"] = ct_peak if ct_peak is not None else ""
        row["Current Source"] = source
        if source == "Shunt":
            statuses.append("SHUNT_FALLBACK")
        if voltage_ch is not None:
            row["Voltage Channel"] = channel_name(voltage_ch)
            try:
                result.voltage_v, result.voltage_t, voltage_async = read_cached(voltage_ch)
            except Exception as exc:
                statuses.append(f"VOLTAGE_READ_ERROR:{exc}")
        else:
            statuses.append("NO_VOLTAGE_CHANNEL")
        if current_ch is None:
            statuses.append("NO_USABLE_CURRENT_CHANNEL")
            row["Status"] = ";".join(statuses)
            continue

        row["Current Channel"] = channel_name(current_ch)
        current_a, current_t, current_async = current
        result.current_a, result.current_t = current_a, current_t
        try:
            # Detection hold times require regularly spaced current samples.
            # Do not silently apply the base rate to asynchronous/gapped data.
            dt = np.diff(current_t)
            interval = float(np.median(dt)) if dt.size else 1.0 / base_rate
            if dt.size and not np.allclose(dt, interval, rtol=1e-5, atol=1e-9):
                raise ValueError("NONUNIFORM_CURRENT_TIMESTAMPS")
            sample_rate_hz = 1.0 / interval
            row["Sample Rate (Hz)"] = sample_rate_hz
            if not math.isclose(sample_rate_hz, EXPECTED_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1.0):
                statuses.append(f"UNEXPECTED_SAMPLE_RATE_{sample_rate_hz:g}Hz")
            event, event_status = detect_event(
                current_a, sample_rate_hz, settings.detection_config())
            if event is None:
                statuses.append(event_status)
                row["Status"] = ";".join(statuses)
                continue
            result.event = event
            if event_status != "OK":
                statuses.append(event_status)
            start, end = event
            if start == 0:
                statuses.append("EVENT_START_AT_RECORD_BOUNDARY")
            if end == current_a.size:
                statuses.append("EVENT_END_AT_RECORD_BOUNDARY")
            metrics = calculate_current_metrics(current_a, current_t, current_async, sample_rate_hz, start, end)
        except Exception as exc:
            statuses.append(f"CURRENT_ANALYSIS_ERROR:{exc}")
            row["Status"] = ";".join(statuses)
            continue

        row["Start Time (s)"] = metrics["start_time_s"]
        row["End Time (s)"] = metrics["end_time_s"]
        row["Duration (s)"] = metrics["duration_s"]
        row["Peak Current (kA)"] = metrics["peak_current_a"] / 1_000.0
        row["RMS Current (kA)"] = metrics["rms_current_a"] / 1_000.0
        row["I^2t (A^2 s x10^6)"] = metrics["i2t_a2s"] / 1_000_000.0

        if result.voltage_v is not None:
            try:
                row["RMS Voltage (V)"] = calculate_voltage_rms(
                    result.voltage_v, result.voltage_t, metrics["start_time_s"], base_rate, voltage_async)
                row["Voltage RMS Start Time (s)"] = float(result.voltage_t[0])
                row["Voltage RMS End Time (s)"] = metrics["start_time_s"]
            except ValueError as exc:
                statuses.append(str(exc))
        row["Status"] = ";".join(statuses) if statuses else "OK"
    return results


def process_open_file(reader: READER_HANDLE, path: Path, settings: ProcessingSettings) -> list[dict]:
    return [result.row for result in analyse_open_file(reader, path, settings)]


def process_file(reader: READER_HANDLE, path: Path, settings: ProcessingSettings) -> list[dict]:
    file_info = DWFileInfo(0, 0, 0)
    opened = False
    try:
        c_filename = ctypes.c_char_p(str(path).encode("utf-8"))
        check_error(lib, lib.DWIOpenDataFile(reader, c_filename, ctypes.byref(file_info)))
        opened = True
        return process_open_file(reader, path, settings)
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
    parser.add_argument("--ct-min-peak-a", type=float, help="Shared CT peak threshold in amps; saved for future runs")
    for field in fields(DetectionConfig):
        note = "; 0 disables duration filtering" if field.name == "minimum_event_ms" else ""
        parser.add_argument("--" + field.name.replace("_", "-"), type=float,
                            help=f"{DETECTION_COLUMNS[field.name]}; saved for future runs (default: {field.default}{note})")
    parser.add_argument("--settings", type=Path, default=default_settings_path(), help="Settings JSON path")
    args = parser.parse_args()
    try:
        settings = load_settings(args.settings)
        if settings is None and args.ct_min_peak_a is not None:
            settings = ProcessingSettings(args.ct_min_peak_a)
        if settings is not None:
            overrides = {field.name: getattr(args, field.name) for field in fields(ProcessingSettings)
                         if getattr(args, field.name) is not None}
            settings = replace(settings, **overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if settings is None:
        parser.error("No CT threshold configured. Supply --ct-min-peak-a AMPS on the first run.")

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    output_path = args.output.resolve() if args.output else folder / "dxd_summary.csv"
    files = find_dxd_files(folder, args.recursive)
    if not files:
        raise SystemExit(f"No .dxd files found in {folder}")

    if overrides:
        try:
            save_settings(settings, args.settings)
        except OSError as exc:
            parser.error(f"Cannot save settings: {exc}")
    print(f"CT minimum peak: {settings.ct_min_peak_a:g} A (settings: {args.settings})")
    print(f"Minimum event duration: {settings.minimum_event_ms:g} ms")
    lib = configure_library(load_library())
    reader = READER_HANDLE()
    check_error(lib, lib.DWICreateReader(ctypes.byref(reader)))

    rows = []
    try:
        for number, path in enumerate(files, start=1):
            print(f"[{number}/{len(files)}] {path.name}")
            try:
                rows.extend(process_file(reader, path, settings))
            except Exception as exc:
                # Keep the batch running and leave an auditable failure row for each phase.
                for phase in PHASES:
                    row = empty_row(path.name, phase)
                    row.update(settings_columns(settings))
                    row["Status"] = f"FILE_ERROR:{exc}"
                    rows.append(row)
    finally:
        check_error(lib, lib.DWIDestroyReader(reader))

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()

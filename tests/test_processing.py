"""Analytical checks, selection rules and optional native-reader integration."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1] / "DWDataReader" / "examples" / "Python"
sys.path.insert(0, str(PYTHON_DIR))
import dxd_analysis as analysis
from dxd_settings import ProcessingSettings, load_settings, save_settings
import process_dxd_folder as processor

FS = 20_000.0


class AnalysisTests(unittest.TestCase):
    def test_rectangular_event_has_exact_duration_rms_and_i2t(self):
        current = np.zeros(2000)
        current[400:1200] = -1000.0
        bounds, status = analysis.detect_event(current, FS)
        self.assertEqual(bounds, (400, 1200))
        self.assertEqual(status, "OK")
        metrics = analysis.calculate_current_metrics(current, np.arange(2000) / FS, False, FS, *bounds)
        self.assertAlmostEqual(metrics["duration_s"], 0.04)
        self.assertAlmostEqual(metrics["rms_current_a"], 1000.0)
        self.assertAlmostEqual(metrics["peak_current_a"], 1000.0)
        self.assertAlmostEqual(metrics["i2t_a2s"], 40_000.0)

    def test_multiple_events_select_largest_i2_content(self):
        current = np.zeros(5000)
        current[400:800] = 100
        current[2000:2400] = 200
        self.assertEqual(analysis.detect_event(current, FS), ((2000, 2400), "MULTIPLE_EVENTS"))

    def test_flat_short_and_invalid_signals(self):
        self.assertEqual(analysis.detect_event(np.zeros(1000), FS)[1], "NO_EVENT_DETECTED")
        self.assertEqual(analysis.moving_rms(np.zeros(5), 20).size, 5)
        current = np.zeros(2000)
        current[400:440] = 1000
        self.assertEqual(analysis.detect_event(current, FS)[1], "EVENT_SHORTER_THAN_MINIMUM")
        with self.assertRaises(ValueError):
            analysis.detect_event(np.array([0.0, np.nan]), FS)

    def test_async_rms_uses_same_time_weights_as_i2t(self):
        current = np.array([1.0, 3.0, 3.0])
        times = np.array([0.0, 0.001, 0.010])
        m = analysis.calculate_current_metrics(current, times, True, FS, 0, 2)
        self.assertAlmostEqual(m["duration_s"], 0.010)
        self.assertAlmostEqual(m["i2t_a2s"], 0.082)
        self.assertAlmostEqual(m["rms_current_a"], np.sqrt(8.2))

    def test_bad_timestamps_are_rejected(self):
        with self.assertRaises(ValueError):
            analysis.calculate_current_metrics(np.ones(3), np.array([0.0, 0.1, 0.1]), True, FS, 0, 3)

    def test_voltage_uses_all_samples_before_event_and_excludes_event(self):
        voltage = np.array([3.0, 4.0, 1000.0, 2000.0])
        times = np.arange(4) / FS
        self.assertAlmostEqual(analysis.calculate_voltage_rms(voltage, times, times[2], FS), np.sqrt(12.5))
        with self.assertRaisesRegex(ValueError, "NO_PRE_EVENT_VOLTAGE"):
            analysis.calculate_voltage_rms(voltage, times, 0.0, FS)

    def test_async_voltage_clips_last_interval(self):
        voltage = np.array([1.0, 3.0, 1000.0])
        times = np.array([0.0, 0.001, 0.010])
        self.assertAlmostEqual(analysis.calculate_voltage_rms(voltage, times, 0.005, FS, True), np.sqrt(7.4))


class SelectionTests(unittest.TestCase):
    def select(self, ct_values, threshold=200.0, shunt=True):
        signals = {"ct": (np.array(ct_values, dtype=float), None, False),
                   "shunt": (np.array([1000.0]), None, False)}
        read = Mock(side_effect=lambda channel: signals[channel])
        selected = processor.select_current_channel("ct" if ct_values is not None else None,
                                                    "shunt" if shunt else None, read, threshold)
        return selected, read

    def test_absolute_peak_equality_prefers_ct(self):
        selected, read = self.select([-200, 0, 100])
        self.assertEqual(selected[1], "C.T.")
        self.assertEqual(selected[3], "CT_MEETS_THRESHOLD")
        read.assert_called_once_with("ct")

    def test_missing_flat_and_below_threshold_fall_back(self):
        for values, reason in [(None, "CT_MISSING"), ([0, 0], "CT_FLAT"), ([199, -150], "CT_BELOW_THRESHOLD")]:
            with self.subTest(reason=reason):
                selected, _ = self.select(values)
                self.assertEqual(selected[1], "Shunt")
                self.assertEqual(selected[3], reason)

    def test_no_shunt_does_not_reuse_rejected_ct(self):
        selected, _ = self.select([1, 2], shunt=False)
        self.assertIsNone(selected[0])
        self.assertIn("NO_SHUNT_CHANNEL", selected[3])

    def test_si_units_and_zero_threshold(self):
        amps = np.array([0.2]) * processor.unit_factor_to_si("kA", "current")
        self.assertTrue(analysis.ct_is_usable(amps, 200.0))
        self.assertFalse(analysis.ct_is_usable(np.zeros(10), 0.0))

    def test_failed_ct_read_is_reported_and_shunt_is_used(self):
        def read(channel):
            if channel == "ct":
                raise ValueError("invalid samples")
            return np.ones(10), np.arange(10) / FS, False
        selected = processor.select_current_channel("ct", "shunt", read, 200)
        self.assertEqual(selected[1], "Shunt")
        self.assertIn("CT_READ_ERROR:invalid samples", selected[3])


class SettingsTests(unittest.TestCase):
    def test_unset_and_saved_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "settings.json"
            self.assertIsNone(load_settings(path))
            save_settings(ProcessingSettings(123.5), path)
            self.assertEqual(load_settings(path).ct_min_peak_a, 123.5)
            save_settings(ProcessingSettings(200.0), path)
            self.assertEqual(load_settings(path).ct_min_peak_a, 200.0)

    def test_invalid_settings_cannot_silently_change_selection(self):
        for value in [-1, float("nan"), float("inf"), "200", True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProcessingSettings(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"ct_min_peak_a": "invalid"}')
            with self.assertRaises(ValueError):
                load_settings(path)


class PipelineTests(unittest.TestCase):
    def test_phases_have_independent_windows_and_pre_event_voltage(self):
        channels = []
        data = {}
        times = 50.0 + np.arange(2400) / FS
        for i, phase in enumerate(processor.PHASES):
            start = 400 + i * 200
            current = np.zeros(2400)
            current[start:start + 400] = 1000
            for offset, name, unit, values in [(0, f"C.T. {phase}", "A", current),
                                               (1, f"T.X. {phase}", "V", np.arange(2400, dtype=float))]:
                ch = processor.DWChannel()
                ch.index = 2 * i + offset
                ch.name, ch.unit, ch.array_size = name.encode(), unit.encode(), 1
                channels.append(ch)
                data[ch.index] = (values, times.copy(), False)
        library = Mock()
        def measurement(reader, output):
            output._obj.sample_rate = FS
            return 0
        library.DWIGetMeasurementInfo.side_effect = measurement
        with patch.object(processor, "lib", library), patch.object(processor, "get_channels", return_value=channels), \
                patch.object(processor, "read_channel", side_effect=lambda reader, ch, rate: data[ch.index]):
            results = processor.analyse_open_file(None, Path("test.dxd"), ProcessingSettings(200))
        for i, result in enumerate(results):
            start = 400 + i * 200
            self.assertEqual(result.event, (start, start + 400))
            self.assertAlmostEqual(result.row["Start Time (s)"], start / FS)
            self.assertAlmostEqual(result.row["RMS Voltage (V)"], np.sqrt(np.mean(np.arange(start, dtype=float) ** 2)))
            self.assertEqual(result.row["Status"], "OK")
            self.assertIsNotNone(result.current_a)


@unittest.skipUnless(os.environ.get("DXD_TEST_DATA_DIR"), "Set DXD_TEST_DATA_DIR for native-reader checks")
class NativeReaderTests(unittest.TestCase):
    def test_attached_recordings(self):
        root = Path(os.environ["DXD_TEST_DATA_DIR"])
        processor.lib = processor.configure_library(processor.load_library())
        reader = processor.READER_HANDLE()
        processor.check_error(processor.lib, processor.lib.DWICreateReader(ctypes.byref(reader)))
        try:
            for name, source in [("8361_Cal3_1.dxd", "Shunt"), ("8361_Test13_1.dxd", "C.T."),
                                 ("8361_Test26_1.dxd", "Shunt"), ("8361_Test42_1.dxd", "Shunt")]:
                with self.subTest(file=name):
                    matches = list(root.glob("*" + name))
                    self.assertEqual(len(matches), 1)
                    rows = processor.process_file(reader, matches[0], ProcessingSettings(200))
                    self.assertEqual(len(rows), 3)
                    for row in rows:
                        self.assertEqual(row["Current Source"], source)
                        self.assertNotIn("ERROR", row["Status"])
                        if row["Duration (s)"] != "":
                            self.assertGreater(row["Duration (s)"], 0)
                            self.assertGreater(row["RMS Voltage (V)"], 0)
                            self.assertEqual(row["Voltage RMS End Time (s)"], row["Start Time (s)"])
        finally:
            processor.check_error(processor.lib, processor.lib.DWIDestroyReader(reader))


if __name__ == "__main__":
    unittest.main()

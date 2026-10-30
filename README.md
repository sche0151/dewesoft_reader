# Dewesoft Reader

Batch analysis of Dewesoft DXD recordings. The processing backend is being
prepared for a Windows desktop application; a GUI and packaged executable are
not included in this change.

## Run on Windows

Use 64-bit Python 3.10 or later with the supplied 64-bit reader DLL. From the
repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
$ctThresholdAmps = Read-Host "CT minimum peak current (amps)"
python DWDataReader/examples/Python/process_dxd_folder.py "D:\Measurements" --ct-min-peak-a $ctThresholdAmps -o "D:\Measurements\summary.csv"
```

Choose the threshold for your measurements. No default threshold is assumed.
The setting is shared by all three phases and saved in
`%APPDATA%\DewesoftReader\settings.json`. Subsequent runs can omit
`--ct-min-peak-a`. Pass it again to change the saved setting. `--settings PATH`
selects an alternative settings file, and `--recursive` includes subfolders.
Invalid settings produce an error instead of silently selecting a different
current source. A missing, negative or non-finite threshold is rejected; zero
is permitted, but an exactly zero CT signal is still rejected.

Minimum event duration is also a saved setting, shared across phases. Use
`--minimum-event-ms VALUE` to change it; for example, `--minimum-event-ms 2`
sets a 2 ms minimum. Omit the flag on later runs to reuse the saved value.
Changing either setting preserves the other. Existing settings files that
contain only the CT threshold retain the previous 3 ms minimum.

The duration must be finite and non-negative. A value of `0` disables only the
minimum-duration filter; the detector's other signal and hold-time criteria
still apply. The value is available through `ProcessingSettings` for the future
GUI settings panel, and each CSV row records `Minimum Event Duration (ms)`.

## Measurement rules

- For each phase, prefer the CT if its maximum absolute scaled current over the
  recording is at least the threshold in **amps**. Otherwise use the shunt.
  A missing/unreadable CT also permits fallback, with the reason recorded.
- A qualifying CT is retained even if event detection subsequently fails.
  This rule does not attempt to identify CT saturation, clipping or spikes.
- Each phase has its own event window based on its selected current channel.
  The existing RMS envelope and hold times locate candidate events; raw
  current threshold crossings refine the start and exclusive end indices.
  Metrics use original scaled samples, not the smoothed envelope.
- The saved minimum event duration controls which detected events qualify.
  Durations shorter than that minimum are flagged; a minimum between sample
  boundaries rounds up to the next sample, allowing for timestamp round-off.
- Current RMS, peak, duration and I²t are calculated within that event window.
- Voltage RMS uses all available voltage samples from the beginning of the
  voltage signal up to, but excluding, that phase's current-event start.
  If there are no pre-event samples, voltage RMS is blank and the row is flagged.
- Calibration files produce measurements using the same rules; they do not
  generate or apply calibration correction factors.

Native timestamps retain channel offsets. Exported times are relative to the
earliest successfully read candidate current/voltage sample in that recording.
Current detection uses the channel's observed sample interval rather than
assuming the file's base sample rate applies to every channel. Nonuniform
current timestamps are flagged for review rather than silently analysed as
uniform. The metric helpers support time-weighted RMS and I²t for irregular
timestamps, but automatic event detection still requires uniform current data.

CSV results include the threshold, measured CT peak, current source, selection
reason, event start/end and voltage RMS interval. `RMS Voltage (V)` now means
**pre-event** RMS; it is not directly comparable to older exports that used
voltage during the current event. Existing result CSVs are historical outputs,
not verified reference measurements, and are left unchanged.

## Structure

- `dxd_analysis.py`: pure NumPy detection and metric functions.
- `dxd_settings.py`: validated, atomically saved user preferences.
- `process_dxd_folder.py`: reader bindings, channel selection and batch CSV CLI.
  `analyse_open_file` also returns `PhaseAnalysis` objects containing original
  SI waveforms and event indices for a future GUI review screen. The current
  native-reader integration is sequential; do not share its global library
  state or reader handles across concurrent GUI workers.

These files are under `DWDataReader/examples/Python/` alongside the vendor
reader module and native libraries.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The optional integration check needs the four supplied recordings:
`8361_Cal3_1.dxd`, `8361_Test13_1.dxd`, `8361_Test26_1.dxd` and
`8361_Test42_1.dxd` (upload filename prefixes are accepted).

```powershell
$env:DXD_TEST_DATA_DIR = "D:\SampleRecordings"
python -m unittest discover -s tests -v
```

The tests use **200 A only as a test fixture**, not an application default or a
recommended operating threshold. Checks cover threshold equality and fallback,
settings persistence, independent phase windows, pre-event voltage selection,
irregular time weights, and a known 40 ms/1,000 A rectangular pulse with an
expected I²t of 40,000 A²s.

The native integration checks ran on Linux with the bundled library; Windows
DLL execution and future executable packaging still need Windows validation.
The supplied files are smoke-test inputs, not independently verified numerical
references. Test13 blue is flagged `EVENT_SHORTER_THAN_MINIMUM` at the default
3 ms minimum; the integration test confirms it is included when the saved
minimum is changed to 2 ms. These values demonstrate the setting's effect,
not a recommendation for the appropriate measurement rule.

## Next desktop milestone

Add a single-file review screen with channel selection, current/voltage plots,
editable event boundaries, shared CT threshold and minimum-duration fields, recalculation and
CSV export. Validate representative results against independently checked
measurements before packaging the Windows executable.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from zoneinfo import ZoneInfo
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def format_tick(value, _position):
    local_time = mdates.num2date(value, tz=sydney)
    return local_time.strftime("%M:%S.%f")[:-3]

sydney = ZoneInfo("Australia/Sydney")

ROOT = "C:/Users/shang/Desktop/Ausnet - P02616733 - 252108466/Dewesoft"

MAXVAL_THRESHOLD = 1

for csv_path in sorted(Path(ROOT).glob("**.csv")):

    df = pd.read_csv(csv_path)
    # Shift time domain to time relative to t=0 and to ms
    x = pd.to_datetime(df["Time (Australia/Sydney)"], utc=True).dt.tz_convert(sydney)

    # Checking which phases were recorded
    channels = df.columns.tolist()
    shunt_switch = []
    if any("80kA" in channel for channel in channels):
        shunt_switch = "80kA"
    elif any("40kA" in channel for channel in channels):
        shunt_switch = "40kA"
    else:
        print(f"Skipping {csv_path.name}: no shunt channel")
        continue

    phases = ["Red", "White", "Blue"]
    recorded_phases = []
    for i, phase in enumerate(phases):
        if any(phase in channel for channel in channels):
            recorded_phases.append(phase)

    valid_phases = [
        phase for phase in recorded_phases
        if df[f"{shunt_switch} Shunt {phase} [kA]"].max() >= MAXVAL_THRESHOLD
    ]

    if not valid_phases:
        print(f"Skipping {csv_path.name}: no phase reached the threshold")
        continue

    ncols = len(valid_phases)
    fig = plt.figure(figsize=(ncols*12, 10), layout="constrained")

    for col, phase in enumerate(valid_phases, start=1):
        current_col = f"{shunt_switch} Shunt {phase} [kA]"
        voltage_col = f"T.X. {phase} [V]"

        maxval = df[f"{shunt_switch} Shunt {phase} [kA]"].max()

        ax_current = fig.add_subplot(2, ncols, col)
        ax_current.plot(x, df[current_col], linewidth=0.8)
        ax_current.set(title=current_col, ylabel="Current (kA)")

        ax_voltage = fig.add_subplot(2, ncols, ncols + col, sharex=ax_current)
        ax_voltage.plot(x, df[voltage_col], linewidth=0.8)
        ax_voltage.set(title=voltage_col, ylabel="Voltage (V)")
        ax_voltage.xaxis.set_major_formatter(FuncFormatter(format_tick))
        ax_voltage.set_xlabel("Sydney time (MM:SS.mmm)")

        for ax in (ax_current, ax_voltage):
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.4)

    # plt.tight_layout()
    output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path.name}")
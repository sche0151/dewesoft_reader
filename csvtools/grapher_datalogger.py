import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from zoneinfo import ZoneInfo
from pathlib import Path
from matplotlib.ticker import FuncFormatter

ROOT = "C:/Users/shang/Desktop/Ausnet - P02616733 - 252108466/Datalogger/test"
HEATING_THRESHOLD_C = 40.0

for csv_path in sorted(Path(ROOT).glob("**.csv")):

    df = pd.read_csv(csv_path, skiprows=11)
    channels = [col for col in df.columns if col.endswith("[C]")]
    temps = df[channels].apply(pd.to_numeric, errors="coerce")
    clock = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")

    crossed = temps.ge(HEATING_THRESHOLD_C).any(axis=1)
    if not crossed.any():
        print(f"Skipping {csv_path.name}: no temperature channel reached 40 °C")
        continue

    trigger_time = clock.loc[crossed].iloc[0]
    start = trigger_time - pd.Timedelta(milliseconds=300)
    end = trigger_time + pd.Timedelta(milliseconds=9500)
    keep = clock.between(start, end)

    x = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")

    fig, axes = plt.subplots(
        len(channels), 1, sharex=True, squeeze=False,
        figsize=(12, 3 * len(channels)), layout="constrained",
    )

    for ax, channel in zip(axes[:, 0], channels):
        ax.plot(clock.loc[keep], temps.loc[keep, channel], linewidth=0.8)
        ax.set(title=channel, ylabel="Temperature (°C)")
        ax.grid(True, linewidth=0.5, alpha=0.4)
        lowtemp = temps.loc[keep, channel].iloc[0] * 0.5
        hightemp = temps.loc[keep, channel].iloc[-1] * 1.1
        ax.set_ylim(lowtemp, hightemp)

    bottom = axes[-1, 0]
    bottom.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: mdates.num2date(value, tz="UTC").strftime("%M:%S.%f")[:-3]
    ))
    bottom.set_xlabel("Recorded time (MM:SS.mmm)")
    fig.savefig(csv_path.with_suffix(".png"), dpi=150)
    print(f"Saved {csv_path.with_suffix(".png")}")
    plt.close(fig)
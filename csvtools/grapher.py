import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

ROOT = "C:/Users/shang/Desktop/Ausnet - P02616733 - 252108466/Dewesoft"


df = pd.read_csv(f"{ROOT}/Test1_1.csv")

# Shift time domain to time relative to t=0 and to ms
x = (df["Time (s)"] - df["Time (s)"][0])*1000

MAXVAL_THRESHOLD = 1

# Checking which phases were recorded
channels = df.columns.tolist()
shunt_switch = []
if any("80kA" in channel for channel in channels):
    shunt_switch = "80kA"
elif any("40kA" in channel for channel in channels):
    shunt_switch = "40kA"
else:
    print("No shunt channels")

phases = ["Red", "White", "Blue"]
recorded_phases = []
for i, phase in enumerate(phases):
    if any(phase in channel for channel in channels):
        recorded_phases.append(phase)

valid_phases = [
    phase for phase in recorded_phases
    if df[f"{shunt_switch} Shunt {phase} [kA]"].max() >= MAXVAL_THRESHOLD
]

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
    ax_voltage.set(title=voltage_col, xlabel="Time (ms)", ylabel="Voltage (V)")

    for ax in (ax_current, ax_voltage):
        ax.set_axisbelow(True)
        ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.4)

# plt.tight_layout()
plt.show()
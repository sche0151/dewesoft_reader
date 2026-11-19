import ctypes as ct
from pathlib import Path
import numpy as np
import pandas as pd
from DWDataReaderHeader import (
    load_library, READER_HANDLE, DWFileInfo, DWChannel,
    decode_bytes, check_error,
)

FOLDER = Path(r"D:/Ausnet - P02616733 - 252108466/Dewesoft")
lib = load_library()
reader = READER_HANDLE()

def call(function, *args):
    check_error(lib, function(reader, *args))

check_error(lib, lib.DWICreateReader(ct.byref(reader)))
try:
    for path in sorted(FOLDER.iterdir()):
        if path.suffix.lower() != ".dxd":
            continue

        info = DWFileInfo()
        call(lib.DWIOpenDataFile, str(path).encode(), ct.byref(info))
        try:
            count = ct.c_int()
            call(lib.DWIGetChannelListCount, ct.byref(count))
            channels = (DWChannel * count.value)()
            call(lib.DWIGetChannelList, channels)
            frames = []

            for ch in channels:
                n = ct.c_longlong()
                call(lib.DWIGetScaledSamplesCount, ch.index, ct.byref(n))
                if not n.value:
                    continue

                values = (ct.c_double * (n.value * ch.array_size))()
                times = (ct.c_double * n.value)()
                call(lib.DWIGetScaledSamples, ch.index,
                     ct.c_longlong(0), n, values, times)

                name = f"{decode_bytes(ch.name)} [{decode_bytes(ch.unit)}]"
                columns = [
                    name if ch.array_size == 1 else f"{name}[{i}]"
                    for i in range(ch.array_size)
                ]
                frames.append(pd.DataFrame(
                    np.ctypeslib.as_array(values).reshape(n.value, ch.array_size),
                    index=np.ctypeslib.as_array(times).copy(),
                    columns=columns,
                ))

            if frames:
                output = path.with_suffix(".csv")
                pd.concat(frames, axis=1).sort_index().to_csv(
                    output, index_label="Time (s)"
                )
                print(f"Exported: {output.name}")
        finally:
            call(lib.DWICloseDataFile)
finally:
    call(lib.DWIDestroyReader)
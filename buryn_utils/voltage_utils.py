import numpy as np
import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt
import csv
import re
from pathlib import Path
import numpy as np

_STEP_RE = re.compile(r"_step(\d+)\.txt$", re.IGNORECASE)

def load_voltage_recording(directory: Path, pattern: str = "*_step*.txt"):
    """
    Reads per-timestep voltage text files in `directory`.

    Each file contains a header: neuron_id, voltage
    Each file corresponds to one timestep and is named like: *_stepX.txt

    Returns:
        steps: (T,) int array of timesteps (sorted ascending)
        voltages: (T, N) float array of voltages aligned to neuron_ids
        neuron_ids: list[int] of neuron ids (column order for voltages)
    """
    directory = Path(directory)

    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No voltage files found in {directory} matching {pattern!r}")

    # Parse (step, path), filter only those matching *_stepX.txt
    step_paths = []
    for p in files:
        m = _STEP_RE.search(p.name)
        if m:
            step_paths.append((int(m.group(1)), p))
    if not step_paths:
        raise FileNotFoundError(f"No files matched '*_stepX.txt' in {directory}")

    step_paths.sort(key=lambda t: t[0])

    steps = []
    voltages_rows = []
    neuron_ids = None
    id_to_col = None

    for step, path in step_paths:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{path} is missing a header")

            header_norm = [h.strip().lower() for h in header]
            try:
                id_idx = header_norm.index("neuron_id")
                v_idx = header_norm.index("voltage")
            except ValueError as e:
                raise ValueError(
                    f"{path} header must contain 'neuron_id' and 'voltage' (got {header})"
                ) from e

            pairs = []
            for row in reader:
                if not row:
                    continue
                # allow blank/whitespace lines
                if all(not c.strip() for c in row):
                    continue
                nid = int(row[id_idx])
                v = float(row[v_idx])
                pairs.append((nid, v))

        if not pairs:
            raise ValueError(f"{path} contains no data rows")

        # Establish neuron id ordering from the first file (sorted by neuron_id)
        if neuron_ids is None:
            neuron_ids = [nid for nid, _ in sorted(pairs, key=lambda t: t[0])]
            id_to_col = {nid: i for i, nid in enumerate(neuron_ids)}

        # Fill a row aligned to neuron_ids
        row_voltages = np.full(len(neuron_ids), np.nan, dtype=float)
        for nid, v in pairs:
            if nid not in id_to_col:
                raise ValueError(
                    f"{path} contains neuron_id={nid} not present in first timestep file"
                )
            row_voltages[id_to_col[nid]] = v

        # Optional: ensure no missing voltages for known neuron_ids
        if np.isnan(row_voltages).any():
            missing = [neuron_ids[i] for i in np.where(np.isnan(row_voltages))[0]]
            raise ValueError(f"{path} is missing voltages for neuron_ids: {missing}")

        steps.append(step)
        voltages_rows.append(row_voltages)

    return np.array(steps, dtype=int), np.vstack(voltages_rows), neuron_ids


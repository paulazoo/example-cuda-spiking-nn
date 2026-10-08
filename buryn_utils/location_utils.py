import numpy as np
import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt

def load_neuron_locations(filepath):
    with filepath.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)

        neuron_ids = []
        neuron_locations = []
        if not header:
            raise ValueError("Voltage file is missing a header")


        for row in reader:
            if not row or not row[0].strip():
                continue
            neuron_ids.append(int(row[0]))
            neuron_locations.append([float(coord) for coord in row[1:]])

    return np.array(neuron_ids), np.array(neuron_locations)
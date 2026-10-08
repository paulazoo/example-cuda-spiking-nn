import numpy as np
import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt

def load_spikes(filepath):
    spike_steps = []
    neuron_ids = []
    with filepath.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            step_text = row[0].strip()
            if not step_text:
                continue
            step = int(step_text)
            neuron_column = row[1].strip() if len(row) > 1 else ""
            if neuron_column:
                ids = [int(x) for x in neuron_column.split()]
                spike_steps.extend([step] * len(ids))
                neuron_ids.extend(ids)
    return np.array(spike_steps, dtype=int), np.array(neuron_ids, dtype=int)


def build_spike_matrix(spike_steps: np.ndarray, neuron_ids: np.ndarray, step_max: int, num_neurons: int = None) -> np.ndarray:
    # matrix.shape = (num_neurons, step_max + 1)
    if num_neurons is None:
        num_neurons = int(neuron_ids.max()) + 1 if neuron_ids.size else 0
    matrix = np.zeros((num_neurons, step_max + 1), dtype=bool)
    if matrix.size == 0:
        return matrix

    valid = (spike_steps >= 0) & (spike_steps <= step_max)
    matrix[neuron_ids[valid], spike_steps[valid]] = True
    return matrix


def plot_spike_recording(spike_matrix, *, title, color='black', step_min=None, step_max=None, ax=None):
    """Plot spike recordings for a neuron group using a binary spike matrix.

    spike_matrix is a 2D array of shape (num_neurons, num_steps) where a True
    entry indicates a spike for a neuron at a simulation step.
    step_min and step_max define the displayed simulation step range and
    default to the bounds of the provided matrix.
    """
    if step_min is None:
        step_min = 0
    if step_max is None:
        step_max = spike_matrix.shape[1] - 1

    capped_step_min = max(step_min, 0)
    capped_step_max = min(step_max, spike_matrix.shape[1] - 1)

    step_slice = slice(capped_step_min, capped_step_max + 1)
    neuron_ids, steps = np.nonzero(spike_matrix[:, step_slice])
    steps = steps + capped_step_min

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.scatter(steps, neuron_ids, s=10, marker='.', color=color, linewidths=0)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Neuron ID')
    ax.set_title(title)
    ax.set_ylim(-1, spike_matrix.shape[0])
    ax.set_xlim(step_min, step_max)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig, ax



def plot_smoothed_spike_rates(spike_matrix, title, color_map, smoothing_window, step_min=None, step_max=None, ax=None):
    counts = spike_matrix.astype(float)
    window = np.ones(smoothing_window) / smoothing_window
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=1, arr=counts)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        smoothed,
        aspect='auto',
        origin='lower',
        extent=[0, spike_matrix.shape[1] - 1, -0.5, spike_matrix.shape[0] - 0.5],
        cmap=color_map
    )
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Neuron ID')
    ax.set_title(f"{title} (smoothed firing rate)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average spikes_utils per step')
    fig.tight_layout()
    ax.set_xlim(step_min, step_max)
    ax.set_ylim(-1, spike_matrix.shape[0])
    return fig, ax

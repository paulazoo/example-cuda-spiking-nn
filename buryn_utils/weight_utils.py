import csv
import numpy as np
import warnings

def load_weights(weights_filepath):
    weights = []
    with weights_filepath.open() as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if not header:
            raise ValueError(
                f'Weights file {weights_filepath} is missing a header'
            )

        for line_number, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            if len(row) < 3:
                # Large recordings can occasionally end with one truncated token if
                # the simulation/process exits while a row is being written.
                if len(row) == 1 and row[0].strip().isdigit():
                    warnings.warn(
                        "Skipping truncated weights row in "
                        f"{weights_filepath} at line {line_number}: {row!r}",
                        RuntimeWarning,
                    )
                    continue
                raise ValueError(
                    "Malformed weights row in "
                    f"{weights_filepath} at line {line_number}: expected 3 columns "
                    f"(pre_neuron_id, post_neuron_id, weight), got {len(row)} -> {row!r}"
                )
            weights.append((int(row[0]), int(row[1]), float(row[2])))
    return weights


def build_segments_incoming(weights, target_id, source_locations, target_location):
    segments = []
    weight_values = []
    for pre_id, post_id, weight in weights:
        if post_id != target_id or weight == 0.0:
            continue
        if pre_id not in source_locations:
            print(
                f'Warning: presynaptic neuron ID {pre_id} not found in presynaptic locations.'
            )
            continue
        segments.append([source_locations[pre_id], target_location])
        weight_values.append(weight)
    return segments, np.array(weight_values)

def build_segments_outgoing(weights, target_id, target_location, post_locations):
    segments = []
    weight_values = []
    for pre_id, post_id, weight in weights:
        if pre_id != target_id or weight == 0.0:
            continue
        if post_id not in post_locations:
            print(
                f'Warning: postsynaptic neuron ID {post_id} not found in postsynaptic locations.'
            )
            continue
        segments.append([target_location, post_locations[post_id]])
        weight_values.append(weight)
    return segments, np.array(weight_values)


def build_weighted_locations_incoming(weights, target_id, source_locations):
    locations = []
    weight_values = []
    for pre_id, post_id, weight in weights:
        if post_id != target_id or weight == 0.0:
            continue
        if pre_id not in source_locations:
            print(
                f'Warning: presynaptic neuron ID {pre_id} not found in presynaptic locations.'
            )
            continue
        locations.append(source_locations[pre_id])
        weight_values.append(weight)
    return np.array(locations), np.array(weight_values)


def build_weighted_locations_outgoing(weights, target_id, post_locations):
    locations = []
    weight_values = []
    for pre_id, post_id, weight in weights:
        if pre_id != target_id or weight == 0.0:
            continue
        if post_id not in post_locations:
            print(
                f'Warning: postsynaptic neuron ID {post_id} not found in postsynaptic locations.'
            )
            continue
        locations.append(post_locations[post_id])
        weight_values.append(weight)
    return np.array(locations), np.array(weight_values)

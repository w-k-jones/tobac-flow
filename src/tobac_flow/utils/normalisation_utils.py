import numpy as np


def linearise_field(field, lower_threshold, upper_threshold):
    if lower_threshold > upper_threshold:
        upper_threshold, lower_threshold = lower_threshold, upper_threshold
    return np.maximum(
        np.minimum((field - lower_threshold) / (upper_threshold - lower_threshold), 1),
        0,
    )

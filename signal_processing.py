import numpy as np

from config import (
    BIN_SIZE_M,
    GUARD_CELLS,
    TRAINING_CELLS,
    THRESHOLD_FACTOR,
)


def iq_to_amplitude(iq):
    """Converte il segnale complesso I/Q in ampiezza."""
    return np.abs(iq)


def remove_static_background(amplitude, background):
    """Sottrae il fondo statico e limita i valori negativi a zero."""
    result = amplitude - background
    result[result < 0] = 0
    return result


def cfar_ca(
    signal,
    guard_cells=GUARD_CELLS,
    training_cells=TRAINING_CELLS,
    threshold_factor=THRESHOLD_FACTOR,
):
    """CA-CFAR semplificato su un vettore di ampiezze."""

    detections = np.zeros(len(signal), dtype=bool)

    for i in range(len(signal)):

        left_start = i - guard_cells - training_cells
        left_end = i - guard_cells

        right_start = i + guard_cells + 1
        right_end = i + guard_cells + training_cells + 1

        # Ai bordi non ci sono abbastanza celle di training.
        if left_start < 0:
            continue

        if right_end > len(signal):
            continue

        training_left = signal[left_start:left_end]
        training_right = signal[right_start:right_end]

        training_values = np.concatenate(
            (training_left, training_right)
        )

        noise_estimate = np.mean(training_values)
        threshold = threshold_factor * noise_estimate

        if signal[i] > threshold:
            detections[i] = True

    return detections


def extract_peaks(signal, detections):
    """Estrae i massimi locali tra le celle segnalate dal CFAR."""

    peaks = []

    for i in range(1, len(signal) - 1):

        if not detections[i]:
            continue

        if (
            signal[i] >= signal[i - 1]
            and signal[i] >= signal[i + 1]
        ):
            peaks.append(i)

    return peaks


def sub_bin_interpolate(signal, idx):
    """
    Interpolazione parabolica a tre punti.
    Il risultato è espresso in unità di bin.
    """

    if idx <= 0 or idx >= len(signal) - 1:
        return float(idx)

    y1 = signal[idx - 1]
    y2 = signal[idx]
    y3 = signal[idx + 1]

    denominator = y1 - 2.0 * y2 + y3

    if denominator == 0:
        return float(idx)

    delta = 0.5 * (y1 - y3) / denominator

    return idx + delta


def bins_to_distance(sub_bin):
    """Converte la posizione frazionaria in bin in metri."""
    return sub_bin * BIN_SIZE_M


def process_frame(iq, background):
    """
    Elabora un frame I/Q.

    Restituisce:
      detections_m, amplitude, signal, cfar_detections
    """

    amplitude = iq_to_amplitude(iq)

    signal = remove_static_background(
        amplitude,
        background,
    )

    cfar_detections = cfar_ca(signal)

    peaks = extract_peaks(
        signal,
        cfar_detections,
    )

    detections_m = []

    for peak in peaks:

        sub_bin = sub_bin_interpolate(signal, peak)
        distance_m = bins_to_distance(sub_bin)

        detections_m.append(distance_m)

    return (
        detections_m,
        amplitude,
        signal,
        cfar_detections,
    )

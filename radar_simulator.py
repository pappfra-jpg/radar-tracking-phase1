import numpy as np

from config import (
    N_BINS,
    BIN_SIZE_M,
    NOISE_STD,
    TARGET_AMPLITUDE,
    STATIC_BACKGROUND,
)


def make_iq_frame(target_positions):
    """
    Genera un frame I/Q sintetico.

    target_positions contiene le distanze dei target in metri.
    Un valore None indica un target non visibile in questo frame.
    """

    I = np.random.normal(0.0, NOISE_STD, N_BINS)
    Q = np.random.normal(0.0, NOISE_STD, N_BINS)

    iq = I + 1j * Q

    # Aggiunge il fondo statico simulato.
    iq += STATIC_BACKGROUND

    for target_position in target_positions:

        if target_position is None:
            continue

        bin_position = target_position / BIN_SIZE_M
        center_bin = int(np.floor(bin_position))

        # Il segnale del target si distribuisce su tre bin.
        for offset in [-1, 0, 1]:

            bin_index = center_bin + offset

            if bin_index < 0 or bin_index >= N_BINS:
                continue

            distance_from_center = bin_index - bin_position

            weight = np.exp(
                -0.5 * distance_from_center ** 2
            )

            phase = np.random.uniform(0, 2 * np.pi)

            target_signal = (
                TARGET_AMPLITUDE
                * weight
                * np.exp(1j * phase)
            )

            iq[bin_index] += target_signal

    return iq

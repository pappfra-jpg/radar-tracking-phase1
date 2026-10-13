import numpy as np
import matplotlib.pyplot as plt

from config import (
    DT,
    N_FRAMES,
    TARGET_POSITIONS,
    STATIC_BACKGROUND,
)

from radar_simulator import make_iq_frame
from signal_processing import process_frame
from kalman import Kalman1D
from data_association import (
    associate_tracks,
    build_cost_matrix,
)


# ============================================================
# GENERAZIONE DEI FRAME RADAR
# ============================================================

iq_frames = []

for frame_index in range(N_FRAMES):

    target_positions = TARGET_POSITIONS[frame_index]
    iq = make_iq_frame(target_positions)

    iq_frames.append(iq)


# ============================================================
# SIGNAL PROCESSING
# ============================================================

all_detections = []

for iq in iq_frames:

    detections, _, _, _ = process_frame(
        iq,
        STATIC_BACKGROUND,
    )

    all_detections.append(detections)


# ============================================================
# INIZIALIZZAZIONE TRACK
# ============================================================

initial_detections = all_detections[0]

tracks = []

for detection in initial_detections:
    tracks.append(Kalman1D(detection))

if len(tracks) < 2:
    raise RuntimeError(
        "Nel primo frame sono state inizializzate meno di due tracce. "
        "Prova a rieseguire la simulazione oppure controlla i parametri CFAR."
    )


# ============================================================
# STORICO
# ============================================================

true_history = [[], []]
predicted_history = [[], []]
detection_history = [[], []]


# ============================================================
# CICLO PRINCIPALE
# ============================================================

for frame_index in range(N_FRAMES):

    detections = all_detections[frame_index]

    true_target_1 = TARGET_POSITIONS[frame_index][0]
    true_target_2 = TARGET_POSITIONS[frame_index][1]

    print()
    print("=" * 75)
    print(f"FRAME {frame_index + 1}")
    print("=" * 75)

    # --------------------------------------------------------
    # POSIZIONI VERE
    # --------------------------------------------------------

    print()
    print("Posizioni vere:")

    if true_target_1 is None:
        print("  Target 1: NON VISIBILE")
    else:
        print(f"  Target 1: {true_target_1:.3f} m")

    if true_target_2 is None:
        print("  Target 2: NON VISIBILE")
    else:
        print(f"  Target 2: {true_target_2:.3f} m")

    true_history[0].append(true_target_1)
    true_history[1].append(true_target_2)

    # --------------------------------------------------------
    # DETECTION RADAR
    # --------------------------------------------------------

    print()
    print("Detection radar:")

    if len(detections) == 0:
        print("  NESSUNA DETECTION")
    else:
        for detection_index, detection in enumerate(detections):
            print(f"  D{detection_index}: {detection:.4f} m")

    # --------------------------------------------------------
    # KALMAN PREDICTION
    # --------------------------------------------------------

    for track_index, track in enumerate(tracks):

        prediction = track.predict()

        predicted_history[track_index].append(
            prediction[0, 0]
        )

        # Nessuna misura associata finché non viene eseguita
        # la data association di questo frame.
        detection_history[track_index].append(None)

    print()
    print("Prediction Kalman:")

    for track_index, track in enumerate(tracks):
        print(
            f"  T{track_index}: "
            f"pos={track.x[0, 0]:.4f} m, "
            f"vel={track.x[1, 0]:.4f} m/s"
        )

    # --------------------------------------------------------
    # MATRICE MAHALANOBIS
    # --------------------------------------------------------

    cost_matrix = build_cost_matrix(tracks, detections)

    print()
    print("Matrice Mahalanobis^2:")

    if len(detections) == 0:
        print("  Nessuna detection")
    else:
        header = "          "

        for detection_index in range(len(detections)):
            header += f"D{detection_index:<9}"

        print(header)

        for track_index in range(len(tracks)):

            row = f"T{track_index}:   "

            for detection_index in range(len(detections)):

                value = cost_matrix[
                    track_index,
                    detection_index,
                ]

                if np.isinf(value):
                    row += f"{'INF':<10}"
                else:
                    row += f"{value:<10.3f}"

            print(row)

    # --------------------------------------------------------
    # DATA ASSOCIATION
    # --------------------------------------------------------

    matches, unmatched_tracks, unmatched_detections = (
        associate_tracks(tracks, detections)
    )

    print()
    print("Associazioni Hungarian:")

    if len(matches) == 0:
        print("  NESSUNA ASSOCIAZIONE")
    else:
        for track_index, detection_index in matches:

            cost = cost_matrix[
                track_index,
                detection_index,
            ]

            print(
                f"  T{track_index} <-> D{detection_index}   "
                f"Mahalanobis^2={cost:.3f}"
            )

    # --------------------------------------------------------
    # KALMAN UPDATE
    # --------------------------------------------------------

    print()
    print("Update Kalman:")

    for track_index, detection_index in matches:

        track = tracks[track_index]
        measurement = detections[detection_index]

        # Salva la detection radar associata alla traccia.
        detection_history[track_index][-1] = measurement

        prediction_position = track.x[0, 0]

        innovation, S = track.innovation_data(measurement)

        innovation_value = innovation[0, 0]
        S_value = S[0, 0]

        mahalanobis = innovation_value ** 2 / S_value

        K = (
            track.P
            @ track.H.T
            @ np.linalg.inv(S)
        )

        K_position = K[0, 0]
        K_velocity = K[1, 0]

        updated_state = track.update(measurement)

        print()
        print(f"  T{track_index} <- D{detection_index}")
        print(f"    z                = {measurement:.4f} m")
        print(f"    prediction       = {prediction_position:.4f} m")
        print(f"    innovation       = {innovation_value:.4f} m")
        print(f"    S                = {S_value:.6f}")
        print(f"    Mahalanobis^2    = {mahalanobis:.3f}")
        print(f"    K_position       = {K_position:.4f}")
        print(f"    K_velocity       = {K_velocity:.4f}")
        print(f"    posizione finale = {updated_state[0, 0]:.4f} m")
        print(f"    velocita finale  = {updated_state[1, 0]:.4f} m/s")

    # --------------------------------------------------------
    # TRACK SENZA DETECTION
    # --------------------------------------------------------

    for track_index in unmatched_tracks:

        track = tracks[track_index]

        print()
        print(f"  T{track_index}: NESSUNA DETECTION")
        print("    prediction mantenuta")
        print(f"    posizione = {track.x[0, 0]:.4f} m")
        print(f"    velocita  = {track.x[1, 0]:.4f} m/s")

    # --------------------------------------------------------
    # DETECTION NON ASSOCIATE
    # --------------------------------------------------------

    for detection_index in unmatched_detections:
        print()
        print(f"  D{detection_index}: NON ASSOCIATA")


# ============================================================
# FINE SIMULAZIONE
# ============================================================

print()
print("=" * 75)
print("FINE SIMULAZIONE")
print("=" * 75)


# ============================================================
# PLOT SEPARATI PER TRACCIA
# ============================================================

time = np.arange(N_FRAMES) * DT

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 9),
    sharex=True,
)

track_names = [
    "Track 1 - Target 1",
    "Track 2 - Target 2",
]

for track_index, ax in enumerate(axes):

    # Posizione reale del target.
    ax.plot(
        time,
        true_history[track_index],
        "o-",
        label="Real position",
    )

    # Detection radar effettivamente associata alla traccia.
    # I None generano dei vuoti nei frame senza misura associata.
    ax.plot(
        time,
        detection_history[track_index],
        "s",
        markersize=7,
        label="Detection radar associated",
    )

    # Predizione Kalman prima dell'aggiornamento con la misura.
    ax.plot(
        time,
        predicted_history[track_index],
        "x--",
        label=" Kalman prediction",
    )

    ax.set_title(track_names[track_index])
    ax.set_ylabel("Range [m]")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")

fig.suptitle(
    "Radar Tracking: Ground Truth, Detections and Kalman Prediction"
)

plt.tight_layout()
plt.show()

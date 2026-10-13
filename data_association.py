import numpy as np

from scipy.optimize import linear_sum_assignment

from config import MAHALANOBIS_GATE


def mahalanobis_distance_squared(track, measurement):
    """Distanza di Mahalanobis al quadrato tra track e misura."""

    innovation, S = track.innovation_data(measurement)

    d_squared = (
        innovation.T
        @ np.linalg.inv(S)
        @ innovation
    )

    return float(d_squared[0, 0])


def build_cost_matrix(tracks, detections):
    """
    Costruisce la matrice dei costi.

    Le associazioni fuori dal gate ricevono costo infinito.
    """

    n_tracks = len(tracks)
    n_detections = len(detections)

    cost_matrix = np.full(
        (n_tracks, n_detections),
        np.inf,
    )

    for track_index, track in enumerate(tracks):

        for detection_index, measurement in enumerate(detections):

            d_squared = mahalanobis_distance_squared(
                track,
                measurement,
            )

            if d_squared < MAHALANOBIS_GATE:
                cost_matrix[track_index, detection_index] = d_squared

    return cost_matrix


def associate_tracks(tracks, detections):
    """
    Associa le tracce alle detection con l'algoritmo ungherese.

    Restituisce:
      matches, unmatched_tracks, unmatched_detections
    """

    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    cost_matrix = build_cost_matrix(tracks, detections)

    # Nota: se la matrice contiene righe/colonne interamente infinite,
    # scipy può segnalare un problema di assegnamento infeasible.
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matches = []

    for track_index, detection_index in zip(
        track_indices,
        detection_indices,
    ):

        cost = cost_matrix[track_index, detection_index]

        if np.isfinite(cost):
            matches.append((track_index, detection_index))

    matched_tracks = {
        track_index for track_index, _ in matches
    }

    matched_detections = {
        detection_index for _, detection_index in matches
    }

    unmatched_tracks = [
        i for i in range(len(tracks))
        if i not in matched_tracks
    ]

    unmatched_detections = [
        i for i in range(len(detections))
        if i not in matched_detections
    ]

    return matches, unmatched_tracks, unmatched_detections

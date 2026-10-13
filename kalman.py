import numpy as np

from config import DT, Q, R


class Kalman1D:
    """
    Filtro di Kalman 1D con stato:
        x = [posizione, velocità]^T

    La misura contiene soltanto la posizione.
    """

    def __init__(self, initial_position):

        self.x = np.array([
            [initial_position],
            [0.0],
        ])

        self.P = np.array([
            [0.01, 0.0],
            [0.0,  1.0],
        ])

        self.F = np.array([
            [1.0, DT],
            [0.0, 1.0],
        ])

        self.H = np.array([
            [1.0, 0.0],
        ])

    def predict(self):
        """Predice stato e covarianza al frame successivo."""

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        return self.x.copy()

    def innovation_data(self, measurement):
        """Calcola innovazione e covarianza dell'innovazione."""

        z = np.array([[measurement]])

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        return innovation, S

    def update(self, measurement):
        """Aggiorna stato e covarianza con una misura."""

        z = np.array([[measurement]])

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

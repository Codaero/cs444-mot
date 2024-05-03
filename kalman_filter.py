# TODO: Implement Kalman Filter
import numpy as np


# 30 FPS so each frame differs by 1/30th of a second


# State Vector: x, y, x_vel, y_vel, width, height
class KalmanFilter:
    def __init__(self):
        # Constants
        self.deltaT = 1 / 30

        # Measurement Noise Covariance (diagonal elements are covariances)
        self.R = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Tune these
        self.process_cv = {
            "wx": 1000,
            "wy": 1000,
            "ww": 1000,
            "wh": 1000,
        }
        # Process Noise Covariance
        self.Q = np.array([[(self.deltaT ** 4) / 4 * self.process_cv["wx"], 0,
                            (self.deltaT ** 3) / 2 * self.process_cv["wx"], 0, 0, 0],
                           [0, (self.deltaT ** 4) / 4 * self.process_cv["wy"], 0,
                            (self.deltaT ** 3) / 2 * self.process_cv["wy"], 0, 0],
                           [(self.deltaT ** 3) / 2 * self.process_cv["wx"], 0, self.deltaT ** 2 * self.process_cv["wx"],
                            0, 0, 0],
                           [0, (self.deltaT ** 3) / 2 * self.process_cv["wy"], 0,
                            self.deltaT ** 2 * self.process_cv["wy"], 0, 0],
                           [0, 0, 0, 0, self.process_cv["ww"], 0],
                           [0, 0, 0, 0, 0, self.process_cv["wh"]]
                           ])

        # State Transition Matrix
        self.F = np.array([[1, 0, self.deltaT, 0, 0, 0],
                           [0, 1, 0, self.deltaT, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]
                           ])

        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]
                           ])

    def predict(self, x_curr, P_curr):
        x_next = self.F @ x_curr.T
        p_next = self.F @ P_curr @ self.F.T + self.Q
        return x_next, p_next

    def update(self, measurement, x_curr, P_curr):
        kalman_gain = P_curr @ self.H.T @ np.linalg.inv(self.H @ P_curr @ self.H.T + self.R)

        new_state = x_curr + kalman_gain @ (measurement - self.H @ x_curr)

        p_new = (np.identity(6) - kalman_gain @ self.H) @ P_curr @ (
                    np.identity(6) - kalman_gain @ self.H).T + kalman_gain @ self.R @ kalman_gain.T

        return new_state, p_new

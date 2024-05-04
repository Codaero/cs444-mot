# TODO: Implement Kalman Filter
import numpy as np


# 30 FPS so each frame differs by 1/30th of a second


# State Vector: x, y, x_vel, y_vel, width, height
class KalmanFilter:
    def __init__(self):
        # Constants
        self.deltaT = 1 / 30

        # Measurement Noise Covariance (diagonal elements are covariances)
        self.R = np.array([[1, 0],
                           [0, 1]])

        # Tune these
        self.process_cv = {
            "wx": 1000,
            "wy": 1000,
            "ww": 1000,
            "wh": 1000,
        }
        # Process Noise Covariance
        self.Q = np.array([[(self.deltaT ** 4) / 4 * self.process_cv["wx"], 0,
                            (self.deltaT ** 3) / 2 * self.process_cv["wx"], 0],
                           [0, (self.deltaT ** 4) / 4 * self.process_cv["wy"], 0,
                            (self.deltaT ** 3) / 2 * self.process_cv["wy"]],
                           [(self.deltaT ** 3) / 2 * self.process_cv["wx"], 0, self.deltaT ** 2 * self.process_cv["wx"],
                            0],
                           [0, (self.deltaT ** 3) / 2 * self.process_cv["wy"], 0,
                            self.deltaT ** 2 * self.process_cv["wy"]],
                           ])

        # State Transition Matrix
        self.A = np.array([[1, 0, self.deltaT, 0],
                           [0, 1, 0, self.deltaT],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           ])

        # Positional Extracter
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])


    def predict(self, x_curr, P_curr):
        x_next = self.A @ x_curr.T
        p_next = self.A @ P_curr @ self.A.T + self.Q
        return x_next, p_next

    def update(self, measurement, x_curr, P_curr):
        y_k = self.C @ measurement

        kalman_gain = P_curr @ self.H.T @ np.linalg.inv(self.H @ P_curr @ self.H.T + self.R)

        new_state = x_curr + kalman_gain @ (y_k - self.H @ x_curr)

        p_new = (np.identity(4) - kalman_gain @ self.H) @ P_curr

        return new_state, p_new

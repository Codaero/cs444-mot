from kalman_filter import KalmanFilter
import numpy as np

def main():
    kf = KalmanFilter()
    WIDTH = 10
    HEIGHT = 10

    # x, y, vx, vy
    curr_state = np.array([1, 1, 1, 1])
    curr_est_cov = np.identity(4)


    x_next, p_next = kf.predict(curr_state, curr_est_cov)
    print(f"Predicted State:\n{x_next}\n")
    print(f"Extrapolated Estimate Covariance:\n{p_next}")

    measurement = [1.2, 1.2, 1, 1]

    new_state, p_new = kf.update(measurement, x_next, p_next)

    print(f"Corrected State: {new_state}")
    print(f"New Estimate Covariance: {p_new}")


if __name__ == "__main__":
    main()

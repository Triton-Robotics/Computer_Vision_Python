from filterpy.kalman import KalmanFilter
import numpy as np

# creates kalman filter with specified parameters
def create_kalman():
    # constants
    mod = 0.01
    dt = 0.1
    obs = 10

    kalman = KalmanFilter(dim_x=8, dim_z=4)

    # initial state - might change to center of frame and set square size
    # current stae is uninitialized
    # kalman.x = np.array([[1, 0, 0, 0],
    #                      [0, 1, 0, 1],
    #                      [0, 0, 1, 0],
    #                      [0, 0, 0, 1]])

    # state transition matrix
    # [cx, cy, w, h, vx, vy, vw, vh]
    kalman.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                         [0, 1, 0, 0, 0, dt, 0, 0],
                         [0, 0, 1, 0, 0, 0, dt, 0],
                         [0, 0, 0, 1, 0, 0, 0, dt],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    # measurement function
    kalman.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0]])

    # covariance matrix
    # try using default first of 1's uncomment to use something other than 1
    # kalman.P *= 100

    # measurement noise
    # lets assume that the center coords are off by 1, and the width and 
    # height are off by 10
    kalman.R = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 10, 0],
                         [0, 0, 0, 10]])

    # process noise covariance matrix 
    kalman.Q = np.matrix(np.eye(8)) * mod # default is 0.01 process noise
    return kalman
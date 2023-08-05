import numpy as np

def estimate_parameters(u, y, nr_poles=1, nr_zeros=1, dead_time=0):
    N = len(u)
    # Calculate the regressor matrix
    od = max(nr_poles, nr_zeros)
    X = np.zeros((N-od-dead_time, nr_poles+nr_zeros))
    for i in range(nr_poles):
        X[:, i] = -y[od+dead_time-i-1:N-i-1]
    for i in range(nr_zeros):
        X[:, i+nr_poles] = u[od-i-1:N-dead_time-i-1]
        
    Y = y[od+dead_time:]
    TH = np.linalg.lstsq(X, Y, rcond=None)[0]
    return TH

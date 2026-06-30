import numpy as np

CWC_ETA = 50    # penalty slope for under-coverage

coverage_dict = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}

def compute_metrics(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, coverage: float):
    """Pure (non-conformalized) MCD interval metrics from Gaussian mean +/- z*std."""
    p = picp(lower, upper, y_true, coverage=coverage)
    m = mpiw(lower, upper, y_true, coverage=coverage)
    pn = pinaw(lower, upper, y_true, coverage=coverage)
    c = cwc(lower, upper, y_true, coverage=coverage)
    return p, m, pn, c


def picp(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, coverage=0.95):
    """compute picp."""
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return picp

def mpiw(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, coverage=0.95):
    """compute mpiw."""
    mpiw = float(np.mean(upper - lower))
    return mpiw

def pinaw(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, coverage=0.95):
    """compute pinaw."""
    pinaw = float(np.mean(upper - lower)) / (y_true.max() - y_true.min())
    return pinaw

def cwc(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, coverage=0.95):
    """compute cwc."""
    alpha = 1-coverage

    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    y_range = float(y_true.max() - y_true.min())
    pinaw = mpiw / y_range if y_range > 0 else float("nan")

    nominal = 1.0 - alpha
    gamma = 1.0 if picp < nominal else 0.0
    cwc = pinaw * (1.0 + gamma * np.exp(-CWC_ETA * (picp - nominal)))

    return cwc




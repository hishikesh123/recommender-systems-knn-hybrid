import numpy as np

EPSILON = 1e-9


def evaluate(test_ds: np.ndarray, predicted_ds: np.ndarray):
    """
    Compute MAE and RMSE on only the observed entries in test_ds (where test_ds > 0).

    Parameters
    ----------
    test_ds : np.ndarray
        Ground-truth test ratings matrix, zeros indicate missing.
    predicted_ds : np.ndarray
        Predicted ratings matrix.

    Returns
    -------
    (mae, rmse) : tuple[float, float]
    """
    mask = test_ds > 0
    denom = np.sum(mask.astype(np.float32)) + EPSILON

    mae = np.sum(np.abs(test_ds[mask] - predicted_ds[mask])) / denom
    rmse = np.sqrt(np.sum((test_ds[mask] - predicted_ds[mask]) ** 2) / denom)
    return float(mae), float(rmse)


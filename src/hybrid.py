import numpy as np
from .metrics import evaluate


def hybrid_predict(user_pred: np.ndarray, item_pred: np.ndarray, lam: float) -> np.ndarray:
    """
    Hybrid prediction as in your notebook:
    pred = lam * user_pred + (1 - lam) * item_pred
    """
    lam = float(lam)
    return lam * user_pred + (1.0 - lam) * item_pred


def tune_lambda(
    test_ds: np.ndarray,
    user_pred: np.ndarray,
    item_pred: np.ndarray,
    lambda_values=None,
):
    """
    Tune lambda (0..1) to minimize RMSE, exactly like your notebook.

    Returns
    -------
    best_lambda : float
    best_mae : float
    best_rmse : float
    results : dict[lambda] -> (mae, rmse)
    """
    if lambda_values is None:
        lambda_values = np.linspace(0, 1, 21)

    results = {}
    best_lambda = None
    best_rmse = float("inf")
    best_mae = None

    for lam in lambda_values:
        pred = hybrid_predict(user_pred, item_pred, lam)
        mae, rmse = evaluate(test_ds, pred)
        results[float(lam)] = (float(mae), float(rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_lambda = float(lam)

    return best_lambda, float(best_mae), float(best_rmse), results


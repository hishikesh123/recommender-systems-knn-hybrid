import numpy as np
from .metrics import evaluate, EPSILON
from .similarity import cosine_similarity, pearson_similarity


def user_average_baseline(train_ds: np.ndarray) -> np.ndarray:
    """Per-user mean of non-zero ratings (0 if no ratings)."""
    denom = (train_ds != 0).sum(axis=1)
    avg = np.zeros(train_ds.shape[0], dtype=float)
    mask = denom > 0
    avg[mask] = train_ds.sum(axis=1)[mask] / denom[mask]
    return avg


def item_average_baseline(train_ds: np.ndarray) -> np.ndarray:
    """Per-item mean of non-zero ratings (0 if no ratings)."""
    denom = (train_ds != 0).sum(axis=0)
    avg = np.zeros(train_ds.shape[1], dtype=float)
    mask = denom > 0
    avg[mask] = train_ds.sum(axis=0)[mask] / denom[mask]
    return avg


def global_average(train_ds: np.ndarray) -> float:
    """Global mean of all observed ratings."""
    vals = train_ds[train_ds > 0]
    return float(vals.mean()) if vals.size else 0.0


def _get_similarity(train_ds: np.ndarray, metric: str, mode: str) -> np.ndarray:
    """
    metric: 'cosine' | 'pearson'
    mode: 'user' -> similarity over train_ds rows
          'item' -> similarity over train_ds columns
    """
    if mode == "user":
        mat = train_ds
    elif mode == "item":
        mat = train_ds.T
    else:
        raise ValueError("mode must be 'user' or 'item'")

    if metric == "cosine":
        return cosine_similarity(mat)
    if metric == "pearson":
        return pearson_similarity(mat)

    raise ValueError("metric must be 'cosine' or 'pearson'")


def user_knn_predict(
    train_ds: np.ndarray,
    test_ds: np.ndarray,
    user_avg: np.ndarray,
    global_avg_value: float,
    similarity_metric: str = "cosine",
    k: int = 5,
    sim_exponent: float = 1.0,
    sim_threshold: float = 0.0,
) -> np.ndarray:
    """
    Your notebook's User-KNN CF prediction.

    - compute user-user similarity on train_ds
    - for each (u,i) in test_ds:
        - consider users who rated i
        - filter by similarity threshold
        - take top-k by similarity
        - weight ratings by sim^sim_exponent
        - fallback to user_avg[u], else global_avg_value
    """
    n_users, n_items = train_ds.shape
    user_similarity = _get_similarity(train_ds, similarity_metric, mode="user")

    predicted = np.zeros((n_users, n_items), dtype=float)

    for u in range(n_users):
        sim_u = user_similarity[u]

        for i in range(n_items):
            if test_ds[u, i] <= 0:
                continue

            ratings_i = train_ds[:, i]
            valid = ratings_i > 0

            sim_valid = sim_u[valid]
            ratings_valid = ratings_i[valid]

            # similarity threshold
            keep = sim_valid >= sim_threshold
            sim_valid = sim_valid[keep]
            ratings_valid = ratings_valid[keep]

            if sim_valid.size == 0:
                fb = user_avg[u]
                predicted[u, i] = fb if fb != 0 else global_avg_value
                continue

            # top-k
            if sim_valid.size >= k:
                top_idx = np.argsort(sim_valid)[-k:]
            else:
                top_idx = np.argsort(sim_valid)

            top_sim = sim_valid[top_idx]
            top_ratings = ratings_valid[top_idx]

            weights = np.power(top_sim, sim_exponent)

            s = weights.sum()
            if s > 0:
                predicted[u, i] = float(np.dot(weights, top_ratings) / (s + EPSILON))
            else:
                fb = user_avg[u]
                predicted[u, i] = fb if fb != 0 else global_avg_value

    return predicted


def item_knn_predict(
    train_ds: np.ndarray,
    test_ds: np.ndarray,
    item_avg: np.ndarray,
    similarity_metric: str = "cosine",
    k: int = 5,
) -> np.ndarray:
    """
    Your notebook's Item-KNN CF prediction.

    - compute item-item similarity on train_ds.T
    - for each (u,i) in test_ds:
        - consider items rated by user u
        - take top-k by similarity
        - weighted avg using sim weights
        - fallback to item_avg[i]
    """
    n_users, n_items = train_ds.shape
    item_similarity = _get_similarity(train_ds, similarity_metric, mode="item")

    predicted = np.zeros((n_users, n_items), dtype=float)

    for u in range(n_users):
        user_ratings = train_ds[u]

        rated_items = user_ratings > 0
        rated_ratings = user_ratings[rated_items]

        for i in range(n_items):
            if test_ds[u, i] <= 0:
                continue

            sim_i = item_similarity[i]
            sim_valid = sim_i[rated_items]

            if sim_valid.size == 0:
                predicted[u, i] = float(item_avg[i])
                continue

            if sim_valid.size >= k:
                top_idx = np.argsort(sim_valid)[-k:]
            else:
                top_idx = np.argsort(sim_valid)

            top_sim = sim_valid[top_idx]
            top_ratings = rated_ratings[top_idx]

            s = top_sim.sum()
            if s > 0:
                predicted[u, i] = float(np.dot(top_sim, top_ratings) / (s + EPSILON))
            else:
                predicted[u, i] = float(item_avg[i])

    return predicted


def search_best_user_knn(
    train_ds: np.ndarray,
    test_ds: np.ndarray,
    k_values=(1, 3, 5, 10, 15, 20),
    similarity_metrics=("cosine", "pearson"),
    sim_exponents=(0.5, 1.0),
    sim_thresholds=(0.0, 0.1, 0.2),
):
    """
    Grid-search exactly like your notebook for User-KNN.
    Select best by RMSE.

    Returns
    -------
    best_pred : np.ndarray
    best_params : dict
    best_mae : float
    best_rmse : float
    results : dict[(metric,k,exp,thresh)] -> (mae, rmse)
    """
    uavg = user_average_baseline(train_ds)
    gavg = global_average(train_ds)

    best_rmse = float("inf")
    best_mae = None
    best_pred = None
    best_params = {}
    results = {}

    for metric in similarity_metrics:
        for k in k_values:
            for exp in sim_exponents:
                for thresh in sim_thresholds:
                    pred = user_knn_predict(
                        train_ds,
                        test_ds,
                        user_avg=uavg,
                        global_avg_value=gavg,
                        similarity_metric=metric,
                        k=int(k),
                        sim_exponent=float(exp),
                        sim_threshold=float(thresh),
                    )
                    mae, rmse = evaluate(test_ds, pred)
                    results[(metric, int(k), float(exp), float(thresh))] = (mae, rmse)

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_mae = mae
                        best_pred = pred
                        best_params = {
                            "metric": metric,
                            "k": int(k),
                            "sim_exponent": float(exp),
                            "sim_threshold": float(thresh),
                        }

    return best_pred, best_params, float(best_mae), float(best_rmse), results


def search_best_item_knn(
    train_ds: np.ndarray,
    test_ds: np.ndarray,
    k_values=(1, 3, 5, 10, 15, 20),
    similarity_metrics=("cosine", "pearson"),
):
    """
    Grid-search like your notebook for Item-KNN.
    Select best by RMSE.

    Returns
    -------
    best_pred : np.ndarray
    best_params : dict
    best_mae : float
    best_rmse : float
    all_results : dict[metric][k] -> (mae, rmse)
    """
    iavg = item_average_baseline(train_ds)

    best_rmse = float("inf")
    best_pred = None
    best_params = {}
    all_results = {}

    for metric in similarity_metrics:
        all_results[metric] = {}
        for k in k_values:
            pred = item_knn_predict(
                train_ds,
                test_ds,
                item_avg=iavg,
                similarity_metric=metric,
                k=int(k),
            )
            mae, rmse = evaluate(test_ds, pred)
            all_results[metric][int(k)] = (mae, rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_pred = pred
                best_params = {"metric": metric, "k": int(k)}
                best_mae = mae

    return best_pred, best_params, float(best_mae), float(best_rmse), all_results


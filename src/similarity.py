import numpy as np

EPSILON = 1e-9


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity for a 2D matrix.

    Returns a (n_rows x n_rows) similarity matrix.
    """
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = EPSILON
    normalized = matrix / norm
    return normalized @ normalized.T


def pearson_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Row-wise Pearson-like similarity for a 2D matrix as used in your notebook:
    - Mean-center each row using the mean of non-zero entries
    - Keep missing ratings as zeros
    - Cosine on the centered rows

    Returns a (n_rows x n_rows) similarity matrix.
    """
    counts = (matrix != 0).sum(axis=1) + EPSILON
    means = matrix.sum(axis=1) / counts

    centered = matrix - means[:, None]
    centered[matrix == 0] = 0  # preserve missing as 0

    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    norm[norm == 0] = EPSILON
    normalized = centered / norm
    return normalized @ normalized.T


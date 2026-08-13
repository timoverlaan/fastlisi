"""Implementation of the LISI statistic.

Public names are re-exported from ``fastlisi``; import from there rather than
from this module.

Copyright (C) 2018  Ilya Korsunsky
              2019  Kamil Slowikowski <kslowikowski@gmail.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See <https://www.gnu.org/licenses/>.
"""

from typing import Iterable, Optional

import numpy as np

from fastlisi._fastlisi import compute_lisi_cpp, max_threads

__all__ = ["compute_lisi", "compute_lisi_from_codes", "max_threads"]


def compute_lisi(
    X,
    metadata,
    label_colnames: Iterable[str],
    perplexity: float = 30,
    n_threads: int = 0,
) -> np.ndarray:
    """Compute the Local Inverse Simpson Index (LISI) for each label column.

    LISI is computed for each item (row) of ``X``. Suppose a metadata column is
    categorical with 3 categories:

    - LISI near 3 means the item is surrounded by neighbors from all 3
      categories.
    - LISI near 1 means the item is surrounded by neighbors from 1 category.

    Parameters
    ----------
    X
        ``(n_cells, n_dims)`` embedding, typically PCA or harmonized PCA
        coordinates. Anything ``numpy.asarray`` accepts, including a pandas
        DataFrame.
    metadata
        Anything indexable by column name, e.g. a pandas DataFrame, a dict of
        arrays, or a numpy structured array. Each column must have length
        ``n_cells``.
    label_colnames
        Column names in ``metadata`` to compute LISI for. All of them share a
        single neighbor search, so passing several costs little more than one.
    perplexity
        Effective number of neighbors used for the Gaussian kernel. The search
        collects ``3 * perplexity`` neighbors per cell.
    n_threads
        Threads to use. 0 (default) uses one per core. Results are identical
        regardless of this value.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, n_label_colnames)`` array of LISI values, in the order the
        column names were given.

    References
    ----------
    Korsunsky et al. 2019, doi:10.1038/s41592-019-0619-0
    """
    label_colnames = list(label_colnames)
    if not label_colnames:
        raise ValueError("label_colnames is empty; give at least one column name")

    X_arr = _as_c_float64_2d(X)
    n_cells = X_arr.shape[0]

    codes = np.empty((len(label_colnames), n_cells), dtype=np.int32)
    n_categories = np.empty(len(label_colnames), dtype=np.int32)
    for i, label in enumerate(label_colnames):
        try:
            col = np.asarray(metadata[label])
        except (KeyError, IndexError, TypeError) as err:
            raise KeyError(f"metadata has no column {label!r}") from err
        col = np.reshape(col, -1)
        if col.shape[0] != n_cells:
            raise ValueError(
                f"metadata column {label!r} has {col.shape[0]} rows, "
                f"but X has {n_cells}"
            )
        uniques, inverse = np.unique(col, return_inverse=True)
        codes[i] = np.reshape(inverse, n_cells)
        n_categories[i] = len(uniques)

    return compute_lisi_from_codes(X_arr, codes, n_categories, perplexity, n_threads)


def compute_lisi_from_codes(
    X,
    codes,
    n_categories: Optional[np.ndarray] = None,
    perplexity: float = 30,
    n_threads: int = 0,
) -> np.ndarray:
    """Lower-level entry point taking integer category codes directly.

    Use this to skip the ``numpy.unique`` step when labels are already encoded.

    Parameters
    ----------
    X
        ``(n_cells, n_dims)`` embedding.
    codes
        ``(n_label_columns, n_cells)`` array of 0-based category codes. A 1-D
        array of length ``n_cells`` is accepted for a single label column.
    n_categories
        Per-column category count. Inferred as ``codes.max() + 1`` if omitted.
    perplexity, n_threads
        As in :func:`compute_lisi`.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, n_label_columns)`` array of LISI values.
    """
    X_arr = _as_c_float64_2d(X)
    n_cells = X_arr.shape[0]

    codes_arr = np.ascontiguousarray(np.asarray(codes), dtype=np.int32)
    if codes_arr.ndim == 1:
        codes_arr = codes_arr.reshape(1, -1)
    if codes_arr.ndim != 2:
        raise ValueError("codes must be 1-D or 2-D")
    if codes_arr.shape[1] != n_cells:
        raise ValueError(
            f"codes has {codes_arr.shape[1]} cells but X has {n_cells}; "
            "codes must be shaped (n_label_columns, n_cells)"
        )
    if codes_arr.size and codes_arr.min() < 0:
        raise ValueError("codes must be non-negative")

    if n_categories is None:
        n_categories = (
            codes_arr.max(axis=1) + 1
            if codes_arr.size
            else np.zeros(codes_arr.shape[0])
        )
    n_cat = np.ascontiguousarray(np.asarray(n_categories), dtype=np.int32).reshape(-1)
    if n_cat.shape[0] != codes_arr.shape[0]:
        raise ValueError("n_categories must have one entry per label column")
    if codes_arr.size and np.any(codes_arr.max(axis=1) >= n_cat):
        raise ValueError("a code is >= its column's n_categories")

    if perplexity <= 0:
        raise ValueError("perplexity must be positive")

    lisi = compute_lisi_cpp(X_arr, codes_arr, n_cat, float(perplexity), int(n_threads))
    return np.ascontiguousarray(lisi.T)


def _as_c_float64_2d(X) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n_cells, n_dims), got shape {arr.shape}")
    if not arr.size:
        raise ValueError("X is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains NaN or infinite values")
    return arr

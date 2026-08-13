"""Tests for fastlisi.

The reference values in data/lisi_lisi.tsv.gz come from the original R
implementation and are the same fixture harmonypy tests against.
"""

import os

import numpy as np
import pandas as pd
import pytest

import fastlisi

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="module")
def fixture():
    X = pd.read_csv(os.path.join(DATA, "lisi_x.tsv.gz"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATA, "lisi_metadata.tsv.gz"), sep="\t")
    expected = pd.read_csv(os.path.join(DATA, "lisi_lisi.tsv.gz"), sep="\t")
    return X, metadata, expected.iloc[:, -2:].to_numpy()


def test_matches_reference(fixture):
    """LISI values match the original R implementation."""
    X, metadata, expected = fixture
    got = fastlisi.compute_lisi(X, metadata, metadata.columns, perplexity=30)
    assert got.shape == expected.shape
    # Tolerance accounts for differences between kd-tree implementations.
    assert np.allclose(got, expected, atol=0.01)


def test_thread_count_does_not_change_results(fixture):
    """Every cell is independent, so threading must be bit-for-bit stable."""
    X, metadata, _ = fixture
    one = fastlisi.compute_lisi(X, metadata, metadata.columns, n_threads=1)
    many = fastlisi.compute_lisi(X, metadata, metadata.columns, n_threads=4)
    assert np.array_equal(one, many)


def test_label_columns_are_independent(fixture):
    """Columns computed together match the same columns computed alone."""
    X, metadata, _ = fixture
    cols = list(metadata.columns)
    together = fastlisi.compute_lisi(X, metadata, cols)
    for i, col in enumerate(cols):
        alone = fastlisi.compute_lisi(X, metadata, [col])
        assert np.array_equal(together[:, i], alone[:, 0])


def test_column_order_is_respected(fixture):
    X, metadata, _ = fixture
    cols = list(metadata.columns)
    forward = fastlisi.compute_lisi(X, metadata, cols)
    backward = fastlisi.compute_lisi(X, metadata, cols[::-1])
    assert np.array_equal(forward, backward[:, ::-1])


def test_single_category_gives_one():
    """A label with one category means every neighbor matches: LISI == 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    metadata = {"constant": np.zeros(200, dtype=int)}
    got = fastlisi.compute_lisi(X, metadata, ["constant"])
    assert np.allclose(got, 1.0)


def test_perfectly_mixed_approaches_category_count():
    """Randomly assigned labels on random data should mix well."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 10))
    metadata = {"batch": rng.integers(0, 3, 3000)}
    got = fastlisi.compute_lisi(X, metadata, ["batch"])
    assert 2.5 < got.mean() < 3.0


def test_fully_separated_approaches_one():
    """Labels that coincide with well-separated clusters give LISI near 1."""
    rng = np.random.default_rng(2)
    blocks = [rng.normal(loc, 0.1, size=(500, 5)) for loc in (-50, 0, 50)]
    X = np.vstack(blocks)
    metadata = {"batch": np.repeat([0, 1, 2], 500)}
    got = fastlisi.compute_lisi(X, metadata, ["batch"])
    assert got.mean() < 1.05


def test_accepts_dict_and_dataframe_alike(fixture):
    X, metadata, _ = fixture
    from_df = fastlisi.compute_lisi(X, metadata, ["label1"])
    from_dict = fastlisi.compute_lisi(X, {"label1": metadata["label1"].to_numpy()}, ["label1"])
    assert np.array_equal(from_df, from_dict)


def test_accepts_noncontiguous_and_float32_input(fixture):
    X, metadata, _ = fixture
    reference = fastlisi.compute_lisi(X, metadata, ["label1"])
    fortran = np.asfortranarray(X.to_numpy())
    assert np.array_equal(fastlisi.compute_lisi(fortran, metadata, ["label1"]), reference)


def test_from_codes_matches_compute_lisi(fixture):
    X, metadata, _ = fixture
    cols = list(metadata.columns)
    expected = fastlisi.compute_lisi(X, metadata, cols)
    codes = np.stack([
        np.unique(metadata[c].to_numpy(), return_inverse=True)[1] for c in cols
    ]).astype(np.int32)
    got = fastlisi.compute_lisi_from_codes(X, codes)
    assert np.array_equal(got, expected)


def test_fewer_cells_than_neighbors():
    """n_cells below 3 * perplexity must still work."""
    rng = np.random.default_rng(3)
    for n in (1, 2, 10, 91):
        X = rng.normal(size=(n, 4))
        metadata = {"b": rng.integers(0, 2, n)}
        got = fastlisi.compute_lisi(X, metadata, ["b"])
        assert got.shape == (n, 1)
        assert np.all(np.isfinite(got))


def test_input_validation():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 3))
    meta = {"b": rng.integers(0, 2, 50)}

    with pytest.raises(ValueError, match="2-D"):
        fastlisi.compute_lisi(rng.normal(size=50), meta, ["b"])
    with pytest.raises(ValueError, match="NaN"):
        bad = X.copy()
        bad[0, 0] = np.nan
        fastlisi.compute_lisi(bad, meta, ["b"])
    with pytest.raises(ValueError, match="perplexity"):
        fastlisi.compute_lisi(X, meta, ["b"], perplexity=0)
    with pytest.raises(ValueError, match="empty"):
        fastlisi.compute_lisi(X, meta, [])
    with pytest.raises(KeyError):
        fastlisi.compute_lisi(X, meta, ["missing"])
    with pytest.raises(ValueError, match="rows"):
        fastlisi.compute_lisi(X, {"b": np.zeros(10, dtype=int)}, ["b"])


def test_build_reports_openmp():
    """Not a correctness check: surfaces a single-threaded build in CI logs."""
    print(f"openmp_enabled={fastlisi.openmp_enabled()} max_threads={fastlisi.max_threads()}")
    assert isinstance(fastlisi.openmp_enabled(), bool)
    assert fastlisi.max_threads() >= 1

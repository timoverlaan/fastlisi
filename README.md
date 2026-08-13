# fastlisi

Fast **Local Inverse Simpson Index (LISI)** — the standard statistic for judging
whether a single-cell batch-integration method actually mixed the batches.

This is the LISI implementation from
[harmonypy](https://github.com/slowkow/harmonypy), rewritten for speed. It
returns the same values (agreement to ~1e-15) and is typically **15–30× faster**.

Unlike harmonypy it has no Armadillo, BLAS, or LAPACK dependency — just numpy at
runtime — so wheels install cleanly everywhere.

## Install

```bash
pip install fastlisi
```

## Use

```python
import fastlisi

# X: (n_cells, n_dims) embedding, e.g. PCA or harmonized PCA coordinates
# metadata: pandas DataFrame (or dict of arrays) with one row per cell
lisi = fastlisi.compute_lisi(X, metadata, ["batch", "cell_type"])
# -> (n_cells, 2) array
```

Interpretation, for a label with 3 categories:

- LISI ≈ 3 → the cell's neighborhood contains all 3 categories (well mixed)
- LISI ≈ 1 → the cell's neighborhood is a single category (not mixed)

Good integration means **high** LISI on the batch label and **low** LISI on the
cell-type label.

Passing several label columns at once is much cheaper than several calls: the
neighbor graph does not depend on the labels, so it is built once and reused.

```python
# Limit threads (0, the default, uses one per core)
lisi = fastlisi.compute_lisi(X, metadata, ["batch"], perplexity=30, n_threads=4)

# Skip the label-encoding step if you already have integer codes
lisi = fastlisi.compute_lisi_from_codes(X, codes)  # codes: (n_labels, n_cells)

fastlisi.openmp_enabled()  # False means this build runs single-threaded
```

## Speed

Measured on the datasets shipped with harmonypy, on a 2-core machine. `old` is
harmonypy 2.0.0, `new` is fastlisi. Outputs agree to 2.2e-15.

| dataset | cells | dims | old | new | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lisi_x` fixture | 400 | 2 | 0.017 s | 0.008 s | 2.3× |
| `pbmc_3500` | 3,500 | 30 | 1.41 s | 0.17 s | 8.4× |
| `ircolitis_cd8` | 68,785 | 20 | 800 s | 29.6 s | 27.1× |
| `ircolitis_cd8` | 68,785 | 50 | 2,613 s | 94.5 s | 27.7× |

More cores helps roughly linearly beyond this.

### Where the speed comes from

- **Point-contiguous storage.** The original stored the embedding column-major,
  so reading one point's coordinates strided by `8 × n_cells` bytes — a cache
  miss per dimension. Storing points contiguously makes a distance evaluation
  one vectorizable pass over ~2 cache lines.
- **Leaf buckets.** 16 points per kd-tree leaf, with points permuted into leaf
  order so a bucket scan is a sequential read.
- **Tighter pruning.** An incrementally maintained bounding-box lower bound
  instead of testing the splitting plane alone.
- **One neighbor search for all label columns**, instead of rebuilding the tree
  per column.
- **Threaded** over cells, with per-thread scratch buffers replacing roughly
  `4 × n_cells` heap allocations.

### Scaling

Be aware that LISI is close to **quadratic in the number of cells** on real
embeddings — kd-trees prune poorly at 20–50 dimensions, so the neighbor search
tends toward all-pairs. Measured exponent here is about `N^1.95`. This package is
a large constant-factor improvement, not an asymptotic one. Past a few hundred
thousand cells you want an approximate neighbor index instead.

## Accuracy

`tests/` checks against `lisi_lisi.tsv.gz`, the reference values produced by the
original R implementation, and the same fixture harmonypy tests against.

Results are bit-for-bit identical regardless of `n_threads`, since each cell is
computed independently.

## Credit and license

The algorithm and the reference test data are from
[harmonypy](https://github.com/slowkow/harmonypy) by Ilya Korsunsky and Kamil
Slowikowski. If you use LISI, cite the Harmony paper:

> Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell
> data with Harmony. *Nature Methods* (2019).
> [doi:10.1038/s41592-019-0619-0](https://doi.org/10.1038/s41592-019-0619-0)

GPL-3.0-or-later, inherited from harmonypy. See [LICENSE](LICENSE).

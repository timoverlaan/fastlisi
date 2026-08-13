"""fastlisi - Local Inverse Simpson Index (LISI).

LISI measures how mixed the neighborhood of each cell is with respect to a
categorical label. It is the standard statistic for judging whether a batch
integration method (Harmony, scVI, ...) actually mixed the batches.

This is a faster implementation of the LISI code from harmonypy
(https://github.com/slowkow/harmonypy), returning identical values.

Copyright (C) 2018  Ilya Korsunsky
              2019  Kamil Slowikowski <kslowikowski@gmail.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See <https://www.gnu.org/licenses/>.
"""

from importlib.metadata import PackageNotFoundError, version

from fastlisi._core import compute_lisi, compute_lisi_from_codes, max_threads

try:
    __version__ = version("fastlisi")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

del PackageNotFoundError, version

__all__ = ["compute_lisi", "compute_lisi_from_codes", "max_threads"]

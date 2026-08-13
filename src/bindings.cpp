// fastlisi - Python bindings
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <stdexcept>
#include <vector>
#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "lisi.hpp"

namespace nb = nanobind;

NB_MODULE(_fastlisi, m) {
    m.doc() = "Local Inverse Simpson Index (LISI), C++ implementation.";

    using NpF64_2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using NpI32_1D = nb::ndarray<int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using NpI32_2D = nb::ndarray<int, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    m.def("compute_lisi_cpp",
        [](NpF64_2D X, NpI32_2D labels, NpI32_1D n_categories,
           double perplexity, int n_threads)
            -> nb::ndarray<nb::numpy, double, nb::ndim<2>> {
            size_t N = X.shape(0), d = X.shape(1);
            size_t n_labels = labels.shape(0);
            if (labels.shape(1) != N)
                throw std::invalid_argument("labels must have shape (n_labels, n_cells)");
            if (n_categories.shape(0) != n_labels)
                throw std::invalid_argument("n_categories must have length n_labels");
            if (perplexity <= 0)
                throw std::invalid_argument("perplexity must be positive");

            // numpy's row-major buffer is already the layout the kd-tree wants,
            // so it is used directly with no conversion or copy.
            std::vector<double> result;
            {
                nb::gil_scoped_release release;
                result = lisi::compute_lisi_impl(
                    X.data(), static_cast<int>(N), static_cast<int>(d),
                    labels.data(), static_cast<int>(n_labels), n_categories.data(),
                    perplexity, n_threads
                );
            }

            double* out = new double[n_labels * N];
            std::memcpy(out, result.data(), n_labels * N * sizeof(double));
            nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            size_t shape[2] = { n_labels, N };
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(out, 2, shape, std::move(owner));
        },
        nb::arg("X"), nb::arg("labels"), nb::arg("n_categories"),
        nb::arg("perplexity") = 30.0, nb::arg("n_threads") = 0,
        nb::rv_policy::move,
        "Compute LISI for every label column at once, sharing one neighbor search.\n"
        "X: (n_cells, n_dims) float64, C-contiguous.\n"
        "labels: (n_labels, n_cells) int32 category codes.\n"
        "n_categories: (n_labels,) int32.\n"
        "Returns (n_labels, n_cells) float64."
    );

    m.def("openmp_enabled", []() {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }, "True if this build was compiled with OpenMP support.");

    m.def("max_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Number of threads OpenMP will use by default.");
}

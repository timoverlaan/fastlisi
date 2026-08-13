// LISI - The Local Inverse Simpson Index
// C++ implementation replacing sklearn-based Python version.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>

#ifndef LISI_HPP
#define LISI_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cstring>
#include <cstddef>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

namespace lisi {

// Threads to use when the caller does not specify. hardware_concurrency() is
// allowed to return 0 when it cannot tell, in which case fall back to serial.
inline int default_thread_count() {
    unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 1;
}

// ---------------------------------------------------------------------------
// kd-tree for k-nearest-neighbor search.
//
// Points are stored one per row, contiguously, which matters for two reasons:
//
//   1. A distance evaluation reads a point's d coordinates from adjacent
//      memory, so it is a single vectorizable pass over ~2 cache lines. The
//      previous column-major (arma::mat) layout strided by n_rows between
//      successive coordinates, costing one cache miss per dimension.
//   2. Points are permuted into leaf order at build time, so scanning a leaf
//      bucket is a sequential read.
//
// The caller's array is already row-major (numpy's native layout), so no
// conversion of the input buffer is required.
// ---------------------------------------------------------------------------
class KDTree {
public:
    // Per-thread working memory, allocated once and reused for every query.
    struct Scratch {
        std::vector<std::pair<double, int>> heap;
        std::vector<double> offset;
        std::vector<int> nn_idx;
        std::vector<double> nn_dist;
        Scratch(int k, int d) {
            heap.reserve(k + 1);
            offset.assign(d, 0.0);
            nn_idx.reserve(k + 1);
            nn_dist.reserve(k + 1);
        }
    };

    // X: row-major N x d.
    KDTree(const double* X, int N, int d, int leafsize = 16)
        : n_(N), d_(d), leafsize_(leafsize < 1 ? 1 : leafsize) {
        perm_.resize(N);
        std::iota(perm_.begin(), perm_.end(), 0);
        nodes_.reserve(2 * (N / leafsize_ + 1) + 8);
        root_ = build(X, 0, N, 0);
        pts_.resize(static_cast<std::size_t>(N) * d);
        for (int i = 0; i < N; ++i) {
            std::memcpy(&pts_[static_cast<std::size_t>(i) * d],
                        X + static_cast<std::size_t>(perm_[i]) * d,
                        d * sizeof(double));
        }
    }

    // perm()[i] is the original row index of the i-th point in tree order.
    const std::vector<int>& perm() const { return perm_; }
    int size() const { return n_; }

    // k nearest neighbors of point query_idx (in tree order, excluding
    // itself), sorted by increasing distance. Indices returned are in tree
    // order; distances are Euclidean, not squared.
    void knn(int query_idx, int k, Scratch& s) const {
        s.heap.clear();
        std::fill(s.offset.begin(), s.offset.end(), 0.0);
        double max_dist = std::numeric_limits<double>::max();
        const double* qp = &pts_[static_cast<std::size_t>(query_idx) * d_];
        search(root_, qp, query_idx, k, s, 0.0, max_dist);

        std::sort(s.heap.begin(), s.heap.end());
        s.nn_idx.resize(s.heap.size());
        s.nn_dist.resize(s.heap.size());
        for (std::size_t i = 0; i < s.heap.size(); ++i) {
            s.nn_dist[i] = std::sqrt(s.heap[i].first);  // squared -> actual
            s.nn_idx[i] = s.heap[i].second;
        }
    }

private:
    struct Node {
        double split_val;
        int split_dim;   // -1 marks a leaf
        int left, right;
        int begin, end;  // leaf point range, in tree order
    };

    std::vector<double> pts_;
    std::vector<int> perm_;
    std::vector<Node> nodes_;
    int n_, d_, leafsize_, root_;

    int build(const double* X, int begin, int end, int depth) {
        int idx = static_cast<int>(nodes_.size());
        if (end - begin <= leafsize_) {
            nodes_.push_back({0.0, -1, -1, -1, begin, end});
            return idx;
        }
        int dim = depth % d_;
        int mid = begin + (end - begin) / 2;
        std::nth_element(perm_.begin() + begin, perm_.begin() + mid, perm_.begin() + end,
                         [&](int a, int b) {
                             return X[static_cast<std::size_t>(a) * d_ + dim] <
                                    X[static_cast<std::size_t>(b) * d_ + dim];
                         });
        double split_val = X[static_cast<std::size_t>(perm_[mid]) * d_ + dim];
        nodes_.push_back({split_val, dim, -1, -1, begin, end});
        int l = build(X, begin, mid, depth + 1);
        int r = build(X, mid, end, depth + 1);
        nodes_[idx].left = l;
        nodes_[idx].right = r;
        return idx;
    }

    inline void consider(const double* qp, int pi, int k,
                         std::vector<std::pair<double, int>>& heap, double& max_dist) const {
        const double* pp = &pts_[static_cast<std::size_t>(pi) * d_];
        double dist = 0.0;
        for (int j = 0; j < d_; ++j) {
            double diff = qp[j] - pp[j];
            dist += diff * diff;
        }
        if (static_cast<int>(heap.size()) < k) {
            heap.push_back({dist, pi});
            std::push_heap(heap.begin(), heap.end());
            if (static_cast<int>(heap.size()) == k) max_dist = heap.front().first;
        } else if (dist < max_dist) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist, pi};
            std::push_heap(heap.begin(), heap.end());
            max_dist = heap.front().first;
        }
    }

    // box_dist is a lower bound on the distance from the query to this node's
    // region, maintained incrementally: descending into a far child replaces
    // that dimension's contribution rather than discarding the bound. This is
    // strictly tighter than testing the splitting plane alone, so more of the
    // tree is pruned.
    void search(int node_idx, const double* qp, int query_idx, int k, Scratch& s,
                double box_dist, double& max_dist) const {
        const Node& node = nodes_[node_idx];

        if (node.split_dim < 0) {
            for (int p = node.begin; p < node.end; ++p)
                if (p != query_idx) consider(qp, p, k, s.heap, max_dist);
            return;
        }

        int dim = node.split_dim;
        double diff = qp[dim] - node.split_val;
        int near = diff <= 0 ? node.left : node.right;
        int far = diff <= 0 ? node.right : node.left;

        search(near, qp, query_idx, k, s, box_dist, max_dist);

        double old_offset = s.offset[dim];
        double far_dist = box_dist - old_offset * old_offset + diff * diff;
        if (far_dist < max_dist || static_cast<int>(s.heap.size()) < k) {
            s.offset[dim] = diff;
            search(far, qp, query_idx, k, s, far_dist, max_dist);
            s.offset[dim] = old_offset;
        }
    }
};

// Compute Simpson's index for one cell given its neighbor distances and labels.
// P and cat_prob are caller-owned scratch buffers of size >= n_neighbors and
// >= n_categories respectively.
inline double compute_simpson_one(
    const double* distances, const int* indices, int n_neighbors,
    const int* labels, int n_categories, double perplexity,
    double* P, double* cat_prob, double tol = 1e-5
) {
    double logU = std::log(perplexity);
    double beta = 1.0;
    double betamin = -std::numeric_limits<double>::infinity();
    double betamax = std::numeric_limits<double>::infinity();

    double H = 0;

    // Binary search for the beta that gives the target perplexity
    for (int t = 0; t < 50; ++t) {
        double P_sum = 0, sum_dP = 0;
        for (int j = 0; j < n_neighbors; ++j) {
            double p = std::exp(-distances[j] * beta);
            P[j] = p;
            P_sum += p;
            sum_dP += distances[j] * p;
        }
        if (P_sum == 0) {
            H = 0;
            for (int j = 0; j < n_neighbors; ++j) P[j] = 0;
        } else {
            H = std::log(P_sum) + beta * sum_dP / P_sum;
            double inv = 1.0 / P_sum;
            for (int j = 0; j < n_neighbors; ++j) P[j] *= inv;
        }

        double Hdiff = H - logU;
        if (std::abs(Hdiff) < tol) break;

        if (Hdiff > 0) {
            betamin = beta;
            beta = std::isfinite(betamax) ? (beta + betamax) / 2 : beta * 2;
        } else {
            betamax = beta;
            beta = std::isfinite(betamin) ? (beta + betamin) / 2 : beta / 2;
        }
    }

    if (H == 0) return -1.0;

    // Simpson's index: sum of squared category probabilities
    for (int c = 0; c < n_categories; ++c) cat_prob[c] = 0.0;
    for (int j = 0; j < n_neighbors; ++j) cat_prob[labels[indices[j]]] += P[j];
    double simpson = 0.0;
    for (int c = 0; c < n_categories; ++c) simpson += cat_prob[c] * cat_prob[c];
    return simpson;
}

// Compute LISI for all cells and ALL label columns at once.
//
// The neighbor graph does not depend on the labels, so it is built and
// searched once and reused for every label column; only the (much cheaper)
// Simpson step repeats.
//
// X:             row-major N x d
// labels:        row-major n_labels x N, 0-indexed category codes
// n_categories:  length n_labels
// returns:       row-major n_labels x N of LISI values
inline std::vector<double> compute_lisi_impl(
    const double* X, int N, int d,
    const int* labels, int n_labels, const int* n_categories,
    double perplexity, int n_threads = 0
) {
    int k = static_cast<int>(perplexity * 3);
    std::vector<double> result(static_cast<std::size_t>(n_labels) * N);
    if (N <= 0 || n_labels <= 0) return result;

    KDTree tree(X, N, d);
    const std::vector<int>& perm = tree.perm();

    // Reorder each label column into tree order so neighbor lookups index
    // directly into it.
    std::vector<int> lab(static_cast<std::size_t>(n_labels) * N);
    for (int l = 0; l < n_labels; ++l) {
        const int* src = labels + static_cast<std::size_t>(l) * N;
        int* dst = lab.data() + static_cast<std::size_t>(l) * N;
        for (int i = 0; i < N; ++i) dst[i] = src[perm[i]];
    }

    int max_cat = 1;
    for (int l = 0; l < n_labels; ++l) max_cat = std::max(max_cat, n_categories[l]);

    // Cells are independent: each writes only its own slot of `result`, so
    // any work split produces bit-identical output. Blocks are handed out
    // dynamically because query cost varies with local point density.
    const int BLOCK = 256;
    const int n_blocks = (N + BLOCK - 1) / BLOCK;
    std::atomic<int> next_block{0};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker = [&]() {
        try {
            KDTree::Scratch scratch(k, d);
            std::vector<double> P(k + 1), cat_prob(max_cat);
            for (;;) {
                int b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= n_blocks) break;
                int begin = b * BLOCK;
                int end = std::min(N, begin + BLOCK);
                for (int i = begin; i < end; ++i) {
                    tree.knn(i, k, scratch);
                    int n_nb = static_cast<int>(scratch.nn_idx.size());
                    for (int l = 0; l < n_labels; ++l) {
                        double simpson = compute_simpson_one(
                            scratch.nn_dist.data(), scratch.nn_idx.data(), n_nb,
                            lab.data() + static_cast<std::size_t>(l) * N,
                            n_categories[l], perplexity, P.data(), cat_prob.data()
                        );
                        result[static_cast<std::size_t>(l) * N + perm[i]] =
                            (simpson > 0) ? 1.0 / simpson : 0.0;
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(error_mutex);
            if (!first_error) first_error = std::current_exception();
        }
    };

    int n_workers = n_threads > 0 ? n_threads : default_thread_count();
    n_workers = std::max(1, std::min(n_workers, n_blocks));

    if (n_workers == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(n_workers - 1);
        for (int t = 0; t < n_workers - 1; ++t) pool.emplace_back(worker);
        worker();  // the calling thread takes part too
        for (std::thread& t : pool) t.join();
    }

    if (first_error) std::rethrow_exception(first_error);
    return result;
}

} // namespace lisi

#endif // LISI_HPP

#pragma once

#include <NTL/mat_GF2.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

// GCC/Clang, C++17, OpenMP. Basis vectors are ROWS.
// logical_basis must represent independent nonzero classes modulo
// rowspace(stabilizer_basis). This function does not verify that condition.
// Enumerates every nonzero logical combination plus every stabilizer combination.
inline long find_d(const NTL::mat_GF2& logical_basis,
                   const NTL::mat_GF2& stabilizer_basis,
                   bool show_progress = true)
{
    const long k = logical_basis.NumRows();
    const long r = stabilizer_basis.NumRows();
    const long n = logical_basis.NumCols();

    if (n != stabilizer_basis.NumCols())
        throw std::invalid_argument("Basis vectors must have equal lengths");
    if (k == 0)
        throw std::invalid_argument("No nontrivial logical operators");
    if (k >= 64 || r >= 64)
        throw std::invalid_argument("Too many basis rows for 64-bit enumeration");

    const std::uint64_t total1 = std::uint64_t{1} << k;
    const std::uint64_t total2 = std::uint64_t{1} << r;
    if (total1 - 1 > std::numeric_limits<std::uint64_t>::max() / total2)
        throw std::invalid_argument("Search count exceeds the 64-bit progress counter");
    const std::uint64_t total = (total1 - 1) * total2;
    const std::size_t words = static_cast<std::size_t>(n / 64 + (n % 64 != 0));

    // Convert once through NTL's public API. Padding bits stay zero.
    auto pack = [&](const NTL::mat_GF2& matrix) {
        const auto rows = static_cast<std::size_t>(matrix.NumRows());
        if (words != 0 && rows > std::numeric_limits<std::size_t>::max() / words)
            throw std::length_error("Packed matrix is too large");
        std::vector<std::uint64_t> result(rows * words, 0);
        for (long row = 0; row < matrix.NumRows(); ++row) {
            for (long col = 0; col < n; ++col) {
                if (NTL::IsOne(matrix[row][col])) {
                    result[static_cast<std::size_t>(row) * words + col / 64] |=
                        std::uint64_t{1} << (col % 64);
                }
            }
        }
        return result;
    };
    const auto logical = pack(logical_basis);
    const auto stabilizer = pack(stabilizer_basis);

    std::atomic<long> best{n};
    std::atomic<std::uint64_t> completed{0};
    const auto start = std::chrono::steady_clock::now();

    auto publish_best = [&](long value) {
        long old = best.load(std::memory_order_relaxed);
        while (value < old && !best.compare_exchange_weak(
                   old, value, std::memory_order_relaxed)) {}
    };

    auto report = [&] {
        const auto done = completed.load(std::memory_order_relaxed);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cerr << std::fixed << std::setprecision(2)
                  << "Progress: " << 100.0L * done / total << "%"
                  << " | checked: " << done << '/' << total
                  << " | best: " << best.load(std::memory_order_relaxed)
                  << " | elapsed: " << seconds << " s";
        if (done != 0 && seconds > 0) {
            const long double rate = static_cast<long double>(done) / seconds;
            std::cerr << " | rate: " << rate << " vectors/s"
                      << " | ETA: " << static_cast<long double>(total - done) / rate
                      << " s";
        } else {
            std::cerr << " | ETA: pending";
        }
        std::cerr << '\n' << std::flush;
    };

    // Fixed-size arrays let the compiler unroll short word loops and keep
    // candidates in registers. The vector fallback supports longer codewords.
    auto enumerate = [&](const auto& zero_candidate) {
        long minimum = n;
        #pragma omp parallel for reduction(min:minimum) schedule(dynamic, 1)
        for (std::uint64_t i = 1; i < total1; ++i) {
            auto candidate = zero_candidate;
            const auto count = candidate.size();
            std::uint64_t mask = i ^ (i >> 1);
            while (mask != 0) {
                const unsigned bit = __builtin_ctzll(mask);
                for (std::size_t w = 0; w < count; ++w)
                    candidate[w] ^= logical[bit * count + w];
                mask &= mask - 1;
            }

            long weight = 0;
            for (std::size_t w = 0; w < count; ++w)
                weight += __builtin_popcountll(candidate[w]);
            if (weight < minimum) {
                minimum = weight;
                publish_best(weight);
            }
            std::uint64_t pending = 1; // zero stabilizer combination

            for (std::uint64_t j = 1; j < total2; ++j) {
                const auto row = static_cast<std::size_t>(__builtin_ctzll(j));
                const auto offset = row * count;
                weight = 0;
                // Fused XOR and Hamming weight: no NTL calls in this loop.
                for (std::size_t w = 0; w < count; ++w) {
                    candidate[w] ^= stabilizer[offset + w];
                    weight += __builtin_popcountll(candidate[w]);
                }
                if (weight < minimum) {
                    minimum = weight;
                    publish_best(weight);
                }
                if (++pending == 4096) {
                    if (show_progress)
                        completed.fetch_add(pending, std::memory_order_relaxed);
                    pending = 0;
                }
            }
            if (show_progress && pending != 0)
                completed.fetch_add(pending, std::memory_order_relaxed);
        }
        return minimum;
    };

    std::atomic<bool> stop_monitor{false};
    std::thread monitor;
    if (show_progress) {
        monitor = std::thread([&] {
            while (!stop_monitor.load(std::memory_order_relaxed)) {
                report();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }
    auto stop = [&] {
        stop_monitor.store(true, std::memory_order_relaxed);
        if (monitor.joinable())
            monitor.join();
    };

    long answer;
    try {
        switch (words) {
        case 1: answer = enumerate(std::array<std::uint64_t, 1>{}); break;
        case 2: answer = enumerate(std::array<std::uint64_t, 2>{}); break;
        case 3: answer = enumerate(std::array<std::uint64_t, 3>{}); break;
        case 4: answer = enumerate(std::array<std::uint64_t, 4>{}); break;
        default: answer = enumerate(std::vector<std::uint64_t>(words, 0)); break;
        }
    } catch (...) {
        stop();
        throw;
    }
    stop();
    if (show_progress)
        report();
    return answer;
}

#pragma once

#include <NTL/mat_GF2.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>


inline long find_d(const NTL::mat_GF2& logical_basis,
                   const NTL::mat_GF2& stabilizer_basis,
                   bool show_progress = true)
{
    const long k = logical_basis.NumRows();
    const long r = stabilizer_basis.NumRows();

    if (logical_basis.NumCols() != stabilizer_basis.NumCols())
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

    long d = logical_basis.NumCols();
    std::atomic<long> best{d};
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
        const long double percent = 100.0L * done / total;

        // New lines render reliably in the CLion Run console.
        std::cerr << std::fixed << std::setprecision(2)
                  << "Progress: " << percent << "%"
                  << " | checked: " << done << '/' << total
                  << " | best: " << best.load(std::memory_order_relaxed)
                  << " | elapsed: " << seconds << " s";
        if (done != 0 && seconds > 0) {
            const long double rate = static_cast<long double>(done) / seconds;
            const long double eta = static_cast<long double>(total - done) / rate;
            std::cerr << " | rate: " << rate << " vectors/s"
                      << " | ETA: " << eta << " s";
        } else {
            std::cerr << " | ETA: pending";
        }
        std::cerr << '\n' << std::flush;
    };

    std::jthread monitor;
    if (show_progress) {
        monitor = std::jthread([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                report();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }

    #pragma omp parallel for reduction(min:d) schedule(dynamic, 1)
    for (std::uint64_t i = 1; i < total1; ++i) {
        const std::uint64_t mask = i ^ (i >> 1);
        NTL::vec_GF2 candidate;
        candidate.SetLength(logical_basis.NumCols());
        for (long bit = 0; bit < k; ++bit)
            if ((mask >> bit) & 1)
                candidate += logical_basis[bit];

        long weight = NTL::weight(candidate);
        if (weight < d) {
            d = weight;
            publish_best(weight);
        }
        // Include the zero stabilizer combination in the progress count.
        std::uint64_t pending = 1;

        for (std::uint64_t j = 1; j < total2; ++j) {
            candidate += stabilizer_basis[std::countr_zero(j)];
            weight = NTL::weight(candidate);
            if (weight < d) {
                d = weight;
                publish_best(weight);
            }

            // Batch counter updates to avoid an atomic operation per vector.
            if (++pending == 4096) {
                completed.fetch_add(pending, std::memory_order_relaxed);
                pending = 0;
            }
        }
        if (pending != 0)
            completed.fetch_add(pending, std::memory_order_relaxed);
    }

    if (show_progress) {
        monitor.request_stop();
        monitor.join();
        report(); // Final count is exactly total, so progress is 100%.
    }
    return d;
}

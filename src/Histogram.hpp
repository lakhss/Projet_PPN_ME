#pragma once

#include <cstddef>

struct BinStats {
    std::size_t count = 0;
    double sum = 0.0;
    double sq_sum = 0.0;
};

struct HistogramSplit {
    int feature_idx = -1;
    double threshold = 0.0;
    double gain = -1.0;
};
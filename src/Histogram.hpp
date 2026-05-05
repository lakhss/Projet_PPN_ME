#pragma once
#include <cstdint>

// alignas(64) force l'alignement sur une frontière de ligne de cache (64 octets).

struct alignas(64) FastBinStats {
    uint32_t count = 0;
    double sum = 0.0;
    double sq_sum = 0.0;
};

struct HistogramSplit {
    int feature_idx = -1;
    uint8_t threshold_bin = 0; 
    double gain = -1.0;
};
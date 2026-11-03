#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <ttypt/iso.h>

ISO_EXPORT(float, sepal_cosine, const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return (float)(dot / sqrt(na * nb));
}

ISO_EXPORT(int, sepal_wasm_sketch, const float *v, size_t n, uint64_t *words, size_t nwords) {
    size_t req = (n + 63) / 64;
    if (nwords < req) return -1;
    for (size_t w = 0; w < req; w++) {
        uint64_t word = 0;
        size_t limit = (w + 1) * 64 < n ? 64 : (n - w * 64);
        for (size_t b = 0; b < limit; b++) {
            if (v[w * 64 + b] >= 0.0f) {
                word |= ((uint64_t)1 << b);
            }
        }
        words[w] = word;
    }
    return 0;
}

ISO_EXPORT(uint32_t, sepal_wasm_hamming, const uint64_t *a, const uint64_t *b, size_t nwords) {
    uint32_t dist = 0;
    for (size_t i = 0; i < nwords; i++) {
        dist += __builtin_popcountll(a[i] ^ b[i]);
    }
    return dist;
}

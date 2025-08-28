#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

void vector_dot_vector(const uint8_t *a, const uint8_t *b, size_t bit_len, uint32_t result[8]) {
    size_t bytes = (bit_len + 7) / 8;
    size_t i = 0;

    __m256i acc = _mm256_setzero_si256();
    int bit_shift = 0;

    // Process 32 bytes (256 bits) at a time
    #pragma GCC unroll 8
    for (; i + 31 < bytes; i += 32) {
        if (b[i/ 32] & (1 << bit_shift)) {
            // If the bit is set, we want to XOR with va 
            __m256i va = _mm256_load_si256((const __m256i *)(a + i));
            acc = _mm256_xor_si256(acc, va);
        }
        bit_shift = (bit_shift + 1) % 32; // Cycle through bits 0-31
    }

    // Reduce acc to 256 bits → 4×64-bit words
    uint32_t *acc32 = (uint32_t *)&acc;
    for (; i < bytes; i += 4) {
        uint32_t a32 = *(uint32_t *)(&a[i]);
        uint32_t b32 = *(uint32_t *)(&b[i]);
        acc32[i % 4] ^= a32 & b32;
    }
    memcpy(result, acc32, sizeof(uint32_t) * 8);

    return;
    // uint32_t *acc32 = (uint32_t *)&acc;
    // int total = 0;
    // for (int j = 0; j < 4; ++j) {
    //     total ^= __builtin_popcountll(acc64[j]);
    // }

    // // Handle remaining tail bytes
    // for (; i < bytes; ++i) {
    //     total ^= __builtin_popcount(a[i] & b[i]);
    // }

    // return total & 1; // Return parity (mod 2)
}
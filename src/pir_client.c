#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

void fill_random_bools(uint8_t *array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = rand() % 2;  // random boolean [0, 1]
    }
}

void create_queries(size_t query_size, uint8_t **q1, uint8_t **q2, uint8_t **q3) {
    *q1 = calloc(query_size, 1);
    *q2 = calloc(query_size, 1);
    *q3 = calloc(query_size, 1);
    if (!*q1 || !*q2 || !*q3) {
        perror("Error allocating memory for queries");
        free(*q1);
        free(*q2);
        free(*q3);
        return;
    }
    fill_random_bools(*q1, query_size);
    fill_random_bools(*q2, query_size);
    fill_random_bools(*q3, query_size);
}

__m256i * reconstruct(size_t i1, size_t i2, size_t i3, __m256i *server_1_results, __m256i *server_2_results, size_t results_size, size_t blocks_per_entry) {
    size_t query_size = (results_size - 1) / 3;
    __m256i *read_entry = aligned_alloc(32, blocks_per_entry * sizeof(__m256i));
    if (!read_entry) {
        perror("Error allocating memory for read_entry");
        return 0;
    }
    for (int block = 0; block < blocks_per_entry; block++) {
        read_entry[block] = server_1_results[query_size * 3 * blocks_per_entry + block];
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_2_results[query_size * 3 * blocks_per_entry + block]);

        read_entry[block] = _mm256_xor_si256(read_entry[block], server_1_results[i1 * blocks_per_entry + block]);
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_2_results[i1 * blocks_per_entry + block]);

        read_entry[block] = _mm256_xor_si256(read_entry[block], server_1_results[query_size * blocks_per_entry + i2 * blocks_per_entry + block]);
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_2_results[query_size * blocks_per_entry + i2 * blocks_per_entry + block]);

        read_entry[block] = _mm256_xor_si256(read_entry[block], server_1_results[2 * query_size * blocks_per_entry + i3 * blocks_per_entry + block]);
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_2_results[2 * query_size * blocks_per_entry + i3 * blocks_per_entry + block]);
    }

    return read_entry;
}
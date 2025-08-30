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

void create_queries(size_t query_size, uint8_t **q1, uint8_t **q2) {
    *q1 = calloc(query_size, 1);
    *q2 = calloc(query_size, 1);
    if (!*q1 || !*q2) {
        perror("Error allocating memory for queries");
        free(*q1);
        free(*q2);
        return;
    }
    fill_random_bools(*q1, query_size);
    fill_random_bools(*q2, query_size);
}

__m256i *reconstruct(__m256i *server_1_results, __m256i *server_2_results, __m256i *server_3_results, __m256i *server_4_results, size_t blocks_per_entry)
{
    __m256i *read_entry = aligned_alloc(32, blocks_per_entry * sizeof(__m256i));
    if (!read_entry) {
        perror("Error allocating memory for read_entry");
        return 0;
    }
    for (size_t block = 0; block < blocks_per_entry; block++) {
        read_entry[block] = server_1_results[block];
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_2_results[block]);
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_3_results[block]);
        read_entry[block] = _mm256_xor_si256(read_entry[block], server_4_results[block]);
    }

    return read_entry;
}
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
#include "pir_server.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

pir_server_t *pir_server_alloc(uint8_t *db, size_t db_size, size_t blocks_per_entry)
{
    pir_server_t *server = malloc(sizeof(pir_server_t));
    if (!server) {
        perror("Error allocating memory for PIR server");
        return NULL;
    }
    server->db = db;
    server->db_size_bits = db_size * 8;
    server->blocks_per_entry = blocks_per_entry;
    server->query_size = cbrt(db_size / (BLOCK_SIZE * blocks_per_entry));
    return server;
}

void process_slices(const uint8_t *db, size_t bit_len, size_t block_per_entry, size_t query_size, uint8_t *q1, uint8_t *q2, uint8_t *q3, uint8_t dim, __m256i *results) {
    size_t bytes = (bit_len + 7) / 8;

    // Precompute active indices for q1, q2, q3 to reduce branch misprediction
    size_t active_i1[query_size], active_i2[query_size], active_i3[query_size];
    size_t n1 = 0, n2 = 0, n3 = 0;
    for (size_t i = 0; i < query_size; i++)
    {
        if (q1[i])
            active_i1[n1++] = i;
        if (q2[i])
            active_i2[n2++] = i;
        if (q3[i])
            active_i3[n3++] = i;
    }
    size_t *active_slice = NULL;
    if (dim == 0) {
        active_slice = active_i1;
        n1 = query_size;
    }
    else if (dim == 1) {
        active_slice = active_i2;
        n2 = query_size;
    }
    else if (dim == 2) {
        active_slice = active_i3;
        n3 = query_size;
    }
    if (dim <= 2) {
        for (size_t i = 0; i < query_size; i++) {
            active_slice[i] = i;
        }
        for (size_t slice = 0; slice < query_size; slice++) {
            for (size_t block = 0; block < block_per_entry; block++) {
                results[slice * block_per_entry + block] = _mm256_setzero_si256();
            }
        }
    }
    else {
        for (size_t block = 0; block < block_per_entry; block++) {
            results[block] = _mm256_setzero_si256();
        }
    }

    size_t prefetch_lines_count = min(PREFETCH_LINES_COUNT, block_per_entry * BLOCK_SIZE / CACHE_LINE_SIZE);
    for (size_t a1 = 0; a1 < n1; a1++)
    {
        size_t i1 = active_i1[a1];
        for (size_t a2 = 0; a2 < n2; a2++)
        {
            size_t i2 = active_i2[a2];
            for (size_t a3 = 0; a3 < n3; a3++)
            {
                size_t i3 = active_i3[a3];
                size_t base_index = (i1 * query_size * query_size + i2 * query_size + i3) * (BLOCK_SIZE * block_per_entry);
                if (base_index + (BLOCK_SIZE * block_per_entry) > bytes)
                    continue;
                if (a3 + 1 < n3)
                {
                    size_t prefetch_base_index = (i1 * query_size * query_size + i2 * query_size + active_i3[a3 + 1]) * (BLOCK_SIZE * block_per_entry);
                    for (size_t cache_line = 0; cache_line < prefetch_lines_count; cache_line++)
                    {
                        __builtin_prefetch(db + prefetch_base_index + cache_line * CACHE_LINE_SIZE, 0, 3);
                    }
                }
                size_t slice = i1 * (dim == 0) + i2 * (dim == 1) + i3 * (dim == 2);
                for (size_t block = 0; block < block_per_entry; block++) {
                    __m256i va1 = _mm256_load_si256((const __m256i *)(db + base_index + block * BLOCK_SIZE));
                    results[slice * block_per_entry + block] = _mm256_xor_si256(results[slice * block_per_entry + block], va1);
                }
            }
        }
    }
}

__m256i *pir_answer(pir_server_t *self, uint8_t *q1, uint8_t *q2, uint8_t *q3) {
    __m256i *result = aligned_alloc(64, (self->query_size * 3 + 1) * sizeof(__m256i) * self->blocks_per_entry);
    if (!result) {
        perror("Error allocating memory for results");
        return NULL;
    }
    __m256i *main_slice_result = &result[self->query_size * 3 * self->blocks_per_entry];
    process_slices(self->db, self->db_size_bits, self->blocks_per_entry, self->query_size, q1, q2, q3, 0xff, main_slice_result);

    for (uint8_t dim = 0; dim < 3; dim++) {
        process_slices(self->db, self->db_size_bits, self->blocks_per_entry, self->query_size, q1, q2, q3, dim, &result[dim * self->blocks_per_entry * self->query_size]);
    }
    for (size_t i = 0; i < self->query_size * 3; i++) {
        for (size_t block = 0; block < self->blocks_per_entry; block++) {
            result[i * self->blocks_per_entry + block] = _mm256_xor_si256(result[i * self->blocks_per_entry + block], main_slice_result[block]);
        }
    }
    return result;
}
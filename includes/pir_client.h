#ifndef PIR_CLIENT_H
#define PIR_CLIENT_H

#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

void create_queries(size_t query_size, uint8_t **q1, uint8_t **q2, uint8_t **q3);
__m256i * reconstruct(size_t i1, size_t i2, size_t i3, __m256i *server_1_results, __m256i *server_2_results, size_t results_size, size_t blocks_per_entry);

#endif /* PIR_CLIENT_H */
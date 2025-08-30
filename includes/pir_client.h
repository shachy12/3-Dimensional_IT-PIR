#ifndef PIR_CLIENT_H
#define PIR_CLIENT_H

#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

void create_queries(size_t query_size, uint8_t **q1, uint8_t **q2);
__m256i *reconstruct(__m256i *server_1_results, __m256i *server_2_results, __m256i *server_3_results, __m256i *server_4_results, size_t blocks_per_entry);

#endif /* PIR_CLIENT_H */
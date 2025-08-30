#ifndef PIR_SERVER_H
#define PIR_SERVER_H

#include <immintrin.h>
#include <stdint.h>

#define BLOCK_SIZE (sizeof(__m256i))
#define CACHE_LINE_SIZE (64)
#define PREFETCH_LINES_COUNT (32)

typedef struct {
    uint8_t *db;
    size_t db_size_bits;
    size_t blocks_per_entry;
    size_t query_size;
} pir_server_t;

pir_server_t *pir_server_alloc(uint8_t *db, size_t db_size, size_t blocks_per_entry);
__m256i *pir_answer(pir_server_t *self, uint8_t *q1, uint8_t *q2, uint8_t *q3);

#endif
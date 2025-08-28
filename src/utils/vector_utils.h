#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H
#include <stdint.h>

void vector_dot_vector(const uint8_t *a, const uint8_t *b, size_t bit_len, uint32_t result[8]);

#endif
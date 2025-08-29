#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <utils/vector_utils.h>

#define SAMPLES (100)
#define BLOCK_SIZE (32)
#define BLOCKS_PER_ENTRY (128)
#define CACHE_LINE_SIZE (64)
#define PREFETCH_LINES_COUNT (32)


void * create_db(size_t size) {
    uint8_t *db = aligned_alloc(64, size);
    for (size_t i = 0; i < size; i++) {
        db[i] = rand() % 256;
    }
    return db;
}

void fill_random_bools(uint8_t *array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = rand() % 2;  // random boolean [0, 1]
    }
}

void process_pir_variant0(const uint8_t *db, size_t bit_len, size_t query_size, uint8_t *q1, uint8_t *q2, uint8_t *q3, __m256i *result) {
    size_t bytes = (bit_len + 7) / 8;
    __m256i acc = _mm256_setzero_si256();

    // Precompute active indices for q1, q2, q3 to reduce branch misprediction
    size_t active_i1[query_size], active_i2[query_size], active_i3[query_size];
    size_t n1 = 0, n2 = 0, n3 = 0;
    for (size_t i = 0; i < query_size; i++) {
        if (q1[i]) active_i1[n1++] = i;
        if (q2[i]) active_i2[n2++] = i;
        if (q3[i]) active_i3[n3++] = i;
    }

    for (size_t a1 = 0; a1 < n1; a1++) {
        size_t i1 = active_i1[a1];
        for (size_t a2 = 0; a2 < n2; a2++) {
            size_t i2 = active_i2[a2];
            for (size_t a3 = 0; a3 < n3; a3++) {
                size_t i3 = active_i3[a3];
                size_t base_index = (i1 * query_size * query_size + i2 * query_size + i3) * 32;
                if (base_index + 32 > bytes) continue;
                // Use unaligned load for safety and flexibility
                __m256i va = _mm256_loadu_si256((const __m256i *)(db + base_index));
                acc = _mm256_xor_si256(acc, va);
            }
        }
    }
    *result = acc;
}

void prefetch_data(const uint8_t *db, size_t i1, size_t i2, size_t *active_i3, size_t n3, size_t query_size, size_t bytes, size_t start_i3, size_t window_size)
{
    for (size_t a3 = start_i3; a3 < start_i3 + window_size; a3++)
    {
        if (a3 >= n3)
            break;
        size_t i3 = active_i3[a3];
        // Prefetch next db block to L1 cache
        size_t prefetch_index = (i1 * query_size * query_size + i2 * query_size + i3) * 64;
        if (prefetch_index + 64 < bytes)
        {
            for (size_t block = 0; block < BLOCKS_PER_ENTRY; block++) {
                __builtin_prefetch(db + prefetch_index + block * 32);
            }
        }
    }
}

void process_slices(const uint8_t *db, size_t bit_len, size_t query_size, uint8_t *q1, uint8_t *q2, uint8_t *q3, uint8_t dim, __m256i *results) {
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
            for (size_t block = 0; block < BLOCKS_PER_ENTRY; block++) {
                results[slice * BLOCKS_PER_ENTRY + block] = _mm256_setzero_si256();
            }
        }
    }
    else {
        for (size_t block = 0; block < BLOCKS_PER_ENTRY; block++) {
            results[block] = _mm256_setzero_si256();
        }
    }

    for (size_t a1 = 0; a1 < n1; a1++)
    {
        size_t i1 = active_i1[a1];
        for (size_t a2 = 0; a2 < n2; a2++)
        {
            size_t i2 = active_i2[a2];
            for (size_t a3 = 0; a3 < n3; a3++)
            {
                size_t i3 = active_i3[a3];
                size_t base_index = (i1 * query_size * query_size + i2 * query_size + i3) * (BLOCK_SIZE * BLOCKS_PER_ENTRY);
                if (base_index + (BLOCK_SIZE * BLOCKS_PER_ENTRY) > bytes)
                    continue;
                if (a3 + 1 < n3)
                {
                    size_t prefetch_base_index = (i1 * query_size * query_size + i2 * query_size + active_i3[a3 + 1]) * (BLOCK_SIZE * BLOCKS_PER_ENTRY);
                    for (size_t cache_line = 0; cache_line < PREFETCH_LINES_COUNT; cache_line++)
                    {
                        __builtin_prefetch(db + prefetch_base_index + cache_line * CACHE_LINE_SIZE, 0, 3);
                    }
                }
                size_t slice = i1 * (dim == 0) + i2 * (dim == 1) + i3 * (dim == 2);
                for (size_t block = 0; block < BLOCKS_PER_ENTRY; block++) {
                    __m256i va1 = _mm256_load_si256((const __m256i *)(db + base_index + block * BLOCK_SIZE));
                    results[slice * BLOCKS_PER_ENTRY + block] = _mm256_xor_si256(results[slice * BLOCKS_PER_ENTRY + block], va1);
                }
            }
        }
    }
}

__m256i *process_pir_variant1(const uint8_t *db, size_t bit_len, size_t query_size, uint8_t *q1, uint8_t *q2, uint8_t *q3) {
    __m256i acc[BLOCKS_PER_ENTRY] = {_mm256_setzero_si256()};
    __m256i *result = aligned_alloc(64, (query_size * 3 + 1) * sizeof(__m256i) * BLOCKS_PER_ENTRY);
    if (!result) {
        perror("Error allocating memory for results");
        return NULL;
    }
    process_slices(db, bit_len, query_size, q1, q2, q3, 0xff, acc);
    result[query_size * 3 * BLOCKS_PER_ENTRY] = acc[0];
    result[query_size * 3 * BLOCKS_PER_ENTRY + 1] = acc[1];
    for (uint8_t dim = 0; dim < 3; dim++) {
        process_slices(db, bit_len, query_size, q1, q2, q3, dim, &result[dim * BLOCKS_PER_ENTRY * query_size]);
    }
    for (size_t i = 0; i < query_size * 3; i++) {
        for (size_t block = 0; block < BLOCKS_PER_ENTRY; block++) {
            result[i * BLOCKS_PER_ENTRY + block] = _mm256_xor_si256(acc[block], result[i * BLOCKS_PER_ENTRY + block]);
        }
    }
    return result;
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
    // memset(*q1, 0, query_size);
    // memset(*q2, 0, query_size);
    // memset(*q3, 0, query_size);
    // // (*q1)[1] = 1;
    // (*q2)[1] = 1;
    // (*q3)[1] = 1;
}

double test_2_servers_pir(uint8_t *db, size_t bit_len) {
    uint16_t i1 = 0, i2 = 0, i3 = 0;
    size_t query_size = cbrt(bit_len / (BLOCK_SIZE * BLOCKS_PER_ENTRY * 8));
    i1 = rand() % query_size;
    i2 = rand() % query_size;
    i3 = rand() % query_size;
    uint8_t *q1, *q2, *q3;
    __m256i *results_0, *results_1;
    create_queries(query_size, &q1, &q2, &q3);

    printf("query_size: %zu\n", query_size);

    double elapsed = 0;
    clock_t start = clock();
    results_0 = process_pir_variant1(db, bit_len, query_size, q1, q2, q3);
    clock_t end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;
    start = clock();
    q1[i1] ^= 1;
    q2[i2] ^= 1;
    q3[i3] ^= 1;
    results_1 = process_pir_variant1(db, bit_len, query_size, q1, q2, q3);
    end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;
    double throughput = bit_len / (8 * 1024.0 * 1024.0) / (elapsed / 2);

    printf("Elapsed time: %.6f seconds\n", elapsed);
    printf("throughput per server: %.2f MB/s\n",  throughput);

    __m256i read_entry = results_0[query_size * 3 * BLOCKS_PER_ENTRY];
    read_entry = _mm256_xor_si256(read_entry, results_1[query_size * 3 * BLOCKS_PER_ENTRY]);
// 
    read_entry = _mm256_xor_si256(read_entry, results_0[i1 * BLOCKS_PER_ENTRY]);
    read_entry = _mm256_xor_si256(read_entry, results_1[i1 * BLOCKS_PER_ENTRY]);
// 
    read_entry = _mm256_xor_si256(read_entry, results_0[query_size * BLOCKS_PER_ENTRY + i2 * BLOCKS_PER_ENTRY]);
    read_entry = _mm256_xor_si256(read_entry, results_1[query_size * BLOCKS_PER_ENTRY + i2 * BLOCKS_PER_ENTRY]);
// 
    read_entry = _mm256_xor_si256(read_entry, results_0[2 * query_size * BLOCKS_PER_ENTRY + i3 * BLOCKS_PER_ENTRY]);
    read_entry = _mm256_xor_si256(read_entry, results_1[2 * query_size * BLOCKS_PER_ENTRY + i3 * BLOCKS_PER_ENTRY]);
    printf("entry: %08x\n", _mm256_extract_epi32(read_entry, 0));
    uint32_t entry = _mm256_extract_epi32(read_entry, 0);
    printf("db entry: %ld\n", (i1 * query_size * query_size + i2 * query_size + i3) * (BLOCK_SIZE * BLOCKS_PER_ENTRY));
    if (memcmp(&entry, &db[(i1 * query_size * query_size + i2 * query_size + i3) * (BLOCK_SIZE * BLOCKS_PER_ENTRY)], 4) != 0) {
        printf("Error: entry does not match database\n");
    } else {
        printf("Success: entry matches database\n");
    }

    free(q1);
    free(q2);
    free(q3);
    free(results_0);
    free(results_1);

    return throughput;
}

void test_4_servers_pir(uint8_t *db, size_t bit_len) {
    uint16_t i1 = 0, i2 = 0, i3 = 0;
    size_t query_size = cbrt(bit_len / 256); // each entry is 256 bits (32 bytes) 
    uint8_t *q1, *q2, *q3;
    __m256i results[8];
    memset(results, 0, sizeof(results));
    create_queries(query_size, &q1, &q2, &q3);

    printf("query_size: %zu\n", query_size);

    double elapsed = 0;
    for (int s = 0; s < 8; s++){
        q1[i1] ^= s >> 2;
        q2[i2] ^= s >> 1 & 1;
        q3[i3] ^= s & 1;
        clock_t start = clock();
        process_pir_variant0(db, bit_len, query_size, q1, q2, q3, &results[s]);
        clock_t end = clock();
        elapsed += (double)(end - start) / CLOCKS_PER_SEC;
        q1[i1] ^= s >> 2;
        q2[i2] ^= s >> 1 & 1;
        q3[i3] ^= s & 1;
    }
    printf("Elapsed time: %.6f seconds\n", elapsed);
    printf("throughput per server: %.2f MB/s\n",  bit_len / (8 * 1024.0 * 1024.0) / (elapsed / 8));
    printf("throughput for 4 servers in one: %.2f MB/s\n",  bit_len / (8 * 1024.0 * 1024.0) / (elapsed / 2));

    __m256i read_entry = results[0];
    for (int i = 1; i < 8; i++) {
        read_entry = _mm256_xor_si256(read_entry, results[i]);
    }
    printf("entry: %08x\n", _mm256_extract_epi32(read_entry, 0));

    free(q1);
    free(q2);
    free(q3);
}

double average_throughput(double *throughputs, int count) {
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += throughputs[i];
    }
    return sum / count;
}

double std_throughput(double *throughputs, int count) {
    double avg = average_throughput(throughputs, count);
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += (throughputs[i] - avg) * (throughputs[i] - avg);
    }
    return sqrt(sum / count);
}

int main(void) {
    size_t file_size = (1 * 1024 * 1024 * 1024);
    uint8_t *db = create_db(file_size);
    if (!db) {
        return 1;
    }

    test_4_servers_pir(db, file_size * 8);

    printf("\nTesting 2 servers:\n");
    double throughput_agg = 0;
    (void)test_2_servers_pir(db, file_size * 8);

    // 100 samples, ignoring first one
    double throughputs[SAMPLES];
    throughput_agg = 0;
    for (int i = 0; i < SAMPLES; i++) {
        throughputs[i] = test_2_servers_pir(db, file_size * 8);
        throughput_agg += throughputs[i];
    }
    double avg = average_throughput(throughputs, SAMPLES);
    double std = std_throughput(throughputs, SAMPLES);
    printf("Average throughput for 100 samples: %.2f MB/s, Std: %.2f MB/s\n", avg, std);
    free(db);
    return 0;
}
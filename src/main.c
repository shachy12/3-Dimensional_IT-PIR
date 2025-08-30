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
#include "pir_client.h"
#include "pir_server.h"


void * create_db(size_t size) {
    uint8_t *db = aligned_alloc(64, size);
    for (size_t i = 0; i < size; i++) {
        db[i] = rand() % 256;
    }
    return db;
}

double test_2_servers_pir(pir_server_t *server) {
    uint16_t i1 = 0, i2 = 0;

    // choose random index
    i1 = rand() % server->query_size;
    i2 = rand() % server->query_size;

    //create queries
    uint8_t *q1, *q2;
    __m256i *results_0, *results_1, *results_2, *results_3;
    create_queries(server->query_size, &q1, &q2);

    printf("query_size: %zu\n", server->query_size);

    // Simulate servers and measure times for throughput calculation
    double elapsed = 0;
    clock_t start = clock();
    results_0 = pir_answer(server, q1, q2); // server 1
    clock_t end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;

    start = clock();
    q1[i1] ^= 1;
    results_1 = pir_answer(server, q1, q2); // server 2
    end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;

    start = clock();
    q2[i2] ^= 1;
    results_2 = pir_answer(server, q1, q2); // server 3
    end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;

    start = clock();
    q1[i1] ^= 1;
    results_3 = pir_answer(server, q1, q2); // server 3
    end = clock();
    elapsed += (double)(end - start) / CLOCKS_PER_SEC;

    double throughput = server->db_size_bits / (8 * 1024.0 * 1024.0) / (elapsed / 4);
    printf("Elapsed time: %.6f seconds\n", elapsed);
    printf("throughput per server: %.2f MB/s\n",  throughput);
    
    // reconstruct entry and verify data
    __m256i *entry = reconstruct(results_0, results_1, results_2, results_3, server->blocks_per_entry);
    size_t db_index = (i1 * server->query_size + i2) * (BLOCK_SIZE * server->blocks_per_entry);
    for (size_t block = 0; block < server->blocks_per_entry; block++) {
        if (memcmp(&entry[block], &server->db[db_index + block * BLOCK_SIZE], BLOCK_SIZE) != 0) {
            printf("block error: %ld, %08x != %08x\n", block, _mm256_extract_epi32(entry[block], 0), *(uint32_t *)&server->db[db_index + block * BLOCK_SIZE]);
            printf("%08x,%08x,%08x,%08x\n", _mm256_extract_epi32(results_0[block], 0), _mm256_extract_epi32(results_1[block], 0), _mm256_extract_epi32(results_2[block], 0), _mm256_extract_epi32(results_3[block], 0));
            break;
        }
    }
    if (memcmp(entry, &server->db[db_index], server->blocks_per_entry * BLOCK_SIZE) != 0) {
        printf("Error: entry does not match database\n");
    } else {
        printf("Success: entry matches database\n");
    }

    // free resources
    free(entry);
    free(q1);
    free(q2);
    free(results_0);
    free(results_1);

    return throughput;
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

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: ./pir_server <DB_SIZE> <BLOCKS_PER_ENTRY> <SAMPLES>\n"
               "- DB_SIZE               The LOG on the tested DB Bytes size\n"
               "- BLOCKS_PER_ENTRY      The number of blocks per entry, each block is 128 bits\n"
               "- SAMPLES               The number of samples to collect\n"
               "\nExample: create a database with 2^30 bytes, 128 blocks per entry and run 100 experiments to sample throughput\n"
               "./pir_server 30 128 100\n");
        return 1;
    }
    int log_db_size = atoi(argv[1]);
    int blocks_per_entry = atoi(argv[2]);
    int samples = atoi(argv[3]);
    size_t file_size = 1UL << log_db_size;
    printf("Generating DB of size %zu bytes, blocks per entry: %d\n", file_size, blocks_per_entry);
    uint8_t *db = create_db(file_size);
    if (!db) {
        return 1;
    }

    pir_server_t *server = pir_server_alloc(db, file_size, blocks_per_entry);
    if (!server) {
        free(db);
        perror("Error allocating PIR server");
        return 1;
    }
    (void)test_2_servers_pir(server);

    // 100 samples, ignoring first one
    double throughput_agg = 0;
    double throughputs[samples];
    throughput_agg = 0;
    for (int i = 0; i < samples; i++) {
        throughputs[i] = test_2_servers_pir(server);
        throughput_agg += throughputs[i];
    }
    double avg = average_throughput(throughputs, samples);
    double std = std_throughput(throughputs, samples);
    printf("Average throughput per server for %d samples: %.2f MB/s, Std: %.2f MB/s\n", samples, avg, std);
    free(db);
    return 0;
}
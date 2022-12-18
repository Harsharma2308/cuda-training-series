// Copyright
// Author:
#include <stdio.h>

#include <random>
#include <vector>
using namespace std;
#define SIZE 5
#define BLOCK_SIZE 2

#define PRINT

// copied error checking macro
#define cudaCheckErrors(msg)                             \
  do {                                                   \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess) {                          \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

__global__ void add(float *a, float *b, float *c) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id < SIZE) {
    c[global_thread_id] = a[global_thread_id] + b[global_thread_id];
  }
}

void fill_vector(float *vec) {
  for (int i = 0; i < SIZE; i++) {
    vec[i] = (rand() / static_cast<float>(RAND_MAX)) * 50;
  }
}

void print_vec(float *vec) {
  for (int i = 0; i < SIZE; i++)
    printf("%.2f ", vec[i]);
  printf("\n");
}

void sum1D() {
  float *H_a = new float[SIZE];
  float *H_b = new float[SIZE];
  float *H_c = new float[SIZE];
  float *D_a, *D_b, *D_c;
  cudaMalloc(&D_a, SIZE * sizeof(float));
  cudaMalloc(&D_b, SIZE * sizeof(float));
  cudaMalloc(&D_c, SIZE * sizeof(float));

  // Populate host vectors
  fill_vector(H_a);
  fill_vector(H_b);
#ifdef PRINT
  print_vec(H_a);
  print_vec(H_b);
#endif

  // Transfer to device
  cudaMemcpy(D_a, H_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(D_b, H_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

  // kernel launch
  add<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(D_a, D_b, D_c);

  cudaMemcpy(H_c, D_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef PRINT
  print_vec(H_c);
#endif

  // Cleanup
  free(H_a);
  free(H_b);
  free(H_c);
  cudaFree(D_a);
  cudaFree(D_b);
  cudaFree(D_c);
}

void sum2D() {
  float *H_a = new float[SIZE];
  float *H_b = new float[SIZE];
  float *H_c = new float[SIZE];
  float *D_a, *D_b, *D_c;
  cudaMalloc(&D_a, SIZE * sizeof(float));
  cudaMalloc(&D_b, SIZE * sizeof(float));
  cudaMalloc(&D_c, SIZE * sizeof(float));

  // Populate host vectors
  fill_vector(H_a);
  fill_vector(H_b);
#ifdef PRINT
  print_vec(H_a);
  print_vec(H_b);
#endif

  // Transfer to device
  cudaMemcpy(D_a, H_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(D_b, H_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

  // kernel launch
  add<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(D_a, D_b, D_c);

  cudaMemcpy(H_c, D_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef PRINT
  print_vec(H_c);
#endif

  // Cleanup
  free(H_a);
  free(H_b);
  free(H_c);
  cudaFree(D_a);
  cudaFree(D_b);
  cudaFree(D_c);
}
int main() {
  // Allocate memory on device and host
  sum1D();
  return 0;
}

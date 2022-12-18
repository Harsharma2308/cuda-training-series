#include <stdio.h>
#include <vector>
#include <random>
#include <iostream>
using namespace std;

#define ROW1 3
#define COL1 2
#define ROW2 COL1
#define COL2 4
#define BLOCK_SIZE 2
#define PRINT

// copied error checking macro
#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

__global__ void matrixMul(float *vec1, float *vec2, float *result)
{

  // We want resultant vector indices - [row index idx, column index idy] to correspond to global thread indices x and y respectively
  int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x; // making this row number in target
  int idy = threadIdx.y + BLOCK_SIZE * blockIdx.y; // making this col number in target

  if (idx < ROW1 && idy < COL2)
  {
    #ifdef DEBUG
    printf("Index idx,idy : %d , %d \n", idx, idy);
    #endif
    result[idx * COL2 + idy] = 0;
    for (int i = 0; i < COL1; i++)
    {
      #ifdef DEBUG
      printf("Multiplying : %f , %f at %d,%d \n", vec1[idx * COL1 + i], vec2[idy + COL2 * i], idx , idy);
      #endif
      result[idx * COL2 + idy] += vec1[idx * COL1 + i] * vec2[idy + COL2 * i];
      #ifdef DEBUG
      printf("Result at %d,%d : %f \n", idx , idy, result[idx * COL2 + idy]);
      #endif
    }
  }
}

void fill_2DVec(float *vec, int col, int row)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      vec[i * col + j] = int((rand() / (float)RAND_MAX) * 10); // 0 to 5
    }
  }
}

#ifdef PRINT
void print_2DVec(float *vec, int col, int row)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%.2f ", vec[i * col + j]);
    }
    printf("\n");
  }

  printf("\n\n\n");
}
#endif

int main()
{
  float *h_a = new float[ROW1 * COL1];
  float *h_b = new float[ROW2 * COL2];
  float *h_c = new float[ROW1 * COL2];
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, ROW1 * COL1 * sizeof(float));
  cudaMalloc(&d_b, ROW2 * COL2 * sizeof(float));
  cudaMalloc(&d_c, ROW1 * COL2 * sizeof(float));

  // Populate host vectors
  fill_2DVec(h_a, COL1, ROW1);
  fill_2DVec(h_b, COL2, ROW2);

#ifdef PRINT
  std::cout << "Printing input matrices :" << std::endl;
  print_2DVec(h_a, COL1, ROW1);
  print_2DVec(h_b, COL2, ROW2);
#endif

  // Transfer to device
  cudaMemcpy(d_a, h_a, ROW1 * COL1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, ROW2 * COL2 * sizeof(float), cudaMemcpyHostToDevice);

  // kernel launch
  dim3 block(BLOCK_SIZE, BLOCK_SIZE); // x are rows and y are cols
  dim3 grid((ROW1 + block.x - 1) / block.x, (COL2 + block.y - 1) / block.y);
  matrixMul<<<grid, block>>>(d_a, d_b, d_c);

  // Transfer result to host
  cudaMemcpy(h_c, d_c, ROW1 * COL2 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef PRINT
  std::cout << "Resultant matrix :" << std::endl;
  print_2DVec(h_c, COL2, ROW1);
#endif

  // Cleanup
  // free(h_a);
  // free(h_b);
  // free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
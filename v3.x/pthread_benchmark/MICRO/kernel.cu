#include "kernel.h"

__global__
void kernel_strided(int *array_a, int *array_b, int iterations, int jump,
    int stride)
{

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int effective_thread_id = block_id * blockDim.x + thread_id;
  int sum = 0;
  for(int i = 0; i < iterations; i++)
  {
    for(int j = 0; j < jump; j++)
    {
      int index = stride * (jump * effective_thread_id + j);
      sum += array_a[index];
    }
  }
  int index = stride * ( jump * effective_thread_id);
  array_a[index] = sum;

}

__global__
void kernel_reverse(int *array_a, int *array_b, int iterations, int jump,
    int stride)
{

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int effective_thread_id = block_id * blockDim.x + thread_id;
  int sum = 0;
  for(int i = 0; i < iterations; i++)
  {
    for(int j = jump - 1; j >= 0; j--)
    {
      int index = stride * ( jump * effective_thread_id + j);
      sum += array_a[index];
    }
  }
  int index = stride * ( jump * effective_thread_id);
  array_a[index] = sum;

}

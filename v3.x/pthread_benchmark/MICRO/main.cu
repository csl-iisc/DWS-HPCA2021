#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>
#include "../benchmark_common.h"

#include "kernel.h"

void read_parameters(char* filename, int* iterations, int* jump, int* stride,
    int* blocks, int* threads)
{
  std::vector<std::string> vecOfStrs;
  std::ifstream in(filename);
  std::string str;
  while (std::getline(in, str))
  {
    if(str.size() > 0)
      vecOfStrs.push_back(str);
  }
  assert(vecOfStrs.size() >= 5);
  *iterations  =  std::atoi(vecOfStrs[0].c_str());
  *jump        =  std::atoi(vecOfStrs[1].c_str());
  *stride      =  std::atoi(vecOfStrs[2].c_str());
  *blocks      =  std::atoi(vecOfStrs[3].c_str());
  *threads     =  std::atoi(vecOfStrs[4].c_str());
}

int main_micro(cudaStream_t stream_app, pthread_mutex_t* mutexapp, bool flag)
{

  int iterations, jump, stride, blocks, threads;
  read_parameters("MICRO/parameters", &iterations, &jump, &stride, &blocks,
      &threads);
  printf("%d\t%d\t%d\t%d\t%d\n", iterations, jump, stride, blocks, threads);

  int size = stride * jump * blocks * threads;
  int *host_array_a, *host_array_b;
  int *device_array_a, *device_array_b;
  host_array_a = (int*) malloc(sizeof(int) * size);
  host_array_b = (int*) malloc(sizeof(int) * size);
  cudaMalloc((void **)&device_array_a, size*sizeof(int));
  cudaMalloc((void **)&device_array_b, size*sizeof(int));

  cudaMemcpyAsync(device_array_a, host_array_a, size*sizeof(int),
      cudaMemcpyHostToDevice, stream_app);

  void (*kernel) (int*, int*, int, int, int) = &kernel_strided;
  /* void (*kernel) (int*, int*, int, int, int) = &kernel_reverse; */
  (kernel)<<<blocks, threads, 0, stream_app>>>(device_array_a, device_array_b,
      iterations, jump, stride);
  pthread_mutex_unlock(mutexapp);

  cutilSafeCall(cudaStreamSynchronize(stream_app));

  cudaMemcpyAsync(host_array_a, device_array_a, size*sizeof(int),
      cudaMemcpyDeviceToHost, stream_app);
  cudaFree(device_array_a);
  cudaFree(device_array_b);

}

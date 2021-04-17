#ifndef MICRO_KERNEL
#define MICRO_KERNEL

__global__
void kernel_strided(int *array_a, int *array_b, int iterations, int jump,
    int stride);

__global__
void kernel_reverse(int *array_a, int *array_b, int iterations, int jump,
    int stride);

#endif

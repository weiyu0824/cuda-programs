#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char** argv){
  // Data size
  int N = 256;

  // Init 3 host memories
  float *host_A = (float)malloc(N * sizeof(float));
  float *host_B = (float)malloc(N * sizeof(float));
  float *host_C = (float)malloc(N * sizeof(float));

  // Init 3 device memories
  float *device_A = (float)malloc(N * sizeof(float));
  float *device_B = (float)malloc(N * sizeof(float));
  float *device_C = (float)malloc(N * sizeof(float));
  
  // Init data
  
  // Copy data from host -> device

  // Launch kernel
  auto start = std::chrono::high_resolution_clock::now();
   
  auto stop = std::chrono::high_resolution_clock::now();
  
  // Copy data from device -> host_A

  // Free memory
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(decice_C);
  free(host_A);
  free(host_b);
  free(host_c);

  return 0;
}

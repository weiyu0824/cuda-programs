#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N){
     C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv){

  cudaDeviceProp deviceProp;
  printf("Device Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
  // Data size
  int N = 256;

  // Init 3 host memories
  float *host_A = (float*)malloc(N * sizeof(float));
  float *host_B = (float*)malloc(N * sizeof(float));
  float *host_C = (float*)malloc(N * sizeof(float));

  // Init 3 device memories
  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, N * sizeof(float));
  cudaMalloc((void**)&device_B, N * sizeof(float));
  cudaMalloc((void**)&device_C, N * sizeof(float));

  // Init host data
  for (int i = 0; i < N; i ++) {
    host_A[i] = float(i);
    host_B[i] = float(i);
  }
  // std::cout << host_B[5];
  
  // Copy data from host -> device
  cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice); 

  // Launch kernel
  auto start = std::chrono::high_resolution_clock::now();
  VecAdd<<<1, N>>>(device_A, device_B, device_C, N); 

  // Copy result from device -> host
  cudaMemcpy(host_C, device_C, N * sizeof(float), cudaMemcpyDeviceToHost); 
  auto stop = std::chrono::high_resolution_clock::now();  
  
  // Print result
  std::cout << "Result:";
  for (int i = 0; i < N; i ++) {
    std::cout << host_C[i] << " ";  
  }
  std::cout << std::endl;
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time spent:" << duration.count()  << "ms" << std::endl;

  // Free memory
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
  free(host_A);
  free(host_B);
  free(host_C);

  return 0;
}

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
      C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv){

  // Data size
  int N = 10240;

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
  
  // Copy data from host -> device
  cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice); 

  // 
  dim3 threadsPerBlock(32);
  dim3 numBlocks(N/threadsPerBlock.x);
  // Launch kernel
  auto start = std::chrono::high_resolution_clock::now();
  VecAdd<<<numBlocks, threadsPerBlock>>>(device_A, device_B, device_C, N); 

  // Copy result from device -> host
  cudaMemcpy(host_C, device_C, N * sizeof(float), cudaMemcpyDeviceToHost); 
  auto stop = std::chrono::high_resolution_clock::now();  
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      // Handle error
  }else{
    // Print result
    std::cout << "Result:";
    for (int i = 0; i < N; i ++) {
      std::cout << host_C[i] << " ";  
    }
    std::cout << std::endl;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time spent:" << duration.count()  << "ms" << std::endl;
  }
  // Free memory
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
  free(host_A);
  free(host_B);
  free(host_C);

  return 0;
}

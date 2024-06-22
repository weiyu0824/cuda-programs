#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include "../lib/vector_utils.hpp"

__global__ void VecAdd(float* A, float* B, float* C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
      C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv){
  

  // std::cout << argc << std::endl;
  if (argc != 4)
    throw std::runtime_error("Expecting 3 arguments, input_a, input_b, and output");
  

  // Init host memories & data size 
  size_t N;
  float *host_A = vector_utils::read_vector<float>(argv[1], N);
  float *host_B = vector_utils::read_vector<float>(argv[2], N);  
  float *host_ans = vector_utils::read_vector<float>(argv[3], N);
  float *host_C = (float*)malloc(N * sizeof(float));

  // vector_utils::print_vector<float>(host_A, N);
  // vector_utils::print_vector<float>(host_B, N);

  // Init 3 device memories
  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, N * sizeof(float));
  cudaMalloc((void**)&device_B, N * sizeof(float));
  cudaMalloc((void**)&device_C, N * sizeof(float));

  cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice); 

  // 
  dim3 threadsPerBlock(32);
  dim3 numBlocks((int) ceil(N/(float)threadsPerBlock.x));

  // Launch kernel
  auto start = std::chrono::high_resolution_clock::now();
  VecAdd<<<numBlocks, threadsPerBlock>>>(device_A, device_B, device_C, N); 

  // Copy result from device -> host
  cudaMemcpy(host_C, device_C, N * sizeof(float), cudaMemcpyDeviceToHost); 
  auto stop = std::chrono::high_resolution_clock::now();  
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
      // Handle error
  }else{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Log
    if (vector_utils::compare_vector<float>(host_ans, host_C, N))
      std::cout << "[o] ";
    else{
      vector_utils::print_vector(host_C, N);
      vector_utils::print_vector(host_ans, N);
      std::cout << "[x] ";
    }
    std::cout << "Data size:" << N << ", Time spent:" << duration.count()  << "ms" << std::endl;
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

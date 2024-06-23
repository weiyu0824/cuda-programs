#include <cstddef>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include "../lib/vector_utils.hpp"

__global__ void MatMul(float *A, float *B, float *C, size_t M, size_t N, size_t K){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < K) {
    float val = 0;
    for (int seq_idx = 0; seq_idx < N; seq_idx ++){
      val += A[i * N + seq_idx] * B[seq_idx * K + j];
    }
    C[i * K + j] = val;
  }
  // printf("(%d, %d)", i, j);
} 


int main(int argc, char **argv){

  if (argc != 4)
    throw std::runtime_error("Expecting 3 arguments, matrix_a, matrix_b, matric_output");
  // Data size
  size_t M, N, K;
  
  // Init host memory
  float *host_A = vector_utils::read_matrix<float>(argv[1], M, N); 
  float *host_B = vector_utils::read_matrix<float>(argv[2], N, K);
  float *host_ans = vector_utils::read_matrix<float>(argv[3], M, K);

  size_t Mat_A_bytes = M * N * sizeof(float);
  size_t Mat_B_bytes = N * K * sizeof(float);
  size_t Mat_C_bytes = M * K * sizeof(float);

  float *host_C = (float*)malloc(Mat_C_bytes);

  // Init device memory
  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, Mat_A_bytes);
  cudaMalloc((void**)&device_B, Mat_B_bytes);
  cudaMalloc((void**)&device_C, Mat_C_bytes);

  // Copy data to device
  cudaMemcpy(device_A, host_A, Mat_A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, Mat_B_bytes, cudaMemcpyHostToDevice);

  // Lanch kernel 
  dim3 blockDim(4, 4);
  int numBlockRows = (int)ceil(M / (float)blockDim.x);
  int numBlockCols = (int)ceil(K / (float)blockDim.y);

  dim3 gridDim(numBlockRows, numBlockCols);

  auto start = std::chrono::high_resolution_clock::now();
  MatMul<<<gridDim, blockDim>>>(device_A, device_B, device_C, M, N, K);
  auto stop = std::chrono::high_resolution_clock::now();
  
  // Copy back 
  cudaMemcpy(host_C, device_C, Mat_C_bytes, cudaMemcpyDeviceToHost);

  // Check error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Error: " << error << std::endl;
  } else {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 

    if (vector_utils::compare_vector<float>(host_ans, host_C, M * K)){
      std::cout << "[o] ";
    }else{
      std::cout << "[x] ";
    }

    std::cout << "Data size: (" << M << ", " << N << "), Time spent:" << duration.count()  << "ms" << std::endl;
  }

  // Free memory 
  free(host_A);
  free(host_B);
  free(host_C);
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);

  return 0;
}

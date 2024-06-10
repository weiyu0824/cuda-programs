#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>

__global__ void MatMul(float *A, float *B, float *C, int M, int N, int K){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < K) {
    int val = 0;
    for (int seq_idx = 0; seq_idx < N; seq_idx ++){
      val += A[i * N + seq_idx] * B[j + seq_idx * N];
    }
    C[i * K + j] = val;
  }

} 

void print_matrix(float* matrix, int row, int col){
    for (int i = 0; i < row; i ++) {
      for (int j = 0; j < col; j ++) {
        int idx = i * col + j;
        std::cout << matrix[idx] << " "; 
      }
      std::cout << std::endl;
    } 
}

int main(){
  // Data size
  int M = 10;
  int N = 4;
  int K = 4; 
  
  size_t Mat_A_bytes = M * N * sizeof(float);
  size_t Mat_B_bytes = N * K * sizeof(float);
  size_t Mat_C_bytes = M * K * sizeof(float);

  // Init host memory
  float *host_A = (float*)malloc(Mat_A_bytes);
  float *host_B = (float*)malloc(Mat_B_bytes);
  float *host_C = (float*)malloc(Mat_C_bytes);
  
  // Init device memory
  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, Mat_A_bytes);
  cudaMalloc((void**)&device_B, Mat_B_bytes);
  cudaMalloc((void**)&device_C, Mat_C_bytes);

  // Init Data
  for (int i = 0; i < M; i ++) {
    for (int j = 0; j < N; j ++) {
       int idx = i * N + j; 
       host_A[idx] = i;
    }
  } 
  for (int i = 0; i < N; i ++) {
    for (int j = 0; j < K; j ++) {
      int idx = i * K + j;
      host_B[idx] = i;
    }
  } 

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
    print_matrix(host_A, M, N);
    std::cout << std::endl;
    print_matrix(host_B, N, K);
    std::cout << std::endl;
    print_matrix(host_C, M, K);
    std::cout << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time spent:" << duration.count()  << "ms" << std::endl;
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

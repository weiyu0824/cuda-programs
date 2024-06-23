#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include "../lib/vector_utils.hpp"

#define TILE_SIZE 5 

__global__ void MatMul(float *A, float *B, float *C, int M, int N, int K){
  // A = M x N, B = N x K  
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;
  int threadRow = threadIdx.x; 
  int threadCol = threadIdx.y;
 
  // refactor
  int ARow = blockRow * TILE_SIZE + threadRow; 
  int BCol = blockCol * TILE_SIZE + threadCol;
  int CRow = ARow;
  int CCol = BCol;

  __shared__ float ASub[TILE_SIZE][TILE_SIZE];
  __shared__ float BSub[TILE_SIZE][TILE_SIZE];

  float val = 0; 
  for (int tileIdx = 0; tileIdx < (int)ceil(N/(float)TILE_SIZE); tileIdx ++){
    // Cooperative load  
    int ACol = tileIdx * TILE_SIZE + threadCol; 
    int BRow = tileIdx * TILE_SIZE + threadRow;
    if (ARow < M && ACol < N){
      ASub[threadRow][threadCol] = A[ARow * N + ACol];
    }else{
      ASub[threadRow][threadCol] = 0;
    }    
    if (BRow < N && BCol < K){
      BSub[threadRow][threadCol] = B[BRow * K + BCol];
    }else {
      BSub[threadRow][threadCol] = 0;
    }
    __syncthreads();
    
    // Inner product  
    for (int e = 0; e < TILE_SIZE; e ++){
      val += ASub[threadRow][e] * BSub[e][threadCol];
    }

    __syncthreads();
  }
   
  if (CRow < M && CCol < K){
    C[CRow * K + CCol] = val; 
  }
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
  
  float *host_C = (float*) malloc(Mat_C_bytes);
  


  // Init device memory
  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, Mat_A_bytes);
  cudaMalloc((void**)&device_B, Mat_B_bytes);
  cudaMalloc((void**)&device_C, Mat_C_bytes);


  // Copy data to device
  cudaMemcpy(device_A, host_A, Mat_A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, Mat_B_bytes, cudaMemcpyHostToDevice);

  // Lanch kernel 
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
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
    
    // printf("%f, %f", host_C[0], host_C[1]);

  // vector_utils::print_matrix<float>(host_ans, M, K);
    if (vector_utils::compare_vector<float>(host_C, host_ans, M * K))
      std::cout << "[o] ";
    else
      std::cout << "[x] ";
    
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
   

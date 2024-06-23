#include <fstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include "../lib/vector_utils.hpp"
#include <cmath>

#define EPS 1e-3

namespace vector_utils {

  template<typename T>
  T* read_vector(const std::string& filename, size_t& length){
    // std::cout << filename << std::endl;
    std::ifstream file(filename);

    if (!file.is_open()){
      throw std::runtime_error("Error opening the file " + filename);
    }
    if (!(file >> length)) {
      throw std::runtime_error("Error happened when reading length");
    }

    T* arr = new T[length];

    for (size_t i = 0; i < length; ++i){
      if (!(file >> arr[i])) {
        delete [] arr;
        throw std::runtime_error("Error happened when reading vector");
      }
    }
    return arr;
  } 

  template float* read_vector<float>(const std::string&, size_t&);


  template<typename T>
  void print_vector(const T *arr, size_t& length){
    for (int i = 0; i < length; ++ i){
      std::cout << arr[i] << ", "; 
    }
    std::cout << std::endl;
  }
  
  template void print_vector<float>(const float*, size_t&);

  
  template<typename T>
  bool compare_vector(const T *arr_a, const T *arr_b, const size_t& length){
    for (int i = 0; i < length; ++i){
      if (fabs(arr_a[i] - arr_b[i]) > EPS){
        return false;
      }
    }
    return true;
  }
  template bool compare_vector<float>(const float*, const float*, const size_t&);

  template<typename T>
  T* read_matrix(const std::string& filename, size_t& rows, size_t& cols){
     std::ifstream file(filename);

    if (!file.is_open()){
      throw std::runtime_error("Error opening the file " + filename);
    }
    if (!(file >> rows)) {
      throw std::runtime_error("Error happened when reading rows");
    }
    if (!(file >> cols)) {
      throw std::runtime_error("Error happened when reading cols"); 
    }

    T* arr = new T[rows * cols];

    for (size_t i = 0; i < rows * cols; ++i){
      if (!(file >> arr[i])) {
        delete [] arr;
        throw std::runtime_error("Error happened when reading vector");
      }
    }


    // std::cout << arr[0] << " " << arr[rows * cols - 1] << std::endl;
    return arr;
  }  
  template float* read_matrix<float>(const std::string&, size_t&, size_t&); 

}

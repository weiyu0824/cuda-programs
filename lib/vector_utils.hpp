#ifndef VECTOR_UTILS_H 
#define VECTOR_UTILS_H


#include <string>

namespace vector_utils {
  template<typename T>
  T* read_vector(const std::string& filename, size_t& length);
  
  template<typename T>
  void print_vector(const T* arr, size_t& length);

  template<typename T>
  bool compare_vector(const T* arr_a, const T* arr_b, size_t& length);

  // void read_matrix();
  // void print_vector();
  // bool compare_vector();
}
#endif // VECTOR_UTILS_H 

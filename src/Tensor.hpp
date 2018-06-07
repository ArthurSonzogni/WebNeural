#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>
#include "util.hpp"

class Node;

struct Tensor {
  std::vector<float> values;
  std::vector<size_t> sizes;
  Node* producer = nullptr;

  Tensor();
  Tensor(size_t size);
  Tensor(const std::vector<size_t>& sizes);

  std::string ToString();
  void Fill(float value);
  static Tensor Random(const std::vector<size_t>& sizes);
  void Randomize();
  float Error();

  // Operators.
  float& operator[](size_t i) { return values[i]; }
  Tensor operator-(const Tensor& tensor) const;
};

#endif /* end of include guard: TENSOR_H */

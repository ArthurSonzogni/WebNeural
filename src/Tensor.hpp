#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>
#include "util.hpp"

using std::size_t;

class Node;

struct Tensor {
  std::vector<float> values;
  std::vector<size_t> sizes;

  Tensor();
  Tensor(size_t size);
  Tensor(const std::vector<size_t>& sizes);

  static Tensor Random(const std::vector<size_t>& sizes);
  static Tensor SphericalRandom(const std::vector<size_t>& sizes);

  std::string ToString();
  void Fill(float value);
  void Randomize();
  void UniformRandom();
  float Error();
  size_t ArgMax();
  float& at(size_t x, size_t y);
  float& at(size_t x, size_t y, size_t z);

  // Operators.
  void operator*=(float lambda);
  void operator+=(Tensor other);
  float& operator[](size_t i) { return values[i]; }
  Tensor operator-(const Tensor& tensor) const;
  bool operator==(const Tensor& other) const;
  bool operator!=(const Tensor& other) const;

  static Tensor Merge(std::vector<Tensor> tensors);
};

#endif /* end of include guard: TENSOR_H */

#include "Tensor.hpp"
#include <algorithm>
#include <sstream>
#include <random>

Tensor::Tensor() : Tensor({}) {}
Tensor::Tensor(size_t size) : Tensor(std::vector<size_t>{size}) {}
Tensor::Tensor(const std::vector<size_t>& sizes)
    : values(Multiply(sizes)), sizes(sizes) {}

void Tensor::Fill(float value) {
  std::fill(values.begin(), values.end(), value);
}

Tensor Tensor::operator-(const Tensor& tensor) const {
  Tensor output = *this;
  for (size_t i = 0; i < values.size(); ++i) {
    output[i] -= tensor.values.at(i);
  }
  return output;
}

float Tensor::Error() {
  float ret = 0.f;
  for (const auto i : values) {
    ret += i * i;
  }
  return ret;
}

std::string Tensor::ToString() {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      ss << ", ";
    ss << values[i];
  }
  ss << "]";
  return ss.str();
}

// static
Tensor Tensor::Random(const std::vector<size_t>& sizes) {
  Tensor tensor(sizes);
  tensor.Randomize();
  return tensor;
}

void Tensor::Randomize() {
  static std::mt19937 rng;
  std::normal_distribution<float> random(0.0, 1.0);
  for(auto& i : values)
    i = random(rng);
}

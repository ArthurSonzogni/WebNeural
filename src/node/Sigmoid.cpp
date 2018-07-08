#include <cmath>
#include "Sigmoid.hpp"

Sigmoid::Sigmoid(Node& node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  InitInternalSensitivity();
}

void Sigmoid::Forward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = 1.0 / (1.0 + exp(-I[i]));
    }
  }
}

void Sigmoid::Backward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    for (size_t index = 0; index < size; ++index) {
      IS[index] = O[index] * (1.f - O[index]) * OS[index];
    }
  }
}

#include <cmath>
#include "Tanh.hpp"

Tanh::Tanh(Node* node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  InitInternalSensitivity();
}

void Tanh::Forward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = std::tanh(I[i]);
    }
  }
}

void Tanh::Backward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    for (size_t index = 0; index < size; ++index) {
      IS[index] = (1.f - O[index] * O[index]) * OS[index];
    }
  }
}

#include "node/LeakyRelu.hpp"

LeakyRelu::LeakyRelu(Node* node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void LeakyRelu::Forward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = I[i] > 0.f ? I[i] : 0.2f * I[i];
    }
  }
}

void LeakyRelu::Backward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& IS = input_sensitivity[batch];

    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      IS[i] = I[i] > 0.f ? OS[i] : 0.2f * OS[i];
    }
  }
}

#include "node/LeakyRelu.hpp"

LeakyRelu::LeakyRelu(Node* node) {
  AddInput(node);
  InitBuffers(inputs[0]->output[0].sizes);
}

void LeakyRelu::Forward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];
    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = I[i] > 0.f ? I[i] : 0.1f * I[i];
    }
  }
}

void LeakyRelu::Backward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& I = inputs[0]->output[batch];
    const Tensor& OS = grad[batch];
    Tensor& IS = inputs[0]->grad[batch];
    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      IS[i] += I[i] > 0.f ? OS[i] : 0.1f * OS[i];
    }
  }
}

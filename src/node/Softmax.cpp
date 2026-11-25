#include <cmath>
#include "Softmax.hpp"
#include "util/stable_softmax.hpp"

Softmax::Softmax(Node* node) {
  AddInput(node);
  InitBuffers(inputs[0]->output[0].sizes);
}

void Softmax::Forward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];
    StableSoftmax(I.values, O.values);
  }
}

void Softmax::Backward(size_t batch_size) {
  const size_t size = output[0].values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& O = output[batch];
    Tensor& IS = inputs[0]->grad[batch];
    const Tensor& OS = grad[batch];

    float accu = 0.f;
    for (size_t j = 0; j < size; ++j)
      accu += O[j] * OS[j];
    for (size_t i = 0; i < size; ++i)
      IS[i] += O[i] * (OS[i] - accu);
  }
}

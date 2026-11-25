#include "node/Bias.hpp"

Bias::Bias(Node* node) {
  AddInput(node);
  const auto& in_sizes = inputs[0]->output[0].sizes;
  params = Tensor(in_sizes);
  InitBuffers(in_sizes);
}

void Bias::Forward(size_t batch_size) {
  const size_t size = output[0].values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];
    for (size_t index = 0; index < size; ++index)
      O[index] = I[index] + params[index];
  }
}

void Bias::Backward(size_t batch_size) {
  const size_t size = output[0].values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& IS = inputs[0]->grad[batch];
    const Tensor& OS = grad[batch];
    for (size_t index = 0; index < size; ++index) {
      IS[index] += OS[index];
    }
  }

  float* __restrict__ pg = params_grad.values.data();
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ os = grad[batch].values.data();
    #pragma omp simd
    for (size_t index = 0; index < size; ++index) {
      pg[index] += os[index];
    }
  }
}

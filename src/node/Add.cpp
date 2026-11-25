#include "node/Add.hpp"
#include <stdexcept>

Add::Add(Node* a, Node* b) {
  AddInput(a);
  AddInput(b);

  if (a->output[0].values.size() != b->output[0].values.size()) {
    throw std::invalid_argument("Add: input tensor sizes do not match");
  }

  InitBuffers(a->output[0].sizes);
}

void Add::Forward(size_t batch_size) {
  const size_t size = output[0].values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ a = inputs[0]->output[batch].values.data();
    const float* __restrict__ b = inputs[1]->output[batch].values.data();
    float* __restrict__ o = output[batch].values.data();
    #pragma omp simd
    for (size_t i = 0; i < size; ++i) {
      o[i] = a[i] + b[i];
    }
  }
}

void Add::Backward(size_t batch_size) {
  const size_t size = output[0].values.size();
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ g = grad[batch].values.data();
    float* __restrict__ ga = inputs[0]->grad[batch].values.data();
    float* __restrict__ gb = inputs[1]->grad[batch].values.data();
    #pragma omp simd
    for (size_t i = 0; i < size; ++i) {
      ga[i] += g[i];
      gb[i] += g[i];
    }
  }
}

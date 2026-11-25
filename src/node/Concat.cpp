#include "node/Concat.hpp"
#include <algorithm>
#include <stdexcept>

Concat::Concat(Node* a, Node* b) {
  AddInput(a);
  AddInput(b);

  const auto& sa = a->output[0].sizes;
  const auto& sb = b->output[0].sizes;
  size_a_ = a->output[0].values.size();
  size_b_ = b->output[0].values.size();

  std::vector<size_t> out_sizes;
  if (sa.size() >= 3 && sb.size() >= 3) {
    if (sa[0] != sb[0] || sa[1] != sb[1]) {
      throw std::invalid_argument("Concat: spatial dimensions must match");
    }
    out_sizes = {sa[0], sa[1], sa[2] + sb[2]};
  } else {
    out_sizes = {size_a_ + size_b_};
  }

  InitBuffers(out_sizes);
}

void Concat::Forward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ a = inputs[0]->output[batch].values.data();
    const float* __restrict__ b = inputs[1]->output[batch].values.data();
    float* __restrict__ o = output[batch].values.data();
    std::copy_n(a, size_a_, o);
    std::copy_n(b, size_b_, o + size_a_);
  }
}

void Concat::Backward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ g = grad[batch].values.data();
    float* __restrict__ ga = inputs[0]->grad[batch].values.data();
    float* __restrict__ gb = inputs[1]->grad[batch].values.data();
    #pragma omp simd
    for (size_t i = 0; i < size_a_; ++i) {
      ga[i] += g[i];
    }
    #pragma omp simd
    for (size_t i = 0; i < size_b_; ++i) {
      gb[i] += g[size_a_ + i];
    }
  }
}

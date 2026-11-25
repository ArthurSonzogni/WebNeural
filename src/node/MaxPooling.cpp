#include "node/MaxPooling.hpp"
#include <algorithm>
#include <iostream>

MaxPooling::MaxPooling(Node* node) {
  AddInput(node);

  const auto& in_sizes = inputs[0]->output[0].sizes;
  const size_t dim_z = in_sizes.size() >= 3 ? in_sizes[2] : 1;
  InitBuffers({
      in_sizes[0] / 2,
      in_sizes[1] / 2,
      dim_z,
  });
}

void MaxPooling::Forward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& I = inputs[0]->output[batch];
    Tensor& O = output[batch];

    float* i = const_cast<float*>(&I[0]);
    float* o = &O[0];
    for (size_t z = 0; z < O.sizes[2]; ++z) {
      ForwardZ(i, o);
      i += I.sizes[0] * I.sizes[1];
      o += O.sizes[0] * O.sizes[1];
    }
  }
}

void MaxPooling::ForwardZ(float* i, float* o) {
  const size_t dim_ix = inputs[0]->output[0].sizes[0];
  const size_t dim_ox = output[0].sizes[0];
  const size_t dim_oy = output[0].sizes[1];
  for (size_t y = 0; y < dim_oy; ++y) {
    for (size_t x = 0; x < dim_ox; ++x) {
      float m = i[2 * x + 0 + dim_ix * (2 * y + 0)];
      m = std::max(m, i[2 * x + 1 + dim_ix * (2 * y + 0)]);
      m = std::max(m, i[2 * x + 0 + dim_ix * (2 * y + 1)]);
      m = std::max(m, i[2 * x + 1 + dim_ix * (2 * y + 1)]);
      o[x + dim_ox * y] = m;
    }
  }
}

void MaxPooling::Backward(size_t batch_size) {
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& I = inputs[0]->output[batch];
    Tensor& O = output[batch];
    Tensor& IS = inputs[0]->grad[batch];
    const Tensor& OS = grad[batch];

    float* i = const_cast<float*>(&I[0]);
    float* is = &IS[0];
    float* o = &O[0];
    float* os = const_cast<float*>(&OS[0]);
    for (size_t z = 0; z < O.sizes[2]; ++z) {
      BackwardZ(i, is, o, os);
      i += I.sizes[0] * I.sizes[1];
      is += I.sizes[0] * I.sizes[1];
      o += O.sizes[0] * O.sizes[1];
      os += O.sizes[0] * O.sizes[1];
    }
  }
}

void MaxPooling::BackwardZ(float* i, float* is, float* o, float* os) {
  const size_t dim_ix = inputs[0]->output[0].sizes[0];
  const size_t dim_ox = output[0].sizes[0];
  const size_t dim_oy = output[0].sizes[1];

  for (size_t y = 0; y < dim_oy; ++y) {
    for (size_t x = 0; x < dim_ox; ++x) {
      const size_t index_output = x + dim_ox * y;
      const size_t indices[4] = {
          2 * x + 0 + dim_ix * (2 * y + 0),
          2 * x + 1 + dim_ix * (2 * y + 0),
          2 * x + 0 + dim_ix * (2 * y + 1),
          2 * x + 1 + dim_ix * (2 * y + 1),
      };
      for (size_t idx : indices) {
        if (i[idx] == o[index_output]) {
          is[idx] += os[index_output];
          break;
        }
      }
    }
  }
}

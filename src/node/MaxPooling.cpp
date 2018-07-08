#include <iostream>
#include "node/MaxPooling.hpp"

MaxPooling::MaxPooling(Node& node) {
  Link(node);

  // clang-format off
  output = std::vector<Tensor>(T, Tensor({
    input[0]->sizes[0]/2,
    input[0]->sizes[1]/2,
    input[0]->sizes[2],
  }));
  // clang-format on

  InitInternalSensitivity();
}

void MaxPooling::Forward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& O = output[batch];

    float* i = &(I[0]);
    float* o = &(O[0]);
    for (size_t z = 0; z < O.sizes[2]; ++z) {
      ForwardZ(i, o);
      i += I.sizes[0] * I.sizes[1];
      o += O.sizes[0] * O.sizes[1];
    }
  }
}

void MaxPooling::ForwardZ(float* i, float* o) {
  const size_t dim_ix = input[0]->sizes[0];
  // const size_t dim_iy = input->sizes[1];
  const size_t dim_ox = output[0].sizes[0];
  const size_t dim_oy = output[0].sizes[1];
  // Work on one z-layer.
  for (size_t y = 0; y < dim_oy; ++y) {
    for (size_t x = 0; x < dim_ox; ++x) {
      // Do the max of 4 voxel.
      float m = i[2 * x + 0 + dim_ix * (2 * y + 0)];
      m = std::max(m, i[2 * x + 1 + dim_ix * (2 * y + 0)]);
      m = std::max(m, i[2 * x + 0 + dim_ix * (2 * y + 1)]);
      m = std::max(m, i[2 * x + 1 + dim_ix * (2 * y + 1)]);
      // Assign the final value;
      o[x + dim_ox * y] = m;
    }
  }
}

void MaxPooling::Backward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& O = output[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);

    float* i = &(I[0]);
    float* is = &(IS[0]);
    float* o = &(O[0]);
    float* os = &(OS[0]);
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
  const size_t dim_ix = input[0]->sizes[0];
  const size_t dim_ox = output[0].sizes[0];
  const size_t dim_oy = output[0].sizes[1];

  auto update = [&](size_t index_input, size_t index_output) {
    is[index_input] = i[index_input] == o[index_output] ? os[index_output]  //
                                                        : 0.f;              //
  };

  // Work on one z-layer.
  for (size_t y = 0; y < dim_oy; ++y) {
    for (size_t x = 0; x < dim_ox; ++x) {
      update(2 * x + 0 + dim_ix * (2 * y + 0), x + dim_ox * y);
      update(2 * x + 1 + dim_ix * (2 * y + 0), x + dim_ox * y);
      update(2 * x + 0 + dim_ix * (2 * y + 1), x + dim_ox * y);
      update(2 * x + 1 + dim_ix * (2 * y + 1), x + dim_ox * y);
    }
  }
}

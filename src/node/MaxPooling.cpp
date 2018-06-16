#include <iostream>
#include "node/MaxPooling.hpp"

MaxPooling::MaxPooling(Node& node) {
  Link(node);

  // clang-format off
  output = Tensor({
    input->sizes[0]/2,
    input->sizes[1]/2,
    input->sizes[2],
  });
  // clang-format on

  input_sensitivity = Tensor(input->sizes);
}

void MaxPooling::Forward() {
  float* i = &(*input)[0];
  float* o = &output[0];
  for (size_t z = 0; z < output.sizes[2]; ++z) {
    ForwardZ(i, o);
    i += input->sizes[0] * input->sizes[1];
    o += output.sizes[0] * output.sizes[1];
  }
}

void MaxPooling::ForwardZ(float* i, float* o) {
  const size_t dim_ix = input->sizes[0];
  // const size_t dim_iy = input->sizes[1];
  const size_t dim_ox = output.sizes[0];
  const size_t dim_oy = output.sizes[1];
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

void MaxPooling::Backward() {
  float* i = &(*input)[0];
  float* is = &input_sensitivity[0];
  float* o = &output[0];
  float* os = &(*output_sensitivity)[0];
  for (size_t z = 0; z < output.sizes[2]; ++z) {
    BackwardZ(i, is, o, os);
    i += input->sizes[0] * input->sizes[1];
    is += input->sizes[0] * input->sizes[1];
    o += output.sizes[0] * output.sizes[1];
    os += output.sizes[0] * output.sizes[1];
  }
}

void MaxPooling::BackwardZ(float* i, float* is, float* o, float* os) {
  const size_t dim_ix = input->sizes[0];
  // const size_t dim_iy = input->sizes[1];
  const size_t dim_ox = output.sizes[0];
  const size_t dim_oy = output.sizes[1];

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

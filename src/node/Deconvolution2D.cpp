#include "node/Deconvolution2D.hpp"
#include <cmath>
#include <iostream>

Deconvolution2D::Deconvolution2D(Node& node,
                             const std::vector<size_t> sizes,
                             size_t num_features,
                             size_t stride)
    : stride(stride) {
  Link(node);

  // clang-format off
  size_input = {
    input[0]->sizes[0],
    input[0]->sizes[1],
    input[0]->values.size() / (input[0]->sizes[0] * input[0]->sizes[1]),
  };

  size_params = {
    sizes[0],
    sizes[1],
    size_input[2],
    num_features,
  };

  size_output = {
    (size_input[0] - 1) * stride + size_params[0],
    (size_input[1] - 1) * stride + size_params[1],
    num_features,
  };
  // clang-format on

  output = std::vector<Tensor>(T, Tensor(size_output));
  params = Tensor::Random(size_params);
  params *= 1.0f / sqrt(sizes[0] * sizes[1] * size_input[2]);

  InitInternalSensitivity();
}

void Deconvolution2D::Forward(size_t batch_size) {
  // clang-format off
  #pragma omp parallel for
  for(size_t batch = 0; batch<batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& O = output[batch];
    O.Fill(0.f);
    for(size_t z = 0; z<size_input[2]; ++z)
    for(size_t y = 0; y<size_input[1]; ++y)
    for(size_t x = 0; x<size_input[0]; ++x) {
      const float input_value = I.at(x,y,z);
      for(size_t dz = 0; dz < size_params[3]; ++dz)
      for(size_t dy = 0; dy < size_params[1]; ++dy)
      for(size_t dx = 0; dx < size_params[0]; ++dx) {
        const float output_index =
           stride * x + dx + size_output[0] * (
           stride * y + dy + size_output[1] * (
           dz));
        const size_t params_index = 
          dx + size_params[0] * (
          dy + size_params[1] * (
          z  + size_params[2] * (
          dz)));
        O[output_index] += params[params_index] * input_value;
      }
    }
  }
  // clang-format on
}

void Deconvolution2D::Backward(size_t batch_size) {
  // clang-format off
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& IS = input_sensitivity[batch];
    Tensor& PS = params_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    IS.Fill(0.f);
    for(size_t z = 0; z<size_input[2]; ++z)
    for(size_t y = 0; y<size_input[1]; ++y)
    for(size_t x = 0; x<size_input[0]; ++x) {
      const float input_value = I.at(x,y,z);
      for(size_t dz = 0; dz < size_params[3]; ++dz)
      for(size_t dy = 0; dy < size_params[1]; ++dy)
      for(size_t dx = 0; dx < size_params[0]; ++dx) {
        const float output_index =
           stride * x + dx + size_output[0] * (
           stride * y + dy + size_output[1] * (
           dz));
        const size_t params_index = 
          dx + size_params[0] * (
          dy + size_params[1] * (
           z + size_params[2] * (
          dz)));
        const float os = OS[output_index];
        IS.at(x,y,z) += params[params_index] * os;
        PS[params_index] += input_value * os;
      }
    }
  }
  // clang-format on
}

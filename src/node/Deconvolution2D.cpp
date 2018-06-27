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
    input->sizes[0],
    input->sizes[1],
    input->values.size() / (input->sizes[0] * input->sizes[1]),
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

  output = Tensor(size_output);

  params = Tensor::Random(size_params);
  params *= 1.0f / sqrt(sizes[0] * sizes[1] * size_input[2]);

  input_sensitivity = Tensor(input->sizes);
  params_sensitivity = Tensor(params.sizes);
}

void Deconvolution2D::Forward() {
  output.Fill(0.f);
  // clang-format off
  //#pragma omp parallel for
  for(size_t z = 0; z<size_input[2]; ++z)
  for(size_t y = 0; y<size_input[1]; ++y)
  for(size_t x = 0; x<size_input[0]; ++x) {
    const size_t input_index = 
      x + size_input[0] * (
      y + size_input[1] * (
      z));
    const float input_value = input->values[input_index];
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
      output[output_index] += params[params_index] * input_value;
    }
  }
  // clang-format on
}

void Deconvolution2D::Backward() {
  input_sensitivity.Fill(0.f);
  // clang-format off
  //#pragma omp parallel for
  for(size_t z = 0; z<size_input[2]; ++z)
  for(size_t y = 0; y<size_input[1]; ++y)
  for(size_t x = 0; x<size_input[0]; ++x) {
    const size_t input_index = 
      x + size_input[0] * (
      y + size_input[1] * (
      z));
    const float input_value = input->values[input_index];
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
      const float os = output_sensitivity->values[output_index];
      input_sensitivity[input_index] += params[params_index] * os;
      params_sensitivity[params_index] += input_value * os;
    }
  }
  // clang-format on
}

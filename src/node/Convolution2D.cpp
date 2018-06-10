#include "node/Convolution2D.hpp"

Convolution2D::Convolution2D(Node& node,
                             const std::vector<size_t> sizes,
                             size_t num_features) {
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
    size_input[0] - size_params[0] + 1,
    size_input[1] - size_params[1] + 1,
    num_features,
  };
  // clang-format on

  output = Tensor(size_output);
  output.producer = this;

  params = Tensor::Random(size_params);
  params *= 0.01f / (sizes[0] * sizes[1] * size_input[2]);

  input_sensitivity = Tensor(input->sizes);
  params_sensitivity = Tensor(params.sizes);
}

void Convolution2D::Forward() {
  // clang-format off
  //#pragma omp parallel for
  for(size_t f = 0; f<size_output[2]; ++f)
  for(size_t y = 0; y<size_output[1]; ++y)
  for(size_t x = 0; x<size_output[0]; ++x) {
    size_t params_index = f * size_params[0] * size_params[1] * size_params[2];
    float* p = &params[params_index];
    float v = 0.f;
    for(size_t dz = 0; dz < size_params[2]; ++dz)
    for(size_t dy = 0; dy < size_params[1]; ++dy)
    for(size_t dx = 0; dx < size_params[0]; ++dx) {
      const size_t input_index =
         x + dx + size_input[0] * (
         y + dy + size_input[1] * (
         0 + dz));
      v += (*p) * (*input)[input_index];
      ++p;
    }
    output[x + size_output[0] * ( y + size_output[1] * f)] = v;
  }
  // clang-format on
}

void Convolution2D::Backward() {
  input_sensitivity.Fill(0.f);
  // clang-format off
  //#pragma omp parallel for
  for(size_t f = 0; f<size_output[2]; ++f)
  for(size_t y = 0; y<size_output[1]; ++y)
  for(size_t x = 0; x<size_output[0]; ++x) {
    size_t output_index= x + size_output[0] * ( y  + size_output[1] * ( f ) );
    const float os =  (*output_sensitivity)[output_index];
    size_t params_index = f * size_params[0] * size_params[1] * size_params[2];
    float* p  = &params[params_index];
    float* ps = &params_sensitivity[params_index];
    for(size_t dz = 0; dz < size_params[2]; ++dz)
    for(size_t dy = 0; dy < size_params[1]; ++dy)
    for(size_t dx = 0; dx < size_params[0]; ++dx) {
      const size_t input_index =
         x + dx + size_input[0] * (
         y + dy + size_input[1] * (
         0 + dz));
      input_sensitivity[input_index] += (*p) * os;
      *ps += (*input)[input_index] * os;
      ++p;
      ++ps;
    }
  }
  // clang-format on
}

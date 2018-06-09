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

  size_half_params = {
    (sizes[0]-1)/2,
    (sizes[1]-1)/2,
  };

  size_output = {
    size_input[0] - size_params[0] + 1,
    size_input[1] - size_params[1] + 1,
    num_features,
  };
  // clang-format on

  output = Tensor(size_output);
  params = Tensor::Random(size_params);
  params *= 1.f / (sizes[0] * sizes[1]);

  input_sensitivity = Tensor(input->sizes);
  params_sensitivity = Tensor(params.sizes);
}

void Convolution2D::Forward() {
  // clang-format off
  float* o = &(output[0]);
  for(size_t c = 0; c<size_output[2]; ++c)
  for(size_t y = 0; y<size_output[1]; ++y)
  for(size_t x = 0; x<size_output[0]; ++x) {
    float v = 0.f;
    float* p = &params[0];
    for(size_t z = 0; z<size_input[2]; ++z)
    for(size_t dy = -size_half_params[1]; dy <= +size_half_params[1]; ++dy)
    for(size_t dx = -size_half_params[0]; dx <= +size_half_params[0]; ++dx) {
      const size_t input_index = x + dx + size_input[0] * (y + dy + size_input[1] * z);
      v += (*p) * (*input)[input_index];
      ++p;
    }
    *o = v;
    ++o;
  }
  // clang-format on
}

void Convolution2D::Backward() {
  input_sensitivity.Fill(0.f);
  // clang-format off
  float* o = &((*output_sensitivity)[0]);
  for(size_t c = 0; c<size_output[2]; ++c)
  for(size_t y = 0; y<size_output[1]; ++y)
  for(size_t x = 0; x<size_output[0]; ++x) {
    const float v = *(o++);
    float* p = &params.values[0];
    float* ps = &params_sensitivity.values[0];
    for(size_t z = 0; z<size_input[2]; ++z)
    for(size_t dy = -size_half_params[1]; dy <= +size_half_params[1]; ++dy)
    for(size_t dx = -size_half_params[0]; dx <= +size_half_params[0]; ++dx) {
      const size_t input_index = x + dx + size_input[0] * (y + dy + size_input[1] * z);
      input_sensitivity[input_index] += (*p) * v;
      *ps += (*input)[input_index] * v;
      ++p;
      ++ps;
    }
  }
  // clang-format on
}

#include "node/Convolution2D.hpp"
#include <cmath>

Convolution2D::Convolution2D(Node& node,
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
    (size_input[0] - size_params[0]) / stride + 1,
    (size_input[1] - size_params[1]) / stride + 1,
    num_features,
  };
  // clang-format on

  output = std::vector<Tensor>(T, Tensor(size_output));
  params = Tensor::Random(size_params);
  params *= 1.0f / sqrt(sizes[0] * sizes[1] * size_input[2]);

  InitInternalSensitivity();
}

void Convolution2D::Forward(size_t batch_size) {
  // clang-format off
  #pragma omp parallel for
  for(size_t batch = 0; batch<batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& O = output[batch];
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
           stride * x + dx + size_input[0] * (
           stride * y + dy + size_input[1] * (
           0 + dz));
        v += (*p) * I[input_index];
        ++p;
      }
      O.at(x,y,f) = v;
    }
  }
  // clang-format on
}

void Convolution2D::Backward(size_t batch_size) {
  // clang-format off
  #pragma omp parallel for
  for(size_t batch = 0; batch<batch_size; ++batch) {
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& IS = input_sensitivity[batch];
    Tensor& I = *(input[batch]);
    Tensor& P = params;
    Tensor& PS = params_sensitivity[batch];

    IS.Fill(0.f);
    for(size_t f = 0; f<size_output[2]; ++f)
    for(size_t y = 0; y<size_output[1]; ++y)
    for(size_t x = 0; x<size_output[0]; ++x) {
      const float os =  OS.at(x,y,f);
      size_t params_index = f * size_params[0] * size_params[1] * size_params[2];
      float* p  = &P[params_index];
      float* ps = &PS[params_index];
      for(size_t dz = 0; dz < size_params[2]; ++dz)
      for(size_t dy = 0; dy < size_params[1]; ++dy)
      for(size_t dx = 0; dx < size_params[0]; ++dx) {
        const size_t input_index =
           stride * x + dx + size_input[0] * (
           stride * y + dy + size_input[1] * (
           0 + dz));
        IS[input_index] += (*p) * os;
        *ps += I[input_index] * os;
        ++p;
        ++ps;
      }
    }
  }
  // clang-format on
}

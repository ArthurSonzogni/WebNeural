#include "node/Convolution2D.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

Convolution2D::Convolution2D(Node* node,
                             const std::vector<size_t> sizes,
                             size_t num_features,
                             size_t stride)
    : stride(stride) {
  AddInput(node);

  const Tensor& in0 = inputs[0]->output[0];
  size_input = {
      in0.sizes[0],
      in0.sizes[1],
      in0.values.size() / (in0.sizes[0] * in0.sizes[1]),
  };

  size_params = {
      sizes[0],
      sizes[1],
      size_input[2],
      num_features,
  };

  if (stride == 0) {
    throw std::invalid_argument("Convolution2D stride must be > 0");
  }
  if (size_input[0] < size_params[0] || size_input[1] < size_params[1]) {
    throw std::invalid_argument("Convolution2D filter larger than input");
  }

  if ((size_input[0] - size_params[0]) % stride != 0) {
    std::cerr << "Error line = " << __LINE__ << " " << size_input[0] << " "
              << size_params[0] << " " << stride << std::endl;
  }
  if ((size_input[1] - size_params[1]) % stride != 0) {
    std::cerr << "Error line = " << __LINE__ << " " << size_input[1] << " "
              << size_params[1] << " " << stride << std::endl;
  }

  size_output = {
      (size_input[0] - size_params[0]) / stride + 1,
      (size_input[1] - size_params[1]) / stride + 1,
      num_features,
  };

  params = Tensor::Random(size_params);
  params *= 1.0f / sqrt(sizes[0] * sizes[1] * size_input[2]);

  InitBuffers(size_output);
}

void Convolution2D::Forward(size_t batch_size) {
  const size_t in_w = size_input[0];
  const size_t in_h = size_input[1];
  const size_t kw = size_params[0];
  const size_t kh = size_params[1];
  const size_t kc = size_params[2];
  const size_t out_w = size_output[0];
  const size_t out_h = size_output[1];
  const size_t out_f = size_output[2];
  const size_t filter_stride = kw * kh * kc;
  const float* __restrict__ P = params.values.data();

  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ I = inputs[0]->output[batch].values.data();
    float* __restrict__ O = output[batch].values.data();
    std::fill_n(O, out_w * out_h * out_f, 0.f);

    for (size_t f = 0; f < out_f; ++f) {
      const float* __restrict__ pf = P + f * filter_stride;
      float* __restrict__ of = O + f * out_w * out_h;
      for (size_t y = 0; y < out_h; ++y) {
        float* __restrict__ o_row = of + y * out_w;
        const float* __restrict__ p = pf;
        for (size_t dz = 0; dz < kc; ++dz) {
          const float* __restrict__ in_z = I + dz * in_w * in_h;
          for (size_t dy = 0; dy < kh; ++dy) {
            const float* __restrict__ in_y = in_z + (stride * y + dy) * in_w;
            for (size_t dx = 0; dx < kw; ++dx) {
              const float w = *p++;
              const float* __restrict__ in_row = in_y + dx;
              if (stride == 1) {
                #pragma omp simd
                for (size_t x = 0; x < out_w; ++x) {
                  o_row[x] += w * in_row[x];
                }
              } else {
                for (size_t x = 0; x < out_w; ++x) {
                  o_row[x] += w * in_row[stride * x];
                }
              }
            }
          }
        }
      }
    }
  }
}

void Convolution2D::Backward(size_t batch_size) {
  const size_t in_w = size_input[0];
  const size_t in_h = size_input[1];
  const size_t kw = size_params[0];
  const size_t kh = size_params[1];
  const size_t kc = size_params[2];
  const size_t out_w = size_output[0];
  const size_t out_h = size_output[1];
  const size_t out_f = size_output[2];
  const size_t filter_stride = kw * kh * kc;
  const size_t params_size = params.values.size();
  const float* __restrict__ P = params.values.data();
  float* __restrict__ PG = params_grad.values.data();

  #pragma omp parallel if(batch_size > 1)
  {
    std::vector<float> local_ps(params_size, 0.f);
    float* __restrict__ PS = local_ps.data();

    #pragma omp for
    for (size_t batch = 0; batch < batch_size; ++batch) {
      const float* __restrict__ OS = grad[batch].values.data();
      float* __restrict__ IS = inputs[0]->grad[batch].values.data();
      const float* __restrict__ I = inputs[0]->output[batch].values.data();

      for (size_t f = 0; f < out_f; ++f) {
        const float* __restrict__ pf = P + f * filter_stride;
        float* __restrict__ psf = PS + f * filter_stride;
        const float* __restrict__ osf = OS + f * out_w * out_h;
        for (size_t y = 0; y < out_h; ++y) {
          const float* __restrict__ os_row = osf + y * out_w;
          const float* __restrict__ p = pf;
          float* __restrict__ ps = psf;
          for (size_t dz = 0; dz < kc; ++dz) {
            const size_t z_off = dz * in_w * in_h;
            for (size_t dy = 0; dy < kh; ++dy) {
              const size_t y_off = z_off + (stride * y + dy) * in_w;
              for (size_t dx = 0; dx < kw; ++dx) {
                const float w = *p++;
                const float* __restrict__ in_row = I + y_off + dx;
                float* __restrict__ is_row = IS + y_off + dx;
                float ps_acc = 0.f;
                if (stride == 1) {
                  #pragma omp simd reduction(+:ps_acc)
                  for (size_t x = 0; x < out_w; ++x) {
                    const float os = os_row[x];
                    is_row[x] += w * os;
                    ps_acc += in_row[x] * os;
                  }
                } else {
                  for (size_t x = 0; x < out_w; ++x) {
                    const float os = os_row[x];
                    is_row[stride * x] += w * os;
                    ps_acc += in_row[stride * x] * os;
                  }
                }
                *ps++ += ps_acc;
              }
            }
          }
        }
      }
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < params_size; ++i) {
        PG[i] += PS[i];
      }
    }
  }
}

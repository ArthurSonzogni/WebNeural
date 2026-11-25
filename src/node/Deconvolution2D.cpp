#include "node/Deconvolution2D.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

Deconvolution2D::Deconvolution2D(Node* node,
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

  size_output = {
      (size_input[0] - 1) * stride + size_params[0],
      (size_input[1] - 1) * stride + size_params[1],
      num_features,
  };

  params = Tensor::Random(size_params);
  params *= 1.0f / sqrt(sizes[0] * sizes[1] * size_input[2]);

  InitBuffers(size_output);
}

void Deconvolution2D::Forward(size_t batch_size) {
  const size_t in_w = size_input[0];
  const size_t in_h = size_input[1];
  const size_t in_c = size_input[2];
  const size_t kw = size_params[0];
  const size_t kh = size_params[1];
  const size_t out_w = size_output[0];
  const size_t out_h = size_output[1];
  const size_t out_c = size_params[3];
  const float* __restrict__ P = params.values.data();

  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* __restrict__ I = inputs[0]->output[batch].values.data();
    float* __restrict__ O = output[batch].values.data();
    std::fill_n(O, out_w * out_h * out_c, 0.f);

    for (size_t dz = 0; dz < out_c; ++dz) {
      float* __restrict__ out_z = O + dz * out_w * out_h;
      for (size_t z = 0; z < in_c; ++z) {
        const float* __restrict__ in_z = I + z * in_w * in_h;
        const float* __restrict__ pz = P + kw * kh * (z + in_c * dz);
        for (size_t y = 0; y < in_h; ++y) {
          const float* __restrict__ in_row = in_z + y * in_w;
          const float* __restrict__ p = pz;
          for (size_t dy = 0; dy < kh; ++dy) {
            float* __restrict__ out_y = out_z + (stride * y + dy) * out_w;
            for (size_t dx = 0; dx < kw; ++dx) {
              const float w = *p++;
              float* __restrict__ out_row = out_y + dx;
              if (stride == 1) {
                #pragma omp simd
                for (size_t x = 0; x < in_w; ++x) {
                  out_row[x] += w * in_row[x];
                }
              } else {
                for (size_t x = 0; x < in_w; ++x) {
                  out_row[stride * x] += w * in_row[x];
                }
              }
            }
          }
        }
      }
    }
  }
}

void Deconvolution2D::Backward(size_t batch_size) {
  const size_t in_w = size_input[0];
  const size_t in_h = size_input[1];
  const size_t in_c = size_input[2];
  const size_t kw = size_params[0];
  const size_t kh = size_params[1];
  const size_t out_w = size_output[0];
  const size_t out_h = size_output[1];
  const size_t out_c = size_params[3];
  const size_t params_size = params.values.size();
  const float* __restrict__ P = params.values.data();
  float* __restrict__ PG = params_grad.values.data();

  #pragma omp parallel if(batch_size > 1)
  {
    std::vector<float> local_ps(params_size, 0.f);
    float* __restrict__ PS = local_ps.data();

    #pragma omp for
    for (size_t batch = 0; batch < batch_size; ++batch) {
      const float* __restrict__ I = inputs[0]->output[batch].values.data();
      float* __restrict__ IS = inputs[0]->grad[batch].values.data();
      const float* __restrict__ OS = grad[batch].values.data();

      for (size_t dz = 0; dz < out_c; ++dz) {
        const float* __restrict__ os_z = OS + dz * out_w * out_h;
        for (size_t z = 0; z < in_c; ++z) {
          const float* __restrict__ in_z = I + z * in_w * in_h;
          float* __restrict__ is_z = IS + z * in_w * in_h;
          const size_t p_off = kw * kh * (z + in_c * dz);
          const float* __restrict__ pz = P + p_off;
          float* __restrict__ psz = PS + p_off;
          for (size_t y = 0; y < in_h; ++y) {
            const float* __restrict__ in_row = in_z + y * in_w;
            float* __restrict__ is_row = is_z + y * in_w;
            const float* __restrict__ p = pz;
            float* __restrict__ ps = psz;
            for (size_t dy = 0; dy < kh; ++dy) {
              const float* __restrict__ os_y =
                  os_z + (stride * y + dy) * out_w;
              for (size_t dx = 0; dx < kw; ++dx) {
                const float w = *p++;
                const float* __restrict__ os_row = os_y + dx;
                float ps_acc = 0.f;
                if (stride == 1) {
                  #pragma omp simd reduction(+:ps_acc)
                  for (size_t x = 0; x < in_w; ++x) {
                    const float os = os_row[x];
                    is_row[x] += w * os;
                    ps_acc += in_row[x] * os;
                  }
                } else {
                  for (size_t x = 0; x < in_w; ++x) {
                    const float os = os_row[stride * x];
                    is_row[x] += w * os;
                    ps_acc += in_row[x] * os;
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

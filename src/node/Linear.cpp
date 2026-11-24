#include <iostream>
#include "node/Linear.hpp"
#include <cmath>

Linear::Linear(Node* node, std::vector<size_t> output_sizes) {
  Link(node);

  input_size = input[0]->values.size();
  output_size = Multiply(output_sizes);

  params = Tensor({input_size + 1, output_size});
  output = std::vector<Tensor>(T, Tensor(output_sizes));

  params.Randomize();
  params *= 1.f / sqrt(input_size);

  InitInternalSensitivity();
}

void Linear::Forward(size_t batch_size) {
  const size_t row_stride = input_size + 1;
  const float* __restrict__ P = params.values.data();

  #pragma omp parallel for schedule(static) if(output_size * batch_size >= 1024)
  for (size_t output_index = 0; output_index < output_size; ++output_index) {
    const float* __restrict__ w = P + output_index * row_stride;
    const float bias = w[input_size];

    size_t batch = 0;
    for (; batch + 7 < batch_size; batch += 8) {
      const float* __restrict__ i0 = input[batch + 0]->values.data();
      const float* __restrict__ i1 = input[batch + 1]->values.data();
      const float* __restrict__ i2 = input[batch + 2]->values.data();
      const float* __restrict__ i3 = input[batch + 3]->values.data();
      const float* __restrict__ i4 = input[batch + 4]->values.data();
      const float* __restrict__ i5 = input[batch + 5]->values.data();
      const float* __restrict__ i6 = input[batch + 6]->values.data();
      const float* __restrict__ i7 = input[batch + 7]->values.data();

      float v0 = bias, v1 = bias, v2 = bias, v3 = bias;
      float v4 = bias, v5 = bias, v6 = bias, v7 = bias;
      #pragma omp simd reduction(+:v0,v1,v2,v3,v4,v5,v6,v7)
      for (size_t k = 0; k < input_size; ++k) {
        const float wk = w[k];
        v0 += i0[k] * wk;
        v1 += i1[k] * wk;
        v2 += i2[k] * wk;
        v3 += i3[k] * wk;
        v4 += i4[k] * wk;
        v5 += i5[k] * wk;
        v6 += i6[k] * wk;
        v7 += i7[k] * wk;
      }

      output[batch + 0][output_index] = v0;
      output[batch + 1][output_index] = v1;
      output[batch + 2][output_index] = v2;
      output[batch + 3][output_index] = v3;
      output[batch + 4][output_index] = v4;
      output[batch + 5][output_index] = v5;
      output[batch + 6][output_index] = v6;
      output[batch + 7][output_index] = v7;
    }

    for (; batch < batch_size; ++batch) {
      const float* __restrict__ in = input[batch]->values.data();
      float v = bias;
      #pragma omp simd reduction(+:v)
      for (size_t k = 0; k < input_size; ++k) {
        v += in[k] * w[k];
      }
      output[batch][output_index] = v;
    }
  }
}

void Linear::Backward(size_t batch_size) {
  const size_t row_stride = input_size + 1;
  const float* __restrict__ P = params.values.data();
  float* __restrict__ PS0 = params_sensitivity[0].values.data();
  ps_batches_used = 1;

  // 1) Accumulate parameter gradients across the batch directly into PS0
  //    (each output_index owns a disjoint row, so no synchronization needed).
  #pragma omp parallel for schedule(static) if(output_size * batch_size >= 1024)
  for (size_t output_index = 0; output_index < output_size; ++output_index) {
    float* __restrict__ ps = PS0 + output_index * row_stride;
    float bias_grad = 0.f;

    size_t batch = 0;
    for (; batch + 3 < batch_size; batch += 4) {
      const float os0 = (*output_sensitivity[batch + 0])[output_index];
      const float os1 = (*output_sensitivity[batch + 1])[output_index];
      const float os2 = (*output_sensitivity[batch + 2])[output_index];
      const float os3 = (*output_sensitivity[batch + 3])[output_index];
      const float* __restrict__ i0 = input[batch + 0]->values.data();
      const float* __restrict__ i1 = input[batch + 1]->values.data();
      const float* __restrict__ i2 = input[batch + 2]->values.data();
      const float* __restrict__ i3 = input[batch + 3]->values.data();

      #pragma omp simd
      for (size_t k = 0; k < input_size; ++k) {
        ps[k] += i0[k] * os0 + i1[k] * os1 + i2[k] * os2 + i3[k] * os3;
      }
      bias_grad += os0 + os1 + os2 + os3;
    }

    for (; batch < batch_size; ++batch) {
      const float os = (*output_sensitivity[batch])[output_index];
      const float* __restrict__ in = input[batch]->values.data();
      #pragma omp simd
      for (size_t k = 0; k < input_size; ++k) {
        ps[k] += in[k] * os;
      }
      bias_grad += os;
    }
    ps[input_size] += bias_grad;
  }

  // 2) Compute input gradients in batch tiles of 4 to reuse params in cache.
  const size_t full_tiles = batch_size / 4;
  #pragma omp parallel for schedule(static) if(full_tiles > 1)
  for (size_t tile = 0; tile < full_tiles; ++tile) {
    const size_t b = tile * 4;
    float* __restrict__ is0 = input_sensitivity[b + 0].values.data();
    float* __restrict__ is1 = input_sensitivity[b + 1].values.data();
    float* __restrict__ is2 = input_sensitivity[b + 2].values.data();
    float* __restrict__ is3 = input_sensitivity[b + 3].values.data();
    std::fill_n(is0, input_size, 0.f);
    std::fill_n(is1, input_size, 0.f);
    std::fill_n(is2, input_size, 0.f);
    std::fill_n(is3, input_size, 0.f);

    const float* __restrict__ os_ptr0 = output_sensitivity[b + 0]->values.data();
    const float* __restrict__ os_ptr1 = output_sensitivity[b + 1]->values.data();
    const float* __restrict__ os_ptr2 = output_sensitivity[b + 2]->values.data();
    const float* __restrict__ os_ptr3 = output_sensitivity[b + 3]->values.data();

    for (size_t output_index = 0; output_index < output_size; ++output_index) {
      const float* __restrict__ w = P + output_index * row_stride;
      const float os0 = os_ptr0[output_index];
      const float os1 = os_ptr1[output_index];
      const float os2 = os_ptr2[output_index];
      const float os3 = os_ptr3[output_index];
      #pragma omp simd
      for (size_t k = 0; k < input_size; ++k) {
        const float wk = w[k];
        is0[k] += wk * os0;
        is1[k] += wk * os1;
        is2[k] += wk * os2;
        is3[k] += wk * os3;
      }
    }
  }

  for (size_t batch = full_tiles * 4; batch < batch_size; ++batch) {
    float* __restrict__ is = input_sensitivity[batch].values.data();
    std::fill_n(is, input_size, 0.f);
    const float* __restrict__ os_ptr = output_sensitivity[batch]->values.data();
    for (size_t output_index = 0; output_index < output_size; ++output_index) {
      const float* __restrict__ w = P + output_index * row_stride;
      const float os = os_ptr[output_index];
      #pragma omp simd
      for (size_t k = 0; k < input_size; ++k) {
        is[k] += w[k] * os;
      }
    }
  }
}

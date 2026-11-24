#include "LossFunction.hpp"
#include <cmath>
#include <iostream>
#include "util/stable_softmax.hpp"

namespace LossFunction {

void SquaredDifference(const Tensor& target,
                       const Tensor& current,
                       float* error,
                       Tensor* derivative) {
  const size_t size = current.values.size();
  if (derivative->values.size() != size) {
    *derivative = Tensor(current.sizes);
  }
  const float* __restrict__ cur = current.values.data();
  const float* __restrict__ tgt = target.values.data();
  float* __restrict__ der = derivative->values.data();
  float err = 0.f;
  #pragma omp simd reduction(+:err)
  for (size_t i = 0; i < size; ++i) {
    const float d = cur[i] - tgt[i];
    der[i] = d;
    err += d * d;
  }
  *error = err;
}

// Sum(target, log(current));
void CrossEntropy(const Tensor& target,
                  const Tensor& current,
                  float* error,
                  Tensor* derivative) {
  *error = 0.f;
  if (derivative->values.size() != target.values.size()) {
    *derivative = Tensor(target.sizes);
  }
  derivative->Fill(0.f);
  for (size_t i = 0; i < target.values.size(); ++i) {
    if (target[i] > 0.f) {
      float c = std::max(current[i], 1e-10f);
      *error += -target[i] * std::log(c / target[i]);
      (*derivative)[i] -= target[i] / c;
    }
  }
}

void SoftmaxCrossEntropy(const Tensor& target,
                         const Tensor& current,
                         float* error,
                         Tensor* derivative) {
  const size_t size = target.values.size();
  if (derivative->values.size() != size) {
    *derivative = Tensor(size);
  }
  thread_local std::vector<float> softmax;
  softmax.resize(size);
  StableSoftmax(current.values, softmax);

  *error = 0.f;
  for (size_t i = 0; i < size; ++i) {
    if (target[i] > 0.5) {
      *error += -log(std::max(1e-10f, softmax[i]));
    }
  }

  // Compute derivative.
  for (size_t i = 0; i < size; ++i)
    (*derivative)[i] = softmax[i] - target[i];
}

void BinaryCrossEntropy(const Tensor& target,
                        const Tensor& current,
                        float* error,
                        Tensor* derivative) {
  const size_t size = current.values.size();
  if (derivative->values.size() != size) {
    *derivative = Tensor(current.sizes);
  }
  const float* __restrict__ cur = current.values.data();
  const float* __restrict__ tgt = target.values.data();
  float* __restrict__ der = derivative->values.data();

  float err = 0.f;
  constexpr float eps = 1e-5f;
  for (size_t i = 0; i < size; ++i) {
    const float y = tgt[i];
    const float c = std::max(eps, std::min(1.f - eps, cur[i]));
    err -= y * std::log(c) + (1.f - y) * std::log(1.f - c);
    der[i] = (c - y) / (c * (1.f - c));
  }
  *error = err;
}

void WasserStein(const Tensor& target,
                 const Tensor& current,
                 float* error,
                 Tensor* derivative) {
  *error = target[0] * current[0];
  (*derivative)[0] = target[0];
}

}  // namespace LossFunction

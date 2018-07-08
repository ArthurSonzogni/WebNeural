#include <cmath>
#include "Softmax.hpp"

Softmax::Softmax(Node& node) {
  Link(node);
  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void Softmax::Forward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    float best = I[0];
    for (size_t i = 1; i < size; ++i) {
      best = std::max(best, I[i]);
    }
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
      O[i] = exp(I[i] - best);
      sum += O[i];
    }
    float inv_sum = 1.f / sum;
    for (size_t i = 0; i < size; ++i) {
      O[i] *= inv_sum;
    }
  }
}

void Softmax::Backward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);

    IS.Fill(0.f);

    /* Original.
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        float delta = (i == j) ? 1.f : 0.f;
        IS[i] +=
            output[i] * (delta - output[j]) * (*output_sensitivity)[j];
      }
    }
    */

    // Optimized
    float accu = 0.f;
    for (size_t j = 0; j < size; ++j) {
      accu += -O[j] * OS[j];
    }
    for (size_t i = 0; i < size; ++i) {
      IS[i] = O[i] * (accu + OS[i]);
    }
  }
}

#include <cmath>
#include "Softmax.hpp"
#include "util/stable_softmax.hpp"

Softmax::Softmax(Node* node) {
  Link(node);
  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void Softmax::Forward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);
    StableSoftmax(I.values, O.values);
  }
}

void Softmax::Backward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);

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
    for (size_t j = 0; j < size; ++j)
      accu += O[j] * OS[j];
    for (size_t i = 0; i < size; ++i)
      IS[i] = O[i] * (OS[i] - accu);
  }
}

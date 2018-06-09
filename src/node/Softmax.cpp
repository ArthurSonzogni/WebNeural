#include <iostream>
#include <cmath>
#include "Softmax.hpp"

Softmax::Softmax(Node& node) {
  Link(node);

  output = Tensor(input->sizes);
  output.producer = this;
  input_sensitivity = Tensor(input->sizes);
}

void Softmax::Forward() {
  size_t size = input->values.size();
  float best = (*input)[0];
  for (size_t i = 1; i < size; ++i) {
    best = std::max(best, (*input)[i]);
  }
  float sum = 0.f;
  for (size_t i = 0; i < size; ++i) {
    output[i] = exp((*input)[i] - best);
    sum += output[i];
  }
  float inv_sum = 1.f / sum;
  for (size_t i = 0; i < size; ++i) {
    output[i] *= inv_sum;
  }
}

void Softmax::Backward() {
  input_sensitivity.Fill(0.f);
  size_t size = (*input).values.size();

  /* Original.
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      float delta = (i == j) ? 1.f : 0.f;
      input_sensitivity[i] +=
          output[i] * (delta - output[j]) * (*output_sensitivity)[j];
    }
  }
  */

  // Optimized
  float accu = 0.0;
  for (size_t j = 0; j < size; ++j) {
    accu += -output[j] * output_sensitivity->values[j];
  }
  for (size_t i = 0; i < size; ++i) {
    input_sensitivity[i] = output[i] * (accu + (*output_sensitivity)[i]);
  }
}

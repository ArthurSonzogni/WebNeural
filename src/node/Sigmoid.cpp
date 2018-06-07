#include <cmath>
#include "Sigmoid.hpp"

Sigmoid::Sigmoid(Node& node) {
  Link(node);

  output = Tensor(input->sizes);
  output.producer = this;
  input_sensitivity = Tensor(input->sizes);
}

void Sigmoid::Forward() {
  const size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    output[i] = 1.0 / (1.0 + exp(-(*input)[i]));
  }
}

void Sigmoid::Backward() {
  const size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    input_sensitivity[i] =
        output[i] * (1.f - output[i]) * (*output_sensitivity)[i];
  }
}

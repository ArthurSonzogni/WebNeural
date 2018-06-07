#include "node/Relu.hpp"

Relu::Relu(Node& node) {
  Link(node);

  output = Tensor(input->sizes);
  output.producer = this;

  input_sensitivity = Tensor(input->sizes);
}

void Relu::Forward() {
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    output[i] = (*input)[i] > 0.f ? (*input)[i] : 0.f;
  }
}

void Relu::Backward() {
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    input_sensitivity[i] = (*input)[i] > 0.f ? (*output_sensitivity)[i] : 0.f;
  }
}

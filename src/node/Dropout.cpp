#include "node/Dropout.hpp"

Dropout::Dropout(Node& node, float ratio) : ratio(ratio) {
  Link(node);

  output = Tensor(input->sizes);
  random = Tensor(input->sizes);
  input_sensitivity = Tensor(input->sizes);
}

void Dropout::Forward() {
  random.UniformRandom();
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    if (random[i] <= ratio) {
      output[i] = (*input)[i];
    } else {
      output[i] = 0.f;
    }
  }
}

void Dropout::Backward() {
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    if (random[i] <= ratio) {
      input_sensitivity[i] = random[i] * (*output_sensitivity)[i];
    } else {
      input_sensitivity[i] = 0.f;
    }
  }
}

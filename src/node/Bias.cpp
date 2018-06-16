#include "node/Bias.hpp"

Bias::Bias(Node& node) {
  Link(node);

  output = Tensor(input->sizes);
  
  params = Tensor::Random(input->sizes);
  params_sensitivity = Tensor(input->sizes);

  input_sensitivity = Tensor(input->sizes);
}

void Bias::Forward() {
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    output[i] =  (*input)[i] + params[i];
  }
}

void Bias::Backward() {
  size_t size = input->values.size();
  for (size_t i = 0; i < size; ++i) {
    input_sensitivity[i] = (*output_sensitivity)[i];
    params_sensitivity[i] += (*output_sensitivity)[i] * 0.01f;
  }
}

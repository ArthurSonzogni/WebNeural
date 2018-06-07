#include "Node.hpp"
#include <algorithm>

void Node::Update(float lambda) {
  size_t params_size = params.values.size();
  for (size_t p = 0; p < params_size; ++p) {
    params[p] += params_sensitivity[p] * lambda;
  }

  params_sensitivity.Fill(0.f);
}

void Node::Link(Node& operation) {
  input = &operation.output;
  operation.output_sensitivity = &input_sensitivity;
}

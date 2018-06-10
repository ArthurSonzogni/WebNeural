#include "node/Linear.hpp"
#include <cmath>

Linear::Linear(Node& node, size_t num_output) {
  Link(node);

  input_size = input->values.size();
  output_size = num_output;

  params = Tensor((input_size + 1) * output_size);
  output = Tensor(output_size);
  output.producer = this;

  input_sensitivity = Tensor(input->sizes);
  params_sensitivity = Tensor(params.sizes);

  params.Randomize();
  params *= 1.f / sqrt(input_size);
}

void Linear::Forward() {
  //#pragma omp parallel for
  for (size_t o = 0; o < output_size; ++o) {
    size_t p = o * (input_size + 1);
    float v = 0.f;

    // Linear part.
    for (size_t i = 0; i < input_size; ++i) {
      v += (*input)[i] * params[p];
      ++p;
    }

    // Bias pars.
    v += params[p];
    ++p;

    output[o] = v;
  }
}

void Linear::Backward() {
  input_sensitivity.Fill(0.f);
  size_t p = 0;
  for (size_t o = 0; o < output_size; ++o) {

    // Linear part.
    for (size_t i = 0; i < input_size; ++i) {
      params_sensitivity[p] += (*input)[i] * (*output_sensitivity)[o];
      input_sensitivity[i] += params[p] * (*output_sensitivity)[o];
      ++p;
    }

    // Bias pars.
    params_sensitivity[p] += (*output_sensitivity)[o];
    ++p;
  }
}

#include <iostream>
#include "node/Linear.hpp"
#include <cmath>

Linear::Linear(Node* node, std::vector<size_t> output_sizes) {
  Link(node);

  input_size = input[0]->values.size();
  output_size = Multiply(output_sizes);

  params = Tensor({input_size + 1, output_size});
  output = std::vector<Tensor>(T, Tensor(output_sizes));

  params.Randomize();
  params *= 1.f / sqrt(input_size);

  InitInternalSensitivity();
}

void Linear::Forward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);
    size_t p = 0;
    for (size_t output_index = 0; output_index < output_size; ++output_index) {
      float v = 0.f;

      // Linear part.
      for (size_t input_index = 0; input_index < input_size; ++input_index) {
        v += I[input_index] * params[p++];
      }

      // Bias pars.
      v += params[p++];

      O[output_index] = v;
    }
  }
}

void Linear::Backward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& PS = params_sensitivity[batch];
    Tensor& IS = input_sensitivity[batch];
    Tensor& I = *(input[batch]);
    Tensor& OS = *(output_sensitivity[batch]);

    IS.Fill(0.f);
    size_t p = 0;
    for (size_t output_index = 0; output_index < output_size; ++output_index) {
      const float os = OS[output_index];

      // Linear part.
      for (size_t input_index = 0; input_index < input_size; ++input_index) {
        PS[p] += I[input_index] * os;
        IS[input_index] += params[p++] * os;
      }

      // Bias pars.
      PS[p++] += os;
    }
  }
}

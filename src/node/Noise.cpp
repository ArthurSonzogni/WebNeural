#include "Noise.hpp"
#include <cmath>
#include <random>

Noise::Noise(Node* node, float sigma) : sigma(sigma) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  InitInternalSensitivity();
}

void Noise::Forward(size_t batch_size) {
  if (!training) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
      output[batch] = *(input[batch]);
    }
    return;
  }

  thread_local std::mt19937 local_rng;
  std::normal_distribution<float> random(0.0, sigma);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = I[i] + random(local_rng);
    }
  }
}

void Noise::Backward(size_t batch_size) {
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    IS = OS;
  }
}

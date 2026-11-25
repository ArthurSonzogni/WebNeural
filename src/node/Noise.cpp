#include "Noise.hpp"
#include <cmath>
#include <random>

Noise::Noise(Node* node, float sigma) : sigma(sigma) {
  AddInput(node);
  InitBuffers(inputs[0]->output[0].sizes);
}

void Noise::Forward(size_t batch_size) {
  if (!training) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
      output[batch] = inputs[0]->output[batch];
    }
    return;
  }

  thread_local std::mt19937 local_rng;
  std::normal_distribution<float> random(0.0, sigma);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];
    const size_t size = I.values.size();
    for (size_t i = 0; i < size; ++i) {
      O[i] = I[i] + random(local_rng);
    }
  }
}

void Noise::Backward(size_t batch_size) {
  for (size_t batch = 0; batch < batch_size; ++batch) {
    inputs[0]->grad[batch] += grad[batch];
  }
}

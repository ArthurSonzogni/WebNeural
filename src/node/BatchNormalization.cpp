#include "node/BatchNormalization.hpp"
#include <cmath>

BatchNormalization::BatchNormalization(Node* node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void BatchNormalization::Forward(size_t batch_size) {
  size_t size = output[0].values.size();

  float X = 0.f;
  float XX = 0.f;
  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto& I = input[batch]->values;
    for (size_t i = 0; i < size; ++i) {
      float x = I[i];
      X += x;
      XX += x * x;
    }
  }

  float total = size * batch_size;
  X /= total;
  XX /= total;

  float mean = X;
  inv_dev = 1.0 / std::sqrt(XX - X * X + 1e-8);

  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto& I = input[batch]->values;
    auto& O = output[batch].values;
    for (size_t i = 0; i < size; ++i) {
      O[i] = (I[i] - mean) * inv_dev;
    }
  }
}

void BatchNormalization::Backward(size_t batch_size) {
  const size_t size = output[0].values.size();
  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto& IS = input_sensitivity[batch].values;
    auto& OS = output_sensitivity[batch]->values;
    for (size_t i = 0; i < size; ++i) {
      IS[i] = OS[i] * inv_dev;
    }
  }
}

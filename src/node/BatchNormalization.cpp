#include "node/BatchNormalization.hpp"
#include <cmath>

BatchNormalization::BatchNormalization(Node* node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void BatchNormalization::Forward(size_t batch_size) {
  const size_t size = output[0].values.size();
  if (batch_size == 0 || size == 0)
    return;

  if (training) {
    float sum = 0.f;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto& I = input[batch]->values;
      for (size_t i = 0; i < size; ++i) {
        sum += I[i];
      }
    }

    const float total = static_cast<float>(size * batch_size);
    mean = sum / total;

    float var = 0.f;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto& I = input[batch]->values;
      for (size_t i = 0; i < size; ++i) {
        const float d = I[i] - mean;
        var += d * d;
      }
    }
    var /= total;
    inv_dev = 1.f / std::sqrt(var + 1e-4f);

    if (!has_running_stats) {
      running_mean = mean;
      running_var = var;
      has_running_stats = true;
    } else {
      constexpr float momentum = 0.9f;
      running_mean = momentum * running_mean + (1.f - momentum) * mean;
      running_var = momentum * running_var + (1.f - momentum) * var;
    }
  } else {
    mean = running_mean;
    inv_dev = 1.f / std::sqrt(running_var + 1e-4f);
  }

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
  if (batch_size == 0 || size == 0)
    return;

  if (!training) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto& IS = input_sensitivity[batch].values;
      auto& OS = output_sensitivity[batch]->values;
      for (size_t i = 0; i < size; ++i) {
        IS[i] = OS[i] * inv_dev;
      }
    }
    return;
  }

  const float total = static_cast<float>(size * batch_size);
  float sum_os = 0.f;
  float sum_os_xhat = 0.f;
  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto& O = output[batch].values;
    auto& OS = output_sensitivity[batch]->values;
    for (size_t i = 0; i < size; ++i) {
      sum_os += OS[i];
      sum_os_xhat += OS[i] * O[i];
    }
  }
  const float mean_os = sum_os / total;
  const float mean_os_xhat = sum_os_xhat / total;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    auto& O = output[batch].values;
    auto& IS = input_sensitivity[batch].values;
    auto& OS = output_sensitivity[batch]->values;
    for (size_t i = 0; i < size; ++i) {
      IS[i] = inv_dev * (OS[i] - mean_os - O[i] * mean_os_xhat);
    }
  }
}

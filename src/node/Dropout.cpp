#include "node/Dropout.hpp"

Dropout::Dropout(Node* node, float ratio) : ratio(ratio) {
  AddInput(node);
  const auto& sizes = inputs[0]->output[0].sizes;
  random = {Tensor(sizes)};
  InitBuffers(sizes);
}

void Dropout::EnsureBatchSize(size_t batch_size) {
  Node::EnsureBatchSize(batch_size);
  if (output.empty())
    return;
  const std::vector<size_t> sizes = output[0].sizes;
  if (random.size() < batch_size) {
    random.reserve(batch_size);
    while (random.size() < batch_size) {
      random.emplace_back(sizes);
    }
  }
}

void Dropout::Forward(size_t batch_size) {
  if (!training) {
    #pragma omp parallel for if(batch_size > 1)
    for (size_t batch = 0; batch < batch_size; ++batch) {
      output[batch] = inputs[0]->output[batch];
    }
    return;
  }

  const float scale = ratio > 0.f ? 1.f / ratio : 0.f;
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& I = inputs[0]->output[batch];
    Tensor& O = output[batch];
    Tensor& R = random[batch];

    R.UniformRandom();
    const size_t size = I.values.size();
    for (size_t index = 0; index < size; ++index) {
      O[index] = (R[index] <= ratio) ? I[index] * scale : 0.f;
    }
  }
}

void Dropout::Backward(size_t batch_size) {
  if (!training) {
    #pragma omp parallel for if(batch_size > 1)
    for (size_t batch = 0; batch < batch_size; ++batch) {
      inputs[0]->grad[batch] += grad[batch];
    }
    return;
  }

  const float scale = ratio > 0.f ? 1.f / ratio : 0.f;
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& OS = grad[batch];
    Tensor& IS = inputs[0]->grad[batch];
    const Tensor& R = random[batch];

    const size_t size = OS.values.size();
    for (size_t index = 0; index < size; ++index) {
      if (R[index] <= ratio) {
        IS[index] += OS[index] * scale;
      }
    }
  }
}

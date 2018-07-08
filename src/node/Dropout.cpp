#include "node/Dropout.hpp"

Dropout::Dropout(Node& node, float ratio) : ratio(ratio) {
  Link(node);

  params = Tensor();
  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  random = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  InitInternalSensitivity();
}

void Dropout::Forward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& O = output[batch];
    Tensor& R = random[batch];

    R.UniformRandom();
    const size_t size = I.values.size();
    for (size_t index = 0; index < size; ++index) {
      if (R[index] <= ratio) {
        O[index] = I[index];
      } else {
        O[index] = 0.f;
      }
    }
  }
}

void Dropout::Backward(size_t batch_size) {
  #pragma omp parallel for
  for(size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& I = *(input[batch]);
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& IS = input_sensitivity[batch];
    Tensor& R = random[batch];

    const size_t size = I.values.size();
    for (size_t index = 0; index < size; ++index) {
      if (R[index] <= ratio) {
        IS[index] = R[index] * OS[index];
      } else {
        IS[index] = 0.f;
      }
    }
  }
}

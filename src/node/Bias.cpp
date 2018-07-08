#include "node/Bias.hpp"

Bias::Bias(Node& node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  params = Tensor::Random(input[0]->sizes);
  InitInternalSensitivity();
}

void Bias::Forward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);
    // -------------------------------------------------------------------------
    for (size_t index = 0; index < size; ++index)
      O[index] = I[index] + params[index];
    // -------------------------------------------------------------------------
  }
}

void Bias::Backward(size_t batch_size) {
  const size_t size = input[0]->values.size();
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& PS = params_sensitivity[batch];
    for (size_t index = 0; index < size; ++index) {
      IS[index] = OS[index];
      PS[index] += OS[index];
    }
  }
}

#include "node/Border.hpp"

Border::Border(Node* node, size_t border_size, float value)
    : border_size(border_size), value(value) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  dim_x = input[0]->sizes[0];
  dim_y = input[0]->sizes[1];

  InitInternalSensitivity();
}

void Border::Forward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);
    O = I;

    for (size_t y = 0; y < dim_y; ++y) {
      for (size_t x = 0; x < border_size; ++x) {
        O.at(x, y) = value;
        O.at(dim_x - x - 1, y) = value;
      }
    }

    for (size_t x = 0; x < dim_x; ++x) {
      for (size_t y = 0; y < border_size; ++y) {
        O.at(x, y) = value;
        O.at(x, dim_y - y - 1) = value;
      }
    }
  }
}

void Border::Backward(size_t batch_size) {
  #pragma omp parallel for
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& IS = input_sensitivity[batch];

    IS = OS;

    for (size_t y = 0; y < dim_y; ++y) {
      for (size_t x = 0; x < border_size; ++x) {
        IS.at(x, y) = 0.f;
        IS.at(dim_x - x - 1, y) = 0.f;
      }
    }

    for (size_t x = 0; x < dim_x; ++x) {
      for (size_t y = 0; y < border_size; ++y) {
        IS.at(x, y) = 0.f;
        IS.at(x, dim_y - y - 1) = 0.f;
      }
    }
  }
}

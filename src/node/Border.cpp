#include "node/Border.hpp"
#include <algorithm>

Border::Border(Node* node, size_t border_size, float value)
    : border_size(border_size), value(value) {
  AddInput(node);

  const auto& in_sizes = inputs[0]->output[0].sizes;
  dim_x = in_sizes[0];
  dim_y = in_sizes[1];
  dim_z = in_sizes.size() >= 3 ? in_sizes[2] : 1;

  InitBuffers(in_sizes);
}

void Border::Forward(size_t batch_size) {
  const size_t bx = std::min(border_size, dim_x);
  const size_t by = std::min(border_size, dim_y);
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];
    O = I;

    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = 0; y < dim_y; ++y) {
        for (size_t x = 0; x < bx; ++x) {
          O.at(x, y, z) = value;
          O.at(dim_x - x - 1, y, z) = value;
        }
      }
      for (size_t x = 0; x < dim_x; ++x) {
        for (size_t y = 0; y < by; ++y) {
          O.at(x, y, z) = value;
          O.at(x, dim_y - y - 1, z) = value;
        }
      }
    }
  }
}

void Border::Backward(size_t batch_size) {
  const size_t bx = std::min(border_size, dim_x);
  const size_t by = std::min(border_size, dim_y);
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    const Tensor& OS = grad[batch];
    Tensor& IS = inputs[0]->grad[batch];

    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = by; y + by < dim_y; ++y) {
        for (size_t x = bx; x + bx < dim_x; ++x) {
          IS.at(x, y, z) += OS.at(x, y, z);
        }
      }
    }
  }
}

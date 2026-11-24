#include "node/Border.hpp"
#include <algorithm>

Border::Border(Node* node, size_t border_size, float value)
    : border_size(border_size), value(value) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));

  dim_x = input[0]->sizes[0];
  dim_y = input[0]->sizes[1];
  dim_z = input[0]->sizes.size() >= 3 ? input[0]->sizes[2] : 1;

  InitInternalSensitivity();
}

void Border::Forward(size_t batch_size) {
  const size_t bx = std::min(border_size, dim_x);
  const size_t by = std::min(border_size, dim_y);
  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);
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
    Tensor& OS = *(output_sensitivity[batch]);
    Tensor& IS = input_sensitivity[batch];

    IS = OS;

    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = 0; y < dim_y; ++y) {
        for (size_t x = 0; x < bx; ++x) {
          IS.at(x, y, z) = 0.f;
          IS.at(dim_x - x - 1, y, z) = 0.f;
        }
      }

      for (size_t x = 0; x < dim_x; ++x) {
        for (size_t y = 0; y < by; ++y) {
          IS.at(x, y, z) = 0.f;
          IS.at(x, dim_y - y - 1, z) = 0.f;
        }
      }
    }
  }
}

#include "BilinearUpsampling.hpp"

BilinearUpsampling::BilinearUpsampling(Node* node) {
  AddInput(node);

  const auto& in_sizes = inputs[0]->output[0].sizes;
  const size_t dim_z = in_sizes.size() >= 3 ? in_sizes[2] : 1;
  InitBuffers({
      in_sizes[0] * 2 + 1,
      in_sizes[1] * 2 + 1,
      dim_z,
  });
}

void BilinearUpsampling::Forward(size_t batch_size) {
  const auto& in_sizes = inputs[0]->output[0].sizes;
  const size_t dim_z = in_sizes.size() >= 3 ? in_sizes[2] : 1;
  const size_t dim_iy = in_sizes[1];
  const size_t dim_ix = in_sizes[0];

  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& O = output[batch];
    const Tensor& I = inputs[0]->output[batch];

    O.Fill(0.f);
    const float* input_v = I.values.data();
    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = 0; y < dim_iy; ++y) {
        for (size_t x = 0; x < dim_ix; ++x) {
          const float v = *(input_v++);
          const size_t X = 2 * x;
          const size_t Y = 2 * y;
          O.at(X + 0, Y + 0, z) += v * 0.25f;
          O.at(X + 1, Y + 0, z) += v * 0.5f;
          O.at(X + 2, Y + 0, z) += v * 0.25f;
          O.at(X + 0, Y + 1, z) += v * 0.5f;
          O.at(X + 1, Y + 1, z) += v * 1.0f;
          O.at(X + 2, Y + 1, z) += v * 0.5f;
          O.at(X + 0, Y + 2, z) += v * 0.25f;
          O.at(X + 1, Y + 2, z) += v * 0.5f;
          O.at(X + 2, Y + 2, z) += v * 0.25f;
        }
      }
    }
  }
}

void BilinearUpsampling::Backward(size_t batch_size) {
  const auto& in_sizes = inputs[0]->output[0].sizes;
  const size_t dim_z = in_sizes.size() >= 3 ? in_sizes[2] : 1;
  const size_t dim_iy = in_sizes[1];
  const size_t dim_ix = in_sizes[0];

  #pragma omp parallel for if(batch_size > 1)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    Tensor& IS = inputs[0]->grad[batch];
    const Tensor& OS = grad[batch];
    float* IS_p = IS.values.data();
    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = 0; y < dim_iy; ++y) {
        for (size_t x = 0; x < dim_ix; ++x) {
          float v = 0.f;
          const size_t X = 2 * x;
          const size_t Y = 2 * y;
          v += OS.at(X + 0, Y + 0, z) * 0.25f;
          v += OS.at(X + 1, Y + 0, z) * 0.5f;
          v += OS.at(X + 2, Y + 0, z) * 0.25f;
          v += OS.at(X + 0, Y + 1, z) * 0.5f;
          v += OS.at(X + 1, Y + 1, z) * 1.0f;
          v += OS.at(X + 2, Y + 1, z) * 0.5f;
          v += OS.at(X + 0, Y + 2, z) * 0.25f;
          v += OS.at(X + 1, Y + 2, z) * 0.5f;
          v += OS.at(X + 2, Y + 2, z) * 0.25f;
          *(IS_p++) += v;
        }
      }
    }
  }
}

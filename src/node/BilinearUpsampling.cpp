#include "BilinearUpsampling.hpp"

BilinearUpsampling::BilinearUpsampling(Node* node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor({
                                      input[0]->sizes[0] * 2 + 2,  //
                                      input[0]->sizes[1] * 2 + 2,  //
                                      input[0]->sizes[2]           //
                                  }));

  params = Tensor();

  InitInternalSensitivity();
}

void BilinearUpsampling::Forward(size_t batch_size) {
  const size_t dim_z = input[0]->sizes[2];
  const size_t dim_iy = input[0]->sizes[1];
  const size_t dim_ix = input[0]->sizes[0];

  #pragma omp parallel for
  for(size_t batch = 0; batch<batch_size; ++batch) {
    Tensor& O = output[batch];
    Tensor& I = *(input[batch]);

    O.Fill(0.f);
    float* input_v = &I[0];
    for (size_t z = 0; z < dim_z; ++z) {
      for (size_t y = 0; y < dim_iy; ++y) {
        for (size_t x = 0; x < dim_ix; ++x) {
          const float v = *(input_v++);
          const size_t X = 2 * x;
          const size_t Y = 2 * y;
          O.at(X + 0, Y + 0, z) += v * 0.25;
          O.at(X + 1, Y + 0, z) += v * 0.5;
          O.at(X + 2, Y + 0, z) += v * 0.25;
          O.at(X + 0, Y + 1, z) += v * 0.5;
          O.at(X + 1, Y + 1, z) += v * 1.0;
          O.at(X + 2, Y + 1, z) += v * 0.5;
          O.at(X + 0, Y + 2, z) += v * 0.25;
          O.at(X + 1, Y + 2, z) += v * 0.5;
          O.at(X + 2, Y + 2, z) += v * 0.25;
        }
      }
    }
  }
}

void BilinearUpsampling::Backward(size_t batch_size) {
  const size_t dim_z = input[0]->sizes[2];
  const size_t dim_iy = input[0]->sizes[1];
  const size_t dim_ix = input[0]->sizes[0];

  #pragma omp parallel for
  for(size_t batch = 0; batch<batch_size; ++batch) {
    Tensor& IS = input_sensitivity[batch];
    Tensor& OS = *(output_sensitivity[batch]);
    float* IS_p = &(IS.values[0]);
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

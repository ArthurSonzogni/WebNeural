#include "BilinearUpsampling.hpp"

BilinearUpsampling::BilinearUpsampling(Node& node) {
  Link(node);

  output = Tensor({
      input->sizes[0] * 2 + 2,  //
      input->sizes[1] * 2 + 2,  //
      input->sizes[2]           //
  });

  input_sensitivity = Tensor(input->sizes);
}

void BilinearUpsampling::Forward() {
  const size_t dim_z = input->sizes[2];
  const size_t dim_iy = input->sizes[1];
  const size_t dim_ix = input->sizes[0];

  output.Fill(0.f);
  float* input_v = &(input->values[0]);
  for (size_t z = 0; z < dim_z; ++z) {
    for (size_t y = 0; y < dim_iy; ++y) {
      for (size_t x = 0; x < dim_ix; ++x) {
        const float v = *(input_v++);
        const size_t X = 2*x;
        const size_t Y = 2*y;
        output.at(X + 0, Y + 0, z) += v * 0.25;
        output.at(X + 1, Y + 0, z) += v * 0.5;
        output.at(X + 2, Y + 0, z) += v * 0.25;
        output.at(X + 0, Y + 1, z) += v * 0.5;
        output.at(X + 1, Y + 1, z) += v * 1.0;
        output.at(X + 2, Y + 1, z) += v * 0.5;
        output.at(X + 0, Y + 2, z) += v * 0.25;
        output.at(X + 1, Y + 2, z) += v * 0.5;
        output.at(X + 2, Y + 2, z) += v * 0.25;
      }
    }
  }
}

void BilinearUpsampling::Backward() {
  const size_t dim_z = input->sizes[2];
  const size_t dim_iy = input->sizes[1];
  const size_t dim_ix = input->sizes[0];

  float* input_sensitivity_v = &(input_sensitivity.values[0]);
  for (size_t z = 0; z < dim_z; ++z) {
    for (size_t y = 0; y < dim_iy; ++y) {
      for (size_t x = 0; x < dim_ix; ++x) {
        float v = 0.f;
        const size_t X = 2 * x;
        const size_t Y = 2 * y;
        v += output_sensitivity->at(X + 0, Y + 0, z) * 0.25;
        v += output_sensitivity->at(X + 1, Y + 0, z) * 0.5;
        v += output_sensitivity->at(X + 2, Y + 0, z) * 0.25;
        v += output_sensitivity->at(X + 0, Y + 1, z) * 0.5;
        v += output_sensitivity->at(X + 1, Y + 1, z) * 1.0;
        v += output_sensitivity->at(X + 2, Y + 1, z) * 0.5;
        v += output_sensitivity->at(X + 0, Y + 2, z) * 0.25;
        v += output_sensitivity->at(X + 1, Y + 2, z) * 0.5;
        v += output_sensitivity->at(X + 2, Y + 2, z) * 0.25;
        *(input_sensitivity_v++) += v;
      }
    }
  }
}

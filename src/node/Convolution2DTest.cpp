#include <cmath>
#include "node/Input.hpp"
#include "node/Convolution2D.hpp"
#include "Optimizer.hpp"
#include "gtest/gtest.h"
#include "Image.hpp"
#include <fstream>

namespace {

Tensor multi_derivatives(Tensor& input) {
  size_t dimx = input.sizes[0];
  size_t dimy = input.sizes[1];
  Tensor output({dimx-2, dimy-2, 8});
  for(size_t y = 0; y<dimy-2; ++y) {
    for(size_t x = 0; x<dimx-2; ++x) {
      size_t X = x + 1;
      size_t Y = y + 1;

      output.at(x, y, 0) = input.at(X + 1, Y + 0) - input.at(X + 0, Y + 0);
      output.at(x, y, 1) = input.at(X - 1, Y + 0) - input.at(X + 0, Y + 0);
      output.at(x, y, 2) = input.at(X + 0, Y + 1) - input.at(X + 0, Y + 0);
      output.at(x, y, 3) = input.at(X + 0, Y - 1) - input.at(X + 0, Y + 0);

      output.at(x, y, 4) = input.at(X + 1, Y + 1) - input.at(X + 0, Y + 0);
      output.at(x, y, 5) = input.at(X - 1, Y + 1) - input.at(X + 0, Y + 0);
      output.at(x, y, 6) = input.at(X + 1, Y + 1) - input.at(X + 0, Y + 0);
      output.at(x, y, 7) = input.at(X + 1, Y - 1) - input.at(X + 0, Y + 0);
    }
  }
  return output;
}

}  // namespace

TEST(Convolution2D, Convolution2D) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 10000; ++i) {
    Tensor input = Tensor::Random({30,30});
    examples.push_back({input, multi_derivatives(input)});
  }

  // Build a neural network.
  Input input({30,30});
  auto output = Convolution2D(input, {3,3}, 8);

  Optimizer optimizer(output, 100, examples);

  for (int i = 0; i < 1000; ++i) {
    optimizer.Train(0.001f, 100);

    // Check for new predictions.
    float error = optimizer.Error();
    if (error < 0.0001)
      break;
  }

  EXPECT_LE(optimizer.Error(), 0.03);
  Tensor expected_params({3,3,8});
  expected_params.values = {
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +0.f, -1.f, +1.f,  // |
      +0.f, +0.f, +0.f,  // |
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +1.f, -1.f, +0.f,  // |
      +0.f, +0.f, +0.f,  // |
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +0.f, -1.f, +0.f,  // |
      +0.f, +1.f, +0.f,  // |
      // ------------------+
      +0.f, +1.f, +0.f,  // |
      +0.f, -1.f, +0.f,  // |
      +0.f, +0.f, +0.f,  // |
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +0.f, -1.f, +0.f,  // |
      +0.f, +0.f, +1.f,  // |
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +0.f, -1.f, +0.f,  // |
      +1.f, +0.f, +0.f,  // |
      // ------------------+
      +0.f, +0.f, +0.f,  // |
      +0.f, -1.f, +0.f,  // |
      +0.f, +0.f, +1.f,  // |
      // ------------------+
      +0.f, +0.f, +1.f,  // |
      +0.f, -1.f, +0.f,  // |
      +0.f, +0.f, +0.f,  // |
  };

  EXPECT_LE((expected_params - output.params).Error(), 1e-8);
}

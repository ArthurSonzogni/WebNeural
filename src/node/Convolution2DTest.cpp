#include <cmath>
#include "node/Input.hpp"
#include "node/Convolution2D.hpp"
#include "Model.hpp"
#include "gtest/gtest.h"
#include "Image.hpp"

#include <fstream>
#include <chrono>

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
  for (int i = 0; i < 1000; ++i) {
    Tensor input = Tensor::Random({30,30});
    examples.push_back({input, multi_derivatives(input)});
  }

  // Build a neural network.
  Input input({30,30});
  auto output = Convolution2D(&input, {3,3}, 8);

  Model model(&input, &output, examples);

  for (int i = 0; i < 1000; ++i) {
    model.batch_size = 100;
    model.Train(0.000001f, 1000);

    // Check for new predictions.
    float error = model.Error();
    std::cout << "error = " << error << std::endl;
    if (error < 0.0005)
      break;
  }

  EXPECT_LE(model.Error(), 0.0005);
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

  EXPECT_LE((expected_params - output.params).Error(), 0.04);
}

Tensor multi_derivatives_15(Tensor& input) {
  size_t dimx = input.sizes[0];
  size_t dimy = input.sizes[1];
  Tensor output({dimx-14, dimy-14, 8});
  for(size_t y = 0; y<dimy-14; ++y) {
    for(size_t x = 0; x<dimx-14; ++x) {
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

TEST(Convolution2D, PerformanceBigKernel) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 100; ++i) {
    Tensor input = Tensor::Random({128,128});
    examples.push_back({input, multi_derivatives_15(input)});
  }

  // Build a neural network.
  Input input({128,128});
  auto output = Convolution2D(&input, {15,15}, 8);
  Model model(&input, &output, examples);

  auto start = std::chrono::steady_clock::now();
  model.Train(0.001f, 1000);
  auto end = std::chrono::steady_clock::now();

  auto duration = end-start;
  std::cout
      << "duration = "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << std::endl;
}

TEST(Convolution2D, PerformanceSmallKernel) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 100; ++i) {
    Tensor input = Tensor::Random({256,256});
    examples.push_back({input, multi_derivatives(input)});
  }

  // Build a neural network.
  Input input({256,256});
  auto output = Convolution2D(&input, {3,3}, 8);
  Model model(&input, &output, examples);

  auto start = std::chrono::steady_clock::now();
  model.Train(0.001f, 100);
  auto end = std::chrono::steady_clock::now();

  auto duration = end-start;
  std::cout
      << "duration = "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << std::endl;
}

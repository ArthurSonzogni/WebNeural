#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "Optimizer.hpp"
#include "gtest/gtest.h"

namespace {

Tensor f(Tensor& input) {
  Tensor ret({2});
  ret.values[0] = 1 * input[0] + 2 * input[1] + 3 * input[2] + 4;
  ret.values[1] = 5 * input[0] + 6 * input[1] + 7 * input[2] + 8;
  return ret;
}

}  // namespace

TEST(Linear, Linear) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 100; ++i) {
    Tensor input = Tensor::Random({3});
    examples.push_back({input, f(input)});
  }

  // Build a neural network.
  Input input({3});
  auto output = Linear(input, 2);

  // Optimize it.
  Optimizer optimizer(output, 100, examples);
  optimizer.Train(0.4, 2000);

  // Check the tuned params
  Tensor expected_params({8});
  expected_params.values = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  Tensor difference = expected_params - output.params;

  EXPECT_LT(difference.Error(), 1e-4);

  // Check for new predictions.
  for (int i = 0; i < 10; ++i) {
    Tensor input = Tensor::Random({3});
    Tensor output = optimizer.Predict(input);
    EXPECT_LT((output - f(input)).Error(), 1e-4);
  }
}

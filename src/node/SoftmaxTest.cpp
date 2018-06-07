#include <cmath>
#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/Softmax.hpp"
#include "node/Sigmoid.hpp"
#include "Optimizer.hpp"
#include "gtest/gtest.h"

namespace {

Tensor xor_function(Tensor& input) {
  Tensor ret({2});
  if ((input[0] > 0.0) == (input[1] > 0.0))
    ret[0] = 1.0;
  else
    ret[0] = 0.0;
  ret[1] = 1.0 - ret[0];
  return ret;
}

}  // namespace

TEST(Softmax, Softmax) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 10000; ++i) {
    Tensor input = Tensor::Random({2});
    examples.push_back({input, xor_function(input)});
  }

  // Build a neural network.
  Input input({2});
  auto a = Linear(input, 4);
  auto b = Sigmoid(a);
  auto c = Linear(b, 2);
  auto d = Softmax(c);
  auto& output = d;

  Optimizer optimizer(output, 100, examples);

  for (int i = 0; i<1000; ++i) {
    optimizer.Train(1.f, 10000);

    // Check for new predictions.
    float error = optimizer.Error();
    if (error < 0.05)
      break;
  }
  EXPECT_LE(optimizer.Error(), 0.05);
}

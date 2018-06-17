#include <cmath>
#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/Softmax.hpp"
#include "node/Relu.hpp"
#include "Model.hpp"
#include "gtest/gtest.h"

namespace {

Tensor xor_function(Tensor& input) {
  Tensor ret({2,1,1});
  if ((input[0] > 0.0) == (input[1] > 0.0))
    ret[0] = 1.0;
  else
    ret[0] = 0.0;
  ret[1] = 1.0 - ret[0];
  return ret;
}

}  // namespace

TEST(Relu, Relu) {
  // Generate examples.
  std::vector<Example> examples;
  for (int i = 0; i < 10000; ++i) {
    Tensor input = Tensor::Random({2});
    examples.push_back({input, xor_function(input)});
  }

  // Build a neural network.
  Input input({2});
  auto a = Linear(input, {4});
  auto b = Relu(a);
  auto c = Linear(b, {2});
  auto d = Softmax(c);
  auto& output = d;

  Model model(input, output, examples);

  for (int i = 0; i < 1000; ++i) {
    model.Train(0.002f, 10000);

    // Check for new predictions.
    float error = model.Error();
    if (error < 0.03)
      break;
    std::cout << "error = " << error << std::endl;
  }
  EXPECT_LE(model.Error(), 0.03);
}

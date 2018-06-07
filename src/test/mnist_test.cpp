#include "mnist/mnist_reader.hpp"
#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/Sigmoid.hpp"
#include "node/Softmax.hpp"
#include "node/Relu.hpp"
#include "Optimizer.hpp"
#include "gtest/gtest.h"

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({28, 28});
    input_example.values = input[i];
    input_example *= 1.f / 256.f;

    auto output_example = Tensor({10});
    output_example.Fill(0.f);
    output_example.values[output[i]] = 1.f;

    examples.push_back({input_example, output_example});
  }

  return examples;
}

TEST(MNIST, MultiLayerPerceptron) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples(mnist.test_images, mnist.test_labels);

  // Build a neural network.
  Input input({28, 28});
  auto a = Linear(input, 32);
  auto b = Sigmoid(a);
  auto c = Linear(b, 10);
  auto d = Softmax(c);
  auto& output = d;

  Optimizer optimizer(output, 10, training_set);
  Optimizer tester(output, 10, testing_set);

  for (int i = 0; i < 1000; ++i) {
    float error_training = optimizer.ErrorInteger();
    //float error_test = tester.ErrorInteger();
    //std::cout << "Error " << error_training << " " << error_test << std::endl;

    if (error_training < 0.03)
      break;

    optimizer.Train(1.f, training_set.size());
  }

  EXPECT_LE(optimizer.ErrorInteger(), 0.03);
  EXPECT_LE(tester.ErrorInteger(), 0.055);
}

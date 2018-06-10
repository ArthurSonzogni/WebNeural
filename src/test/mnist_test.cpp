#include "Optimizer.hpp"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"
#include "node/Bias.hpp"
#include "node/Convolution2D.hpp"
#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/MaxPooling.hpp"
#include "node/Relu.hpp"
#include "node/Sigmoid.hpp"
#include "node/Softmax.hpp"
#include "node/Dropout.hpp"
#include "Image.hpp"

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({28, 28});
    input_example.values = input[i];
    for (auto& p : input_example.values) {
      p /= 256.0f;
      // p = -1.0 + 2.0*p;
    }

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
    float error_test = tester.ErrorInteger();
    std::cout << "Error " << error_training << " " << error_test << std::endl;

    if (error_training < 0.03)
      break;

    optimizer.Train(1.f, training_set.size());
  }

  EXPECT_LE(optimizer.ErrorInteger(), 0.03);
  EXPECT_LE(tester.ErrorInteger(), 0.055);
}

TEST(MNIST, CNN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples(mnist.test_images, mnist.test_labels);

  // Build a neural network.
  Input input({28, 28});

  // # First convolution layer + pooling
  // {28,28,1}
  auto l1 = Convolution2D(input, {5, 5}, 32);
  auto l2 = MaxPooling(l1);
  auto l3 = Relu(l2);
  // {12,12,32}

  // # Second convolution layer + pooling
  // {12,12,32}
  auto l4 = Convolution2D(l3, {5, 5}, 64);
  auto l5 = MaxPooling(l4);
  auto l6 = Relu(l5);
  // {4,4,64}
  // {4,4,64}
  auto linear = Linear(l6, 128);
  auto relu = Sigmoid(linear);
  // {1024}

  // # Dropout layer
  // {128}
  auto drop = Dropout(relu, 0.4);
  // {128}

  // # Readout layer
  // {128}
  auto l9 = Linear(drop, 10);
  auto output = Softmax(l9);
  //// {10}

  Optimizer optimizer(output, 1000, training_set);
  // Optimizer tester(output, 10, testing_set);

  float lambda = 1.0f;
  for (int i = 1;; ++i) {
    optimizer.Train(lambda, training_set.size() / 1000);

    float error_training = optimizer.LastError();
    for (int i = 0; i < 100; ++i) {
      if (i < error_training * 100)
        std::cout << "#";
      else
        std::cout << "-";
    }
    std::cout << "  Error "
              << " " << error_training << " " << std::endl;

    // if (optimizer.LastError() < 0.03

    if (i % 2 == 0) {
      // std::ofstream("layer_1.pgm") << image_PGM(l1.params);
      // linear.params.sizes = {28,28,32};
      std::ofstream("l1_output.pgm") << image_PGM(l1.output);
      std::ofstream("l1.pgm") << image_PGM(l1.params);
      std::ofstream("m.pgm") << image_PGM(l2.input_sensitivity);
      // std::ofstream("l2_output.pgm") << image_PGM(l4.output);
      // std::ofstream("l2.pgm") << image_PGM(l4.params);
      // std::ofstream("l2_output.pgm") << image_PGM(l2.output);
      // std::ofstream("l3_output.pgm") << image_PGM(l3.output);
      ////std::ofstream("l4_output.pgm") << image_PGM(l4.output);
      ////std::ofstream("layer_2.pgm") << image_PGM(l4.params);
    }
  }

  EXPECT_LE(optimizer.ErrorInteger(), 0.03);
  // EXPECT_LE(tester.ErrorInteger(), 0.055);
}

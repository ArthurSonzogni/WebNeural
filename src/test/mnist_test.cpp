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
  auto l1 = Convolution2D(input, {5,5}, 16);
  auto l2 = MaxPooling(l1);
  auto l3 = Relu(l2);
  // {12,12,16}

  // # Second convolution layer + pooling
  // {12,12,16}
  auto l4 = Convolution2D(l3, {5, 5}, 32);
  auto l5 = MaxPooling(l4);
  auto l6 = Relu(l5);
  // {4,4,32}

  // # Dense layer.
  // {4,4,32}
  auto linear = Linear(l6, 128);
  auto relu = Relu(linear);
  // {128}

  // # Dropout layer
  // {128}
  auto drop = Dropout(relu, 0.4);
  // {128}

  // # Readout layer
  // {128}
  auto l9 = Linear(relu, 10);
  auto output = Softmax(l9);
  // {10}

  training_set.resize(training_set.size()/100);
  testing_set.resize(testing_set.size()/100);
  Optimizer optimizer(output, 100, training_set);
  Optimizer tester(output, 100, testing_set);

  float lambda = 0.01f;
  for (int i = 1;; ++i) {
    float error_training = optimizer.Error();
    //float error_test = tester.Error();
    float error_test = error_training;
    for(int i = 0; i<100; ++i) {
      if (i < error_training * 100)
        std::cout << "O";
      else
        std::cout << "-";
    }
    std::cout << "  Error " << " " << error_training << " " << error_test << " " << std::endl;

    if (error_training < 0.1)
      lambda = 0.1f;

    if (i % 1 == 0) {
      std::ofstream("layer_1.pgm") << image_PGM(l1.params);
      std::ofstream("layer_1_output.pgm") << image_PGM(l1.output);
      //std::ofstream("layer_2.pgm") << image_PGM(l4.params);
    }

    optimizer.Train(lambda, training_set.size());
  }

  EXPECT_LE(optimizer.ErrorInteger(), 0.03);
  EXPECT_LE(tester.ErrorInteger(), 0.055);
}

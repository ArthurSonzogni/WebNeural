#include <random>

#include "Model.hpp"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"
#include "node/Bias.hpp"
#include "node/Convolution2D.hpp"
#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/MaxPooling.hpp"
#include "node/Relu.hpp"
#include "node/BilinearUpsampling.hpp"
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
      p = 0.0 + p;
    }

    auto output_example = Tensor({10,1,1});
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
  auto l_1 = Input({28, 28});
  auto l_2 = Linear(l_1, {32});
  auto l_3 = Sigmoid(l_2);
  auto l_4 = Linear(l_3, {10});
  auto l_5 = Softmax(l_4);

  Model model(l_1, l_5, training_set);
  Model tester(l_1, l_5, testing_set);

  for (int i = 0; i < 1000; ++i) {
    float error_training = model.ErrorInteger();
    float error_test = tester.ErrorInteger();
    std::cout << "Error " << error_training << " " << error_test << std::endl;

    if (error_training < 0.03)
      break;

    model.Train(0.01f, training_set.size());
  }

  EXPECT_LE(model.ErrorInteger(), 0.03);
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
  auto input = Input({28, 28});

  // # First convolution layer + pooling
  // {28,28,1}
  auto l1 = Convolution2D(input, {5, 5}, 10, 2);
  //auto l2 = MaxPooling(l1);
  auto l3 = Relu(l1);
  // {10,10,16}

  // # Second convolution layer + pooling
  // {10,10,16}
  auto l4 = Convolution2D(l3, {5, 5}, 16, 2);
  //auto l5 = MaxPooling(l4);
  auto l6 = Relu(l4);
  // {3,3,32}

  // {3,3,32}
  auto linear = Linear(l3, {512});
  auto relu = Relu(linear);
  // {1024}

  // # Dropout layer
  // {128}
  auto drop = Dropout(relu, 0.4);
  // {128}

  // # Readout layer
  // {128}
  auto l7 = Linear(drop, {10});
  auto output = Softmax(l7);
  //// {10}

  training_set.resize(5000);
  Model model(input, output, training_set);
  Model tester(input, output, testing_set);

  float lambda = 0.002f;
  float last_error = 1.0;
  for (int i = 1;; ++i) {
    model.batch_size = 97 + last_error;
    model.Train(lambda, 200);

    float error_training = model.LastError();
    last_error = last_error * 0.95f + error_training * 0.05f;

    //float error_training = model.ErrorInteger();
    //float error_testing = tester.ErrorInteger();
    for (int i = 0; i < 100; ++i) {
      if (i < last_error * 100)
        std::cout << "#";
      else
        std::cout << "-";
    }
    std::cout << " " << last_error << " " << model.iteration << "/"
              << model.examples.size() << std::endl;

    std::ofstream("l1.ppm") << image_PPM(l1.params);
    std::ofstream("l1.pgm") << image_PGM(l1.params);
    std::ofstream("l1_output.ppm") << image_PPM(l1.output);
    std::ofstream("l1_output.pgm") << image_PGM(l1.output);
    std::ofstream("l3_output.ppm") << image_PPM(l3.output);
    std::ofstream("l3_output.pgm") << image_PGM(l3.output);
    std::ofstream("l4.ppm") << image_PPM(l1.params);

    if (last_error < 0.2)
      break;
  }


  EXPECT_LE(model.ErrorInteger(), 0.005);
  EXPECT_LE(tester.ErrorInteger(), 0.01);
}

TEST(MNIST, GAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples(mnist.test_images, mnist.test_labels);

  // Generative network.
  auto g_i = Input({1});
  auto g_2 = Linear(g_i, {128});
  auto g_3 = Relu(g_2);
  auto g_4 = Linear(g_3, {256});
  auto g_5 = Relu(g_4);
  auto g_6 = Linear(g_5, {784});
  auto g_o = Sigmoid(g_6);
  g_o.output.sizes = {28, 28};

  {
    Model(g_i, g_o).DeserializeParamsFromFile("generative_net");
  }

  // Build a descriminative network.
  auto d_i = Input({28, 28});
  auto d_2 = Linear(d_i, {256});
  auto d_3 = Relu(d_2);
  auto d_4 = Linear(d_3, {64});
  auto d_5 = Relu(d_4);
  auto d_6 = Linear(d_5, {1});
  auto d_o = Sigmoid(d_6);

  {
    Model(d_i, d_o).DeserializeParamsFromFile("discriminative_net");
  }

  std::random_device seed;
  std::mt19937 random_generator(seed());

  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(g_o, d_2);

      // Do not train on the discriminative network.
      Range(d_i, d_o).Apply(&Node::Lock);
      Range(g_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      std::vector<Example> examples;
      for(size_t i = 0; i<training_set.size(); ++i) {
        auto input = Tensor::Random(d_i.params.sizes);
        auto output = Tensor({1,1,1});
        output.values = {1.f};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.batch_size = 100;
      do {
        generated.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); ++i) {
          model.Train(0.001f, 1);
          generated.push_back(g_o.output);
          error += model.LastError();
        }
        generative_error = error /= examples.size();
        std::cerr << "generative_error = " << generative_error << std::endl;
      } while (generative_error > 0.2);

      Model(g_i, g_o).SerializeParamsToFile("generative_net");
    }

    // Print a generated image
    std::ofstream("live.pgm") << image_PGM(generated[0], 0.f, 1.f);

    {
      std::stringstream ss;
      ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
         << ".pgm";
      std::ofstream(ss.str()) << image_PGM(generated[0], 0.f, 1.f);
    }

    // Train the discriminative network.
    {
      // Disconnect the two networks.
      Node::Link(d_i, d_2);
      Range(d_i, d_o).Apply(&Node::Unlock);
      Range(d_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      std::vector<Example> examples;
      for (auto& input : generated) {
        auto output = Tensor({1,1,1});
        output.values = {0.f};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < generated.size(); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1,1,1});
        output.values = {1.f};
        examples.push_back({input, output});
      }
      std::shuffle(examples.begin(), examples.end(), random_generator);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.batch_size = 100;
      do {
        model.Train(0.00001f, examples.size());
        discriminative_error = model.LastError();
        std::cerr << "discriminative_error = " << discriminative_error
                  << std::endl;
      } while(discriminative_error > 0.01);
      Model(d_i, d_o).SerializeParamsToFile("discriminative_net");
    }

    std::cout << "error = " << generative_error << " vs " << discriminative_error << std::endl;
  }
}

TEST(MNIST, CGAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples(mnist.test_images, mnist.test_labels);

  // Generative network.
  auto g_i = Input({20, 20});
  auto g_1 = Linear(g_i, {9, 9, 32});        // -> {9 , 9 , 32}
  auto g_2 = Relu(g_1);                      // -> {9 , 9 , 32}
  auto g_3 = BilinearUpsampling(g_2);        // -> {20, 20, 32}
  auto g_4 = Convolution2D(g_3, {5, 5}, 16); // -> {16, 16, 16}
  auto g_5 = Relu(g_4);                      // -> {16, 16, 16}
  auto g_6 = BilinearUpsampling(g_5);        // -> {34, 34, 16}
  auto g_7 = Convolution2D(g_6, {7, 7}, 1);  // -> {28, 28, 1 }
  auto g_o = Sigmoid(g_7);
  g_o.output.sizes = {28, 28};

  {
    //Model(g_i, g_o).DeserializeParamsFromFile("generative_conv_net");
  }

  // Build a descriminative network.
  auto d_i = Input({28, 28});
  auto d_2 = Convolution2D(d_i, {5,5}, 8); // -> {24,24,8}
  auto d_3 = MaxPooling(d_2);              // -> {12,12,8}
  auto d_4 = Relu(d_3);
  auto d_5 = Convolution2D(d_4, {5,5}, 8); // -> {8,8,16}
  auto d_6 = MaxPooling(d_5);              // -> {4,4,16}
  auto d_7 = Relu(d_6);
  auto d_8 = Linear(d_7, {128});
  auto d_9 = Relu(d_8);
  auto d_10 = Linear(d_9, {1});
  auto d_o = Sigmoid(d_10);

  {
    //Model(g_i, g_o).DeserializeParamsFromFile("discriminative_conv_net");
  }

  std::random_device seed;
  std::mt19937 random_generator(seed());

  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(g_o, d_2);

      // Do not train on the discriminative network.
      Range(d_i, d_o).Apply(&Node::Lock);
      Range(g_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      std::vector<Example> examples;
      for(size_t i = 0; i<2000; ++i) {
        auto input = Tensor::Random(d_i.params.sizes);
        auto output = Tensor({1,1,1});
        output.values = {1.f};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.batch_size = 100;
      do {
        generated.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); ++i) {
          model.Train(0.00001f, 1);
          generated.push_back(g_o.output);
          error += model.LastError();
        }
        generative_error = error /= examples.size();
        std::cerr << "generative_error = " << generative_error << std::endl;
        std::ofstream("other.pgm") << image_PGM(generated[0], 0.f, 1.f);
      } while (generative_error > 0.23f && iteration != 0);

      std::ofstream("generative_net") << Model(g_i, g_o).SerializeParams();
      Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");
    }

    // Print a generated image
    std::ofstream("live.pgm") << image_PGM(generated[0], 0.f, 1.f);

    {
      std::stringstream ss;
      ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
         << ".pgm";
      std::ofstream(ss.str()) << image_PGM(generated[0], 0.f, 1.f);
    }

    // Train the discriminative network.
    {
      // Disconnect the two networks.
      Node::Link(d_i, d_2);
      Range(d_i, d_o).Apply(&Node::Unlock);
      Range(d_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      std::vector<Example> examples;
      for (auto& input : generated) {
        auto output = Tensor({1,1,1});
        output.values = {0.f};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < generated.size(); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1,1,1});
        output.values = {1.f};
        examples.push_back({input, output});
      }
      std::shuffle(examples.begin(), examples.end(), random_generator);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.batch_size = 27;
      do {
        model.Train(0.00005f, examples.size());
        discriminative_error = model.LastError();
        std::cerr << "discriminative_error = " << discriminative_error
                  << std::endl;
      } while(discriminative_error > 0.1f);
      Model(d_i, d_o).SerializeParamsToFile("discriminative_conv_net");
    }

    std::cout << "error = " << generative_error << " vs " << discriminative_error << std::endl;
  }
}

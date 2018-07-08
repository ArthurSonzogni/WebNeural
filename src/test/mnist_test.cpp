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
#include "node/Deconvolution2D.hpp"
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

std::vector<Example> GetExamples2(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({27, 27, 1});
    for(size_t y = 0; y<27; ++y)
    for(size_t x = 0; x<27; ++x) {
      size_t index = x + 28 *y;
      input_example.at(x,y) = input[i][index] / 255.0f;
    }

    examples.push_back({input_example, input_example});
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
  auto l1 = Convolution2D(input, {5, 5}, 8, 2);
  auto l3 = Relu(l1);
  auto l4 = Convolution2D(l3, {5, 5}, 16, 2);
  auto l6 = Relu(l4);
  auto linear = Linear(l3, {512});
  auto relu = Relu(linear);
  auto drop = Dropout(relu, 0.4);
  auto l7 = Linear(drop, {10});
  auto output = Softmax(l7);

  training_set.resize(5000);
  Model model(input, output, training_set);
  Model tester(input, output, testing_set);

  float last_error = 1.0;
  for (int i = 1;; ++i) {
    model.batch_size = 23;
    model.Train(0.5f, training_set.size());

    float error_training = model.LastError();
    last_error = last_error * 0.95f + error_training * 0.05f;

    //float error_training = model.ErrorInteger();
    //float error_testing = tester.ErrorInteger();
    for (int i = 0; i < 100; ++i) {
      if (i < error_training * 100)
        std::cout << "#";
      else
        std::cout << "-";
    }
    std::cout << " " << error_training << " " << model.iteration << "/"
              << model.examples.size() << std::endl;

    std::ofstream("l1.ppm") << image_PPM(l1.params);
    std::ofstream("l1.pgm") << image_PGM(l1.params);
    std::ofstream("l1_output.ppm") << image_PPM(l1.output[0]);
    std::ofstream("l1_output.pgm") << image_PGM(l1.output[0]);
    std::ofstream("l3_output.ppm") << image_PPM(l3.output[0]);
    std::ofstream("l3_output.pgm") << image_PGM(l3.output[0]);
    std::ofstream("l4.ppm") << image_PPM(l1.params);

    if (error_training < 0.05)
      break;
  }


  EXPECT_LE(model.ErrorInteger(), 0.05);
  EXPECT_LE(tester.ErrorInteger(), 0.1);
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
  auto g_3 = Sigmoid(g_2);
  auto g_4 = Linear(g_3, {256});
  auto g_5 = Sigmoid(g_4);
  auto g_6 = Linear(g_5, {28, 28});
  auto g_o = Sigmoid(g_6);

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
          model.Train(0.001f, Node::T);
          for (auto& output : g_o.output) {
            generated.push_back(output);
          }
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
      GetExamples2(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples2(mnist.test_images, mnist.test_labels);

  // Generative network.
  auto g_i = Input({10});
  auto g_1 = Linear(g_i, {40});
  auto g_2 = Sigmoid(g_1);
  auto g_3 = Linear(g_2, {10,10,16});
  auto g_4 = Relu(g_3);
  auto g_5 = Deconvolution2D(g_4, {3, 3}, 14, 1);
  auto g_6 = Sigmoid(g_5);
  auto g_k = Dropout(g_6, 0.8);
  auto g_7 = Deconvolution2D(g_k, {5, 5}, 1, 2);
  auto g_o = Sigmoid(g_7);

  // Build a descriminative network.
  auto d_i = Input({27, 27});
  auto d_2 = Convolution2D(d_i, {5,5}, 7, 2);
  auto d_3 = Sigmoid(d_2);
  auto d_4 = Convolution2D(d_3, {3,3}, 14, 1);
  auto d_d = Dropout(d_4, 0.8);
  auto d_5 = Relu(d_d);
  auto d_6 = Linear(d_5, {40});
  auto d_7 = Sigmoid (d_6);
  auto d_8 = Linear(d_7, {1});
  auto d_o = Sigmoid (d_8);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::uniform_real_distribution<float> fake_value(0.0f, 0.1f);
  std::uniform_real_distribution<float> real_value(0.9f, 1.0f);

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

      if (!is_discriminative_network_loaded) {
        is_discriminative_network_loaded = true;
        Model(g_i, d_o).DeserializeParamsFromFile("network");
      } else {
        //Model(g_i, d_o).DeserializeParamsFromFile("network");
        Model(g_i, d_o).SerializeParamsToFile("network");
      }

      // Generate some examples.
      std::vector<Example> examples;
      for(size_t i = 0; i<10000; ++i) {
        auto input = Tensor::Random(d_i.params.sizes);
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.batch_size = 23;
      model.Train(0.f, 20);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      do {
        generated.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); i+=64) {
          model.Train(0.01f, 64);
          for(auto& output : g_o.output) {
            generated.push_back(output);
          }
          error += model.LastError();
        }
        generative_error = error /= examples.size();
        std::cerr << "generative_error = " << generative_error << std::endl;
        std::ofstream("other.pgm") << image_PGM(generated[0], 0.f, 1.f);
      } while (generative_error > 0.5f && iteration != 0);
      //} while (false);

      //Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");
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
        output.values = {fake_value(seed)};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < generated.size(); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }
      std::shuffle(examples.begin(), examples.end(), random_generator);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.batch_size = 23;
      model.Train(0.f, 20);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      do {
        model.Train(0.01, examples.size()/2);
        discriminative_error = model.LastError();
        std::cerr << "discriminative_error = " << discriminative_error
                  << std::endl;
      } while(discriminative_error > 0.2f);
    }

    std::cout << "error = " << generative_error << " vs " << discriminative_error << std::endl;
  }
}

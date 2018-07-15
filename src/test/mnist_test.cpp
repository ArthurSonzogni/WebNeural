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
#include "node/LeakyRelu.hpp"
#include "node/BilinearUpsampling.hpp"
#include "node/Sigmoid.hpp"
#include "node/Softmax.hpp"
#include "node/Dropout.hpp"
#include "node/BatchNormalization.hpp"
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
      GetExamples2(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples2(mnist.test_images, mnist.test_labels);

  // Generative network.
  auto g_i = Input({100});
  auto g_1 = Linear(g_i, {100});
  auto g_2 = Sigmoid(g_1);
  auto g_3 = Linear(g_2, {150});
  auto g_4 = Sigmoid(g_3);
  auto g_5 = Linear(g_4, {200});
  auto g_6 = Sigmoid(g_5);
  auto g_7 = Linear(g_6, {27, 27});
  auto g_o = Sigmoid(g_7);

  // Build a descriminative network.
  auto d_i = Input({27, 27});
  auto d_1 = Linear(d_i, {200});
  auto d_2 = Sigmoid(d_1);
  auto d_3 = Linear(d_2, {150});
  auto d_4 = Sigmoid(d_3);
  auto d_5 = Linear(d_4, {100});
  auto d_6 = Sigmoid(d_5);
  auto d_7 = Linear(d_6, {1});
  auto d_o = Sigmoid(d_7);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::uniform_real_distribution<float> fake_value(0.01f, 0.1f);
  std::uniform_real_distribution<float> real_value(0.9f, 0.99f);

  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    std::vector<Tensor> v_1;
    std::vector<Tensor> v_2;
    std::vector<Tensor> v_3;
    std::vector<Tensor> v_4;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    if (iteration != 0)
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
      for(size_t i = 0; i<64*10; ++i) {
        auto input = Tensor::SphericalRandom({5,2});
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.batch_size = 64;
      model.Train(0.f, 64);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      do {
        generated.clear();
        v_1.clear();
        v_2.clear();
        v_3.clear();
        v_4.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); i+=64) {
          model.Train(0.1f, 64);
          generated.insert(generated.end(), g_o.output.begin(),
                           g_o.output.end());
          v_1.insert(v_1.end(), g_1.output.begin(),
                           g_1.output.end());
          v_2.insert(v_2.end(), g_2.output.begin(),
                           g_2.output.end());
          v_3.insert(v_3.end(), g_3.output.begin(),
                           g_3.output.end());
          v_4.insert(v_4.end(), g_5.output.begin(),
                           g_5.output.end());
          error += model.LastError();
        }
        generative_error = 64 * error / examples.size();
        std::cerr << "generative_error = " << generative_error << std::endl;
        std::ofstream("other.pgm") << image_PGM(generated[0], 0.f, 1.f);
        std::cerr << "LINE = " << __LINE__;
      } while (generative_error > 0.25f);
      //} while (false);

      //Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");

        std::cerr << "LINE = " << __LINE__;
      //v_1.resize(3*3); std::ofstream("1.pgm") << image_PGM(Tensor::Merge(v_1));
      //v_2.resize(3*3); std::ofstream("2.pgm") << image_PGM(Tensor::Merge(v_2));
      //v_3.resize(3*3); std::ofstream("3.pgm") << image_PGM(Tensor::Merge(v_3));
      //v_4.resize(3*3); std::ofstream("4.pgm") << image_PGM(Tensor::Merge(v_4));
        std::cerr << "LINE = " << __LINE__;

      // Print a generated image
      auto preview = generated;
      preview.resize(6*6);
      std::ofstream("live.pgm") << image_PGM(Tensor::Merge(preview));

        std::cerr << "LINE = " << __LINE__;
      {
        std::cerr << "LINE = " << __LINE__;
        std::stringstream ss;
        ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
           << ".pgm";
        std::ofstream(ss.str()) << image_PGM(Tensor::Merge(preview), 0.f, 1.f);
        std::cerr << "LINE = " << __LINE__;
      }
        std::cerr << "LINE = " << __LINE__;
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
      for (size_t i = 0; i < generated.size() || i < 100; ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }
      std::shuffle(examples.begin(), examples.end(), random_generator);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.batch_size = 64;
      model.Train(0.f, 64);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      int iteration = 0;
      do {
        model.Train(0.01f, examples.size()/2);
        discriminative_error = model.LastError();
        std::cerr << "discriminative_error = " << discriminative_error
                  << std::endl;
      } while(discriminative_error > 0.05f && iteration++<2);
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
  auto g_i = Input({40});
  auto g_1 = Linear(g_i, {512});
  //auto b_1 = BatchNormalization(g_1);
  auto g_2 = LeakyRelu(g_1);
  auto g_3 = Linear(g_2, {9, 9, 32});
  auto b_2 = BatchNormalization(g_3);
  auto g_4 = LeakyRelu(b_2);
  auto g_5 = Deconvolution2D(g_4, {3, 3}, 16, 1);
  auto g_6 = LeakyRelu(g_5);
  auto g_7 = Deconvolution2D(g_6, {7, 7}, 1, 2);
  auto g_o = Sigmoid(g_7);

  // Build a descriminative network.
  auto d_i = Input({27, 27});
  auto d_2 = Convolution2D(d_i, {7, 7}, 16, 2);
  auto d_3 = LeakyRelu(d_2);
  auto d_4 = Convolution2D(d_3, {3, 3}, 32, 1);
  auto d_5 = LeakyRelu(d_4);
  auto d_6 = Linear(d_5, {64});
  auto d_7 = LeakyRelu(d_6);
  auto d_8 = Linear(d_7, {1});
  auto d_o = Sigmoid(d_8);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::uniform_real_distribution<float> fake_value(0.01f, 0.1f);
  std::uniform_real_distribution<float> real_value(0.9f, 0.99f);

  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    std::vector<Tensor> v_1;
    std::vector<Tensor> v_2;
    std::vector<Tensor> v_3;
    std::vector<Tensor> v_4;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    if (iteration != 0)
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
      for(size_t i = 0; i<64*10; ++i) {
        auto input = Tensor::SphericalRandom({5,2});
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.batch_size = 64;
      model.Train(0.f, 64);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      do {
        generated.clear();
        v_1.clear();
        v_2.clear();
        v_3.clear();
        v_4.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); i+=64) {
          model.Train(0.1f, 64);
          generated.insert(generated.end(), g_o.output.begin(),
                           g_o.output.end());
          v_1.insert(v_1.end(), g_1.output.begin(),
                           g_1.output.end());
          v_2.insert(v_2.end(), g_2.output.begin(),
                           g_2.output.end());
          v_3.insert(v_3.end(), g_3.output.begin(),
                           g_3.output.end());
          v_4.insert(v_4.end(), g_5.output.begin(),
                           g_5.output.end());
          error += model.LastError();
        }
        generative_error = 64 * error / examples.size();
        std::cerr << "generative_error = " << generative_error << std::endl;
        std::ofstream("other.pgm") << image_PGM(generated[0], 0.f, 1.f);
        std::cerr << "LINE = " << __LINE__;
      } while (generative_error > 0.25f);
      //} while (false);

      //Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");

        std::cerr << "LINE = " << __LINE__;
      //v_1.resize(3*3); std::ofstream("1.pgm") << image_PGM(Tensor::Merge(v_1));
      //v_2.resize(3*3); std::ofstream("2.pgm") << image_PGM(Tensor::Merge(v_2));
      //v_3.resize(3*3); std::ofstream("3.pgm") << image_PGM(Tensor::Merge(v_3));
      //v_4.resize(3*3); std::ofstream("4.pgm") << image_PGM(Tensor::Merge(v_4));
        std::cerr << "LINE = " << __LINE__;

      // Print a generated image
      auto preview = generated;
      preview.resize(6*6);
      std::ofstream("live.pgm") << image_PGM(Tensor::Merge(preview));

        std::cerr << "LINE = " << __LINE__;
      {
        std::cerr << "LINE = " << __LINE__;
        std::stringstream ss;
        ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
           << ".pgm";
        std::ofstream(ss.str()) << image_PGM(Tensor::Merge(preview), 0.f, 1.f);
        std::cerr << "LINE = " << __LINE__;
      }
        std::cerr << "LINE = " << __LINE__;
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
      for (size_t i = 0; i < generated.size() || i < 100; ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1,1,1});
        output.values = {real_value(seed)};
        examples.push_back({input, output});
      }
      std::shuffle(examples.begin(), examples.end(), random_generator);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.batch_size = 64;
      model.Train(0.f, 64);
      std::cerr << "pretrain error = " << model.LastError() << std::endl;
      int iteration = 0;
      do {
        model.Train(0.01f, examples.size()/2);
        discriminative_error = model.LastError();
        std::cerr << "discriminative_error = " << discriminative_error
                  << std::endl;
      } while(discriminative_error > 0.05f && iteration++<2);
    }

    std::cout << "error = " << generative_error << " vs " << discriminative_error << std::endl;
  }
}

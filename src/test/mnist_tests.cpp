#include <random>

#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({28, 28});
    input_example.values = input[i];
    for (auto& p : input_example.values) {
      p /= 256.0f;
    }

    auto output_example = Tensor({10, 1, 1});
    output_example.values[output[i]] = 1.f;

    examples.push_back({input_example, output_example});
  }

  return examples;
}

std::vector<Example> GetExamplesCentered(
    const std::vector<std::vector<float>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({28, 28});
    input_example.values = input[i];
    for (auto& p : input_example.values) {
      p /= 256.0f;
      p = 2.0 * p - 1;
    }

    auto output_example = Tensor({10, 1, 1});
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
    for (size_t y = 0; y < 27; ++y)
      for (size_t x = 0; x < 27; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y) = input[i][index] / 255.0f;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<Example> GetExamples2Centered(
    const std::vector<std::vector<float>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({27, 27, 1});
    for (size_t y = 0; y < 27; ++y)
      for (size_t x = 0; x < 27; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y) = input[i][index] / 255.0f * 2.0 - 1.0;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<Example> GetExamples3Centered(
    const std::vector<std::vector<float>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({29, 29, 1});
    for (size_t y = 0; y < 28; ++y)
      for (size_t x = 0; x < 28; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y) = input[i][index] / 256.f;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

TEST(MNIST, LinearSoftmax) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::shuffle(training_set.begin(), training_set.end(), random_generator);

  Allocator a;
  // Build a neural network.
  auto network = [&](Node* X) {
    X = a.Linear(X, {10});
    return X;
  };

  auto X = a.Input({28, 28, 1});
  auto Y = network(X);

  Model model(X, Y, training_set);
  Model tester(X, Y, testing_set);

  model.loss_function = LossFunction::SoftmaxCrossEntropy;
  tester.loss_function = LossFunction::SoftmaxCrossEntropy;

  for (size_t i = 0; i < 100; ++i) {
    model.Train(0.001f, 10000);
    std::cerr << '\r' << i << ") error = " << model.LastError() << std::flush;

    Tensor images({28, 28, 10});
    for (size_t x = 0; x < 28; ++x)
      for (size_t y = 0; y < 28; ++y)
        for (size_t t = 0; t < 10; ++t) {
          images.at(x, y, t) = X->next->params[x + 28 * y + (28 * 28 + 1) * t];
        }
    std::ofstream("live.ppm") << image_PGM(images);
  }

  std::cerr << std::endl;
  std::cerr << "error model  = " << model.ErrorInteger() << std::endl;
  std::cerr << "error tester = " << tester.ErrorInteger() << std::endl;

  EXPECT_LE(model.ErrorInteger(), 0.09);
  EXPECT_LE(tester.ErrorInteger(), 0.09);
}

TEST(MNIST, MultilayerPerceptron) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::shuffle(training_set.begin(), training_set.end(), random_generator);

  Allocator a;
  // Build a neural network.
  auto network = [&](Node* X) {
    X = a.LeakyRelu(a.Linear(X, {40}));
    X = a.LeakyRelu(a.Linear(X, {30}));
    X = a.LeakyRelu(a.Linear(X, {}));
    X = a.LeakyRelu(a.Linear(X, {10}));
    return X;
  };

  auto X = a.Input({28, 28, 1});
  auto Y = network(X);

  Model model(X, Y, training_set);
  Model tester(X, Y, testing_set);

  model.loss_function = LossFunction::SoftmaxCrossEntropy;
  tester.loss_function = LossFunction::SoftmaxCrossEntropy;

  for (size_t i = 0; i < 100; ++i) {
    model.Train(0.0005f, 10000);
    std::cerr << '\r' << i << ") error = " << model.LastError() << std::flush;
  }
  std::cerr << std::endl;
  std::cerr << "error model  = " << model.ErrorInteger() << std::endl;
  std::cerr << "error tester = " << tester.ErrorInteger() << std::endl;

  EXPECT_LE(model.ErrorInteger(), 0.03);
  EXPECT_LE(tester.ErrorInteger(), 0.03);
}

TEST(MNIST, CNN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  // training_set.resize(000);
  // testing_set.resize(5000);

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::shuffle(training_set.begin(), training_set.end(), random_generator);

  Allocator a;
  // Build a neural network.
  auto network = [&](Node* X) {
    X = a.LeakyRelu(a.Convolution2D(X, {7, 7}, 18, 2));
    X = a.LeakyRelu(a.Convolution2D(X, {7, 7}, 36, 2));  // 4x4
    X = a.LeakyRelu(a.Linear(X, {20}));
    X = a.LeakyRelu(a.Linear(X, {10}));
    return X;
  };

  auto X = a.Input({27, 27, 1});
  auto Y = network(X);

  Model model(X, Y, training_set);
  Model tester(X, Y, testing_set);

  model.loss_function = LossFunction::SoftmaxCrossEntropy;
  tester.loss_function = LossFunction::SoftmaxCrossEntropy;

  for (size_t i = 0; i < 100; ++i) {
    model.Train(0.001f, 1000);
    std::cerr << '\r' << i << ") error = " << model.LastError() << std::flush;

    std::ofstream("live.pgm") << image_PPM(X->next->params);
  }
  std::cerr << std::endl;
  std::cerr << "error model  = " << model.ErrorInteger() << std::endl;
  std::cerr << "error tester = " << tester.ErrorInteger() << std::endl;

  EXPECT_LE(model.ErrorInteger(), 0.04);
  EXPECT_LE(tester.ErrorInteger(), 0.04);
}

TEST(MNIST, WGAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);
  // float keep_probability = 0.;
  Allocator a;
  auto generator = [&](Node* X) {
    X = a.Linear(X, {10, 10, 1});
    X = a.LeakyRelu(X);
    X = a.BatchNormalization(X);

    X = a.Linear(X, {10, 10, 1});
    X = a.LeakyRelu(X);
    X = a.BatchNormalization(X);

    X = a.Linear(X, {28, 28, 1});
    X = a.Tanh(X);
    return X;
  };

  //(7-1)*2+5 = 17
  //(17-1)*2+5 = 37

  auto discriminator = [&](Node* X) {
    X = a.Deconvolution2D(X, {6, 6}, 8, 2);
    X = a.LeakyRelu(X);
    X = a.BatchNormalization(X);

    X = a.Linear(X, {10, 10, 1});
    X = a.LeakyRelu(X);
    X = a.BatchNormalization(X);

    X = a.Linear(X, {1, 1, 1});
    return X;
  };

  //(2-1)*2+6 = 8
  //(8-1)*2+6 = 20
  //(20-1)*2+6 = 54
  //
  //(2-1)*2+3 = 5
  //(5-1)*2+3 = 11
  //(11-1)*2+3 = 23
  //(23-1)*2+3 = 47
  //
  //(2-1)*2+5 = 7
  //(7-1)*2+5 = 17
  //(17-1)*2+5 = 37

  auto generator_input = a.Input({100});
  auto generator_output = generator(generator_input);

  auto discriminator_input = a.Input({28, 28, 1});
  auto discriminator_output = discriminator(discriminator_input);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());

  // Generate example references.
  std::vector<Example> examples_reference;
  for (size_t i = 0; i < 16; ++i) {
    auto input = Tensor::Random(generator_input->output[0].sizes);
    auto output = Tensor({1});
    output.values = {1.f};
    examples_reference.push_back({input, output});
  }

  std::vector<Example> examples;
  for (size_t iteration = 1; iteration < 1000000; ++iteration) {
    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(generator_output, discriminator_input->next);

      // Do not train on the discriminative network.
      Range(discriminator_input, discriminator_output).Apply(&Node::Lock);

      if (!is_discriminative_network_loaded) {
        is_discriminative_network_loaded = true;
        Model(generator_input, discriminator_output)
            .DeserializeParamsFromFile("network");
      } else {
        Model(generator_input, discriminator_output)
            .SerializeParamsToFile("network");
      }

      // Generate some examples.
      std::vector<Example> examples;
      for (size_t i = 0; i < Node::T; ++i) {
        auto input = Tensor::Random(generator_input->output[0].sizes);
        auto output = Tensor({1});
        output.values = {1.f};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(generator_input, discriminator_output, examples);
      model.loss_function = LossFunction::WasserStein;

      // generated.clear();
      float error = 0.f;
      for (size_t i = 0; i < examples.size(); i += Node::T) {
        // std::cerr << " i = " << i << '\r' << std::flush;
        model.Train(0.0005f, examples.size());
        generated.insert(generated.end(), generator_output->output.begin(),
                         generator_output->output.end());
        error += model.LastError();
      }
      generative_error = Node::T * error / examples.size();
      // std::cerr << "generative_error = " << generative_error << std::endl;
      //} while (iteration != 0 && (++i<5 || generative_error > 0.5));

      // Model(generator_input,
      // generator_output).SerializeParamsToFile("generative_conv_net");
      // Print a generated image

      // Preview ---
      {
        // Train the network.
        auto model =
            Model(generator_input, discriminator_output, examples_reference);
        model.loss_function = LossFunction::WasserStein;

        model.Train(0.f, examples_reference.size());
        generated.insert(generated.end(), generator_output->output.begin(),
                         generator_output->output.end());

        auto preview = generator_output->output;
        preview.resize(examples_reference.size());
        std::ofstream("live.pgm")
            << image_PGM(Tensor::Merge(preview), -1.0, 1.0);
      }
    }

    // Train the discriminative network.
    // examples.resize(examples.size() * 0.9);
    examples.resize(0);
    {
      // Disconnect the two networks.
      Node::Link(discriminator_input, discriminator_input->next);
      Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);
      Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

      // Generate some examples.
      for (auto& input : generated) {
        auto output = Tensor({1});
        output.values = {-1.f};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < std::min(generated.size(), size_t(Node::T)); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1});
        output.values = {1.f};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(discriminator_input, discriminator_output, examples);
      model.loss_function = LossFunction::WasserStein;
      model.post_update_function = PostUpdateFunction::ClipWeight(
          discriminator_input, discriminator_output);

      std::shuffle(examples.begin(), examples.end(), random_generator);
      model.Train(0.001f, examples.size());
      discriminative_error = model.LastError();
    }

    std::cout << "g_error = " << generative_error << " | d_error "
              << discriminative_error << " epoch = "
              << (iteration * float(Node::T) / training_set.size())
              << std::endl;
  }
}

TEST(MNIST, WCGAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples3Centered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples3Centered(mnist.test_images, mnist.test_labels);

  // float keep_probability = 0.;
  Allocator a;
  auto generator = [&](Node* X) {
    X = a.Linear(X, {4, 4, 32});
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 16, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 16, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 1, 1);
    X = a.Sigmoid(X);
    return X;
  };

  auto discriminator = [&](Node* X) {
    X = a.Convolution2D(X, {5, 5}, 16, 1);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {5, 5}, 16, 2);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {5, 5}, 32, 2);
    X = a.LeakyRelu(X);

    X = a.Linear(X, {1, 1, 1});
    return X;
  };

  auto generator_input = a.Input({20});
  auto generator_output = generator(generator_input);

  auto discriminator_input = a.Input({29, 29, 1});
  auto discriminator_output = discriminator(discriminator_input);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());

  // Generate example references.
  std::vector<Example> examples_reference;
  for (size_t i = 0; i < 16*9; ++i) {
    auto input = Tensor::Random(generator_input->output[0].sizes);
    auto output = Tensor({1});
    output.values = {1.0};
    examples_reference.push_back({input, output});
  }

  std::vector<Example> examples;
  for (size_t iteration = 1; iteration < 10000; ++iteration) {
    float learning_rate = 0.003f / std::pow(iteration, 1.0 / 7.0);

    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(generator_output, discriminator_input->next);

      // Do not train on the discriminative network.
      Range(discriminator_input, discriminator_output).Apply(&Node::Lock);
      Range(generator_input, discriminator_output).Apply(&Node::Clear);

      if (!is_discriminative_network_loaded) {
        is_discriminative_network_loaded = true;
        Model(generator_input, discriminator_output)
            .DeserializeParamsFromFile("network");
      } else {
        Model(generator_input, discriminator_output)
            .SerializeParamsToFile("network");
      }

      // Generate some examples.
      std::vector<Example> examples;
      for (size_t i = 0; i < 16; ++i) {
        auto input = Tensor::Random(generator_input->output[0].sizes);
        auto output = Tensor({1});
        output.values = {1.0};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(generator_input, discriminator_output, examples);
      model.loss_function = LossFunction::WasserStein;
      // model.post_update_function = PostUpdateFunction::ClipWeight;

      // generated.clear();
      float error = 0.f;
      for (size_t i = 0; i < examples.size(); i += Node::T) {
        // std::cerr << " i = " << i << '\r' << std::flush;
        model.Train(learning_rate, examples.size());
        generated.insert(generated.end(), generator_output->output.begin(),
                         generator_output->output.end());
        error += model.LastError();
      }
      generative_error = Node::T * error / examples.size();
      // std::cerr << "generative_error = " << generative_error << std::endl;
      //} while (iteration != 0 && (++i<5 || generative_error > 0.5));

      // Model(generator_input,
      // generator_output).SerializeParamsToFile("generative_conv_net");
      // Print a generated image

      // Preview ---
      //if (i++%10 == 0)
      {
        // Train the network.
        auto model =
            Model(generator_input, discriminator_output, examples_reference);
        model.loss_function = LossFunction::WasserStein;
        // model.post_update_function =
        // PostUpdateFunction::ClipWeight(discriminator_input,
        // discriminator_output);

        std::vector<Tensor> preview;
        for (size_t i = 0; i < examples_reference.size(); i += Node::T) {
          model.Train(0.f, examples.size());
          preview.insert(preview.end(), generator_output->output.begin(),
                         generator_output->output.end());
        }
        if (preview.size() > examples_reference.size())
          preview.resize(examples_reference.size());
        std::stringstream number;
        static int i = 0;
        number << std::setw(6) << std::setfill('0') << i++;
        std::ofstream("live_" + number.str() + ".pgm")
            << image_PGM(Tensor::Merge(preview, 16), 0.0, 1.0);
        std::ofstream("live.pgm")
            << image_PGM(Tensor::Merge(preview, 16), 0.0, 1.0);
        std::ofstream("params.pgm")
            << image_PGM(discriminator_input->next->params);
      }
    }

    // Train the discriminative network.
    // examples.resize(examples.size() * 0.9);
    examples.resize(0);
    {
      // Disconnect the two networks.
      Node::Link(discriminator_input, discriminator_input->next);
      Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);
      Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

      // Generate some examples.
      for (auto& input : generated) {
        auto output = Tensor({1});
        output.values = {-1.f};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < std::min(generated.size(), size_t(Node::T)); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1});
        output.values = {1.0};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(discriminator_input, discriminator_output, examples);
      model.loss_function = LossFunction::WasserStein;
      model.post_update_function = PostUpdateFunction::ClipWeight(
          discriminator_input->next, discriminator_output);

      // std::shuffle(examples.begin(), examples.end(), random_generator);
      model.Train(learning_rate * 5.f, examples.size());
      discriminative_error = model.LastError();
    }

    std::cout << "| g_error = " << std::setw(10) << generative_error
              << "| d_error " << std::setw(10) << discriminative_error
              << "| epoch = "
              << (iteration * float(Node::T) / training_set.size())
              << std::endl;
  }
}

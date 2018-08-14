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

    auto output_example = Tensor({10,1,1});
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
    for(size_t y = 0; y<27; ++y)
    for(size_t x = 0; x<27; ++x) {
      size_t index = x + 28 *y;
      input_example.at(x,y) = input[i][index] / 255.0f;
    }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<Example> GetExamples2Centered(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({27, 27, 1});
    for(size_t y = 0; y<27; ++y)
    for(size_t x = 0; x<27; ++x) {
      size_t index = x + 28 *y;
      input_example.at(x,y) = input[i][index] / 255.0f * 2.0 - 1.0;
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

  for(size_t i = 0; i<100; ++i) {
    model.Train(0.001f, 10000);
    std::cerr <<  '\r' << i << ") error = " << model.LastError() << std::flush;

    Tensor images({28,28,10});
    for(size_t x = 0; x<28; ++x)
    for(size_t y = 0; y<28; ++y)
    for(size_t t = 0; t<10; ++t) {
      images.at(x,y,t) = X->next->params[x + 28*y + (28*28+1)*t];
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
    X = a.LeakyRelu(a.Linear(X, {20}));
    X = a.LeakyRelu(a.Linear(X, {10}));
    return X;
  };

  auto X = a.Input({28, 28, 1});
  auto Y = network(X);

  Model model(X, Y, training_set);
  Model tester(X, Y, testing_set);

  model.loss_function = LossFunction::SoftmaxCrossEntropy;
  tester.loss_function = LossFunction::SoftmaxCrossEntropy;

  for(size_t i = 0; i<100; ++i) {
    model.Train(0.0005f, 10000);
    std::cerr <<  '\r' << i << ") error = " << model.LastError() << std::flush;
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

  //training_set.resize(000);
  //testing_set.resize(5000);

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

  for(size_t i = 0; i<100; ++i) {
    model.Train(0.001f, 1000);
    std::cerr <<  '\r' << i << ") error = " << model.LastError() << std::flush;

    std::ofstream("live.pgm") << image_PPM(X->next->params);
  }
  std::cerr << std::endl;
  std::cerr << "error model  = " << model.ErrorInteger() << std::endl;
  std::cerr << "error tester = " << tester.ErrorInteger() << std::endl;

  EXPECT_LE(model.ErrorInteger(), 0.04);
  EXPECT_LE(tester.ErrorInteger(), 0.04);
}

TEST(MNIST, WCGAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples2(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamples2(mnist.test_images, mnist.test_labels);

  const size_t features = 16;
  const float keep_probability = 0.9;

  Allocator a;
  auto generator = [&](Node* X) {

    auto layer = [&](Node* X, size_t stride) {
      X = a.Deconvolution2D(X, {5, 5}, features, stride);
      X = a.LeakyRelu(X);
      X = a.Dropout(X, keep_probability);
      return X;
    };

    // Intermediate layers.
    X = layer(X, 2); // 6 -> 15
    X = layer(X, 1); // 15 -> 19
    X = layer(X, 1); // 19 -> 23

    // Final layer.
    X = a.Deconvolution2D(X, {5, 5}, 1, 1); // 23 -> 27
    X = a.Sigmoid(X);

    return X;
  };

  auto discriminator = [&](Node* X) {
    auto layer = [&](Node* X, size_t stride) {
      X = a.Convolution2D(X, {5, 5}, features, stride);
      X = a.LeakyRelu(X);
      X = a.Dropout(X, keep_probability);
      return X;
    };

    // Intermediate layers.
    X = layer(X, 2);
    X = layer(X, 1);
    X = layer(X, 1); 

    // Final layer.
    X = a.Linear(X, {32});
    X = a.LeakyRelu(X);
    X = a.Linear(X, {1});
    return X;
  };


  auto g_i = a.Input({15, 15, 3});
  auto g_o = generator(g_i);

  auto d_i = a.Input({27, 27, 1});
  auto d_o = discriminator(d_i);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::uniform_real_distribution<float> real_value(0.0f, 0.1f);
  std::uniform_real_distribution<float> fake_value(0.9f, 1.0f);

  std::vector<Example> examples;
  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(g_o, d_i->next);

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
      for(size_t i = 0; i<64; ++i) {
        //auto input = Tensor::SphericalRandom(g_i.output[0].sizes);
        auto input = Tensor::Random(g_i->output[0].sizes);
        auto output = Tensor({1});
        float p = real_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.loss_function = LossFunction::WasserStein;
      model.batch_size = 64;
      model.Train(0.f, 64);
      //std::cerr << "pretrain error = " << model.LastError() << std::endl;
      //size_t i = 0;
      do {
        generated.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); i+=64) {
          //std::cerr << " i = " << i << '\r' << std::flush;
          model.Train(0.001f, 64);
          generated.insert(generated.end(), g_o->output.begin(),
                           g_o->output.end());
          error += model.LastError();
        }
        generative_error = 64 * error / examples.size();
        //std::cerr << "generative_error = " << generative_error << std::endl;
      //} while (iteration != 0 && (++i<5 || generative_error > 0.5));
      } while(false);

      //Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");
      //v_1.resize(3*3); std::ofstream("1.pgm") << image_PGM(Tensor::Merge(v_1));
      //v_2.resize(3*3); std::ofstream("2.pgm") << image_PGM(Tensor::Merge(v_2));
      //v_3.resize(3*3); std::ofstream("3.pgm") << image_PGM(Tensor::Merge(v_3));
      //v_4.resize(3*3); std::ofstream("4.pgm") << image_PGM(Tensor::Merge(v_4));

      // Print a generated image
      auto preview = generated;
      preview.resize(10);
      std::ofstream("live.pgm") << image_PGM(Tensor::Merge(preview));

      {
        std::stringstream ss;
        ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
           << ".pgm";
        std::ofstream(ss.str()) << image_PGM(Tensor::Merge(preview), 0.f, 1.f);
      }
    }

    // Train the discriminative network.
    examples.resize(examples.size() * 0.7);
    examples.resize(0);
    {
      // Disconnect the two networks.
      Node::Link(d_i, d_i->next);
      Range(d_i, d_o).Apply(&Node::Unlock);
      Range(d_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      for (auto& input : generated) {
        auto output = Tensor({1});
        float p = fake_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < std::min(generated.size(), size_t(64)); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1});
        float p = real_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.loss_function = LossFunction::WasserStein;
      model.batch_size = 64;
      model.Train(0.f, 64);
      //std::cerr << "pretrain error = " << model.LastError() << std::endl;
      for(int i = 0; i<5; ++i) {
        std::cout << "i = " << i << '\r' << std::flush;
      //do {
        std::shuffle(examples.begin(), examples.end(), random_generator);
        model.Train(0.002f, 64);
        discriminative_error = model.LastError();
        //std::cerr << "discriminative_error = " << discriminative_error
                  //<< std::endl;
      //} while(discriminative_error > 2.0);
      //} while (false);
      }
    }

    std::cout << "g_error = " << generative_error << " | d_error " << discriminative_error << std::endl;
  }
}


TEST(MNIST, WGAN) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> testing_set =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  Allocator a;
  auto generator = [&](Node* X) {
    X = a.LeakyRelu(a.Linear(X, {100}));
    X = a.LeakyRelu(a.Linear(X, {200}));
    X = a.LeakyRelu(a.Linear(X, {28, 28, 1}));
    return X;
  };

  auto discriminator = [&](Node* X) {
    X = a.LeakyRelu(a.Linear(X, {200}));
    X = a.LeakyRelu(a.Linear(X, {100}));
    X = a.LeakyRelu(a.Linear(X, {10}));
    X = a.Linear(X, {1});
    return X;
  };

  auto g_i = a.Input({10});
  auto g_o = generator(g_i);

  auto d_i = a.Input({28, 28, 1});
  auto d_o = discriminator(d_i);

  bool is_discriminative_network_loaded = false;

  std::random_device seed;
  std::mt19937 random_generator(seed());
  std::uniform_real_distribution<float> real_value(0.0f, 0.1f);
  std::uniform_real_distribution<float> fake_value(0.9f, 1.0f);

  std::vector<Example> examples;
  for(size_t iteration = 0; iteration < 1000; ++iteration) {

    std::vector<Tensor> generated;
    float generative_error = 0.f;
    float discriminative_error = 0.f;

    // Train the generative network.
    {
      // Connect the two network.
      Node::Link(g_o, d_i->next);

      // Do not train on the discriminative network.
      Range(d_i, d_o).Apply(&Node::Lock);
      Range(g_i, d_o).Apply(&Node::Clear);

      if (!is_discriminative_network_loaded) {
        is_discriminative_network_loaded = true;
        Model(g_i, d_o).DeserializeParamsFromFile("network");
      } else {
        Model(g_i, d_o).SerializeParamsToFile("network");
      }

      // Generate some examples.
      std::vector<Example> examples;
      for(size_t i = 0; i<64; ++i) {
        //auto input = Tensor::SphericalRandom(g_i.output[0].sizes);
        auto input = Tensor::Random(g_i->output[0].sizes);
        auto output = Tensor({1});
        float p = real_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }

      // Train the network.
      auto model = Model(g_i, d_o, examples);
      model.loss_function = LossFunction::WasserStein;
      model.batch_size = 64;
      //model.Train(0.f, 64);
      //std::cerr << "pretrain error = " << model.LastError() << std::endl;
      //size_t i = 0;
      do {
        //generated.clear();
        float error = 0.f;
        for (size_t i = 0; i < examples.size(); i+=64) {
          //std::cerr << " i = " << i << '\r' << std::flush;
          model.Train(0.001f, 64);
          generated.insert(generated.end(), g_o->output.begin(),
                           g_o->output.end());
          error += model.LastError();
        }
        generative_error = 64 * error / examples.size();
        //std::cerr << "generative_error = " << generative_error << std::endl;
      //} while (iteration != 0 && (++i<5 || generative_error > 0.5));
      } while(false);

      //Model(g_i, g_o).SerializeParamsToFile("generative_conv_net");
      // Print a generated image
      auto preview = generated;
      preview.resize(10);
      if (iteration % 40 == 0)
      std::ofstream("live.pgm") << image_PGM(Tensor::Merge(preview));

      //if (iteration % 10 == 0)
      //{
        //std::stringstream ss;
        //ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
           //<< ".pgm";
        //std::ofstream(ss.str()) << image_PGM(Tensor::Merge(preview), 0.f, 1.f);
      //}
    }

    // Train the discriminative network.
    //examples.resize(examples.size() * 0.9);
    examples.resize(0);
    {
      // Disconnect the two networks.
      Node::Link(d_i, d_i->next);
      Range(d_i, d_o).Apply(&Node::Unlock);
      Range(d_i, d_o).Apply(&Node::Clear);

      // Generate some examples.
      for (auto& input : generated) {
        auto output = Tensor({1});
        float p = fake_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }
      std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
      for (size_t i = 0; i < std::min(generated.size(), size_t(64)); ++i) {
        auto input = training_set[random_index(random_generator)].input;
        auto output = Tensor({1});
        float p = real_value(seed);
        output.values = {p};
        examples.push_back({input, output});
      }

      std::vector<Tensor> preview;
      for(auto& i : examples)
        preview.push_back(i.input);
      //std::stringstream ss;
      //ss << "generated_" << std::setw(4) << std::setfill('0') << iteration
         //<< ".pgm";
      std::ofstream("coucou.pgm") << image_PGM(Tensor::Merge(preview), 0.f, 1.f);

      // Train the network.
      auto model = Model(d_i, d_o, examples);
      model.loss_function = LossFunction::WasserStein;
      model.batch_size = 64;
      //model.Train(0.f, 64);
      //std::cerr << "pretrain error = " << model.LastError() << std::endl;
      for(int i = 0; i<50; ++i) {
        //std::cout << "i = " << i << '\r' << std::flush;
      //do {
        std::shuffle(examples.begin(), examples.end(), random_generator);
        model.Train(0.0001f, 64);
        discriminative_error = model.LastError();
        //std::cerr << "discriminative_error = " << discriminative_error
                  //<< std::endl;
      //} while(discriminative_error > 2.0);
      //} while (false);
      }
    }

    std::cout << "g_error = " << generative_error << " | d_error " << discriminative_error << std::endl;
  }
}

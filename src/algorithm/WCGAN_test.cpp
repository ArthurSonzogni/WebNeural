#include "Image.hpp"
#include "Model.hpp"
#include "algorithm/WCGAN.hpp"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"

#include "cifar/cifar10_reader.hpp"
auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

#include "../third_party/stb_image/header.hpp"
#include <sstream>

std::vector<Example> GetFlower() {
  std::vector<Example> examples;
  for (int i = 721; i < 800; ++i) {
    int width, height, channels;

    std::stringstream filename;
    filename << "./flower/image_" << std::setfill('0') << std::setw(4) << i
             << ".jpg.png";
    //stbi_set_flip_vertically_on_load(true);
    unsigned char* image =
        stbi_load(filename.str().c_str(), &width, &height, &channels, STBI_rgb);
    std::cout << width << " " << height << " " << channels << std::endl;
    Tensor input(
        std::vector<size_t>({(size_t)width, (size_t)height, (size_t)3}));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < 3; ++c) {
          input.at(x, y, c) = image[c + 3 * (x + width * y)] / 256.0;
        }
      }
    }
    stbi_image_free(image);

    Tensor output(1);
    output[0] = i;

    examples.push_back({input, output});
  }
  return examples;
}

std::vector<Example> GetMNIST(
    const std::vector<std::vector<float>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({29, 29, 1});
    for (size_t y = 0; y < 28; ++y)
      for (size_t x = 0; x < 28; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y, 0) = input[i][index] / 256.f;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<Example> GetCIFAR(
    const std::vector<std::vector<unsigned char>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({32, 32, 3});
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = 0; y < 32; ++y) {
        for (size_t x = 0; x < 32; ++x) {
          size_t index = x + 32 * (y + 32 * (c));
          input_example.at(x, y, c) = input[i][index] / 256.f;
        }
      }
    }

    auto output_example = Tensor({1});
    output_example.values[0] = output[i];

    examples.push_back({input_example, output_example});
  }

  return examples;
}


TEST(WCGAN, ExampleMNIST) {
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetMNIST(mnist.training_images, mnist.training_labels);

  WCGAN wcgan;

  wcgan.latent_size = {20};

  wcgan.generator = [](Allocator& a, Node* X) {
    X = a.Linear(X, {4, 4, 64});
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 64, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 64, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 3, 1);
    X = a.Sigmoid(X);
    return X;
  };

  int i = 0;
  wcgan.input = [&]() {
    i = (i+1) % training_set.size();
    return training_set[i].input;
  };

  wcgan.discriminator = [](Allocator& a, Node* X) {
    X = a.Convolution2D(X, {5, 5}, 64, 1);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {5, 5}, 64, 2);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {5, 5}, 64, 2);
    X = a.LeakyRelu(X);

    X = a.Linear(X, {1, 1, 1});
    return X;
  };

  wcgan.Init();

  wcgan.LoadFromFile("network");
  for(int j = 1; j<10000; ++j) {
    wcgan.learning_rate = 0.003f / std::pow(j, 1.0/7.0);
    wcgan.batch_size = 16;
    wcgan.Train();

    std::ofstream("live.pgm") << image_PGM(wcgan.Generate(), 0.0, 1.0);
    wcgan.SaveToFile("network");
  }
}

TEST(WCGAN, ExampleCIFAR) {
  auto cifar = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::vector<Example> training_set =
      GetCIFAR(cifar.training_images, cifar.training_labels);

  WCGAN wcgan;

  wcgan.latent_size = {20};

  wcgan.generator = [](Allocator& a, Node* X) {
    X = a.Linear(X, {6, 6, 64});
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {3, 3}, 32, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {4, 4}, 24, 2);
    X = a.BatchNormalization(X);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {5, 5}, 3, 1);
    X = a.Sigmoid(X);
    return X;
  };

  int i = 0;
  wcgan.input = [&]() {
    while(true) {
      i = (i+1) % training_set.size();
      constexpr float selected_cat = 5;
      float cat = training_set[i].output[0] - selected_cat;
      if (cat*cat < 0.5)
        break;
    }

    return training_set[i].input;
  };

  wcgan.discriminator = [](Allocator& a, Node* X) {
    X = a.Convolution2D(X, {5, 5}, 24, 1);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {4, 4}, 32, 2);
    X = a.BatchNormalization(X);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {3, 3}, 64, 2);
    X = a.BatchNormalization(X);
    X = a.LeakyRelu(X);

    X = a.Linear(X, {1, 1, 1});
    return X;
  };

  wcgan.Init();

  wcgan.LoadFromFile("network");
  for(int j = 1; j<10000; ++j) {
    wcgan.learning_rate = 0.0001f / std::pow(j, 1.0/7.0);
    wcgan.batch_size = 16;
    for(int i = 0; i<2; ++i) {
      wcgan.Train();
    }

    std::vector<Tensor> img;
    for(int i = 0; i < 10; ++i) {
      img.push_back(wcgan.input());
      img.push_back(wcgan.Generate());
    }
    std::ofstream("live.pgm") << image_PPM(Tensor::Merge(img), 0.0, 1.0);
    //std::ofstream("live.pgm") << image_PPM(wcgan.input(), 0.f, 1.0);
    wcgan.SaveToFile("network");
  }
}

TEST(WCGAN, ExampleFlower) {
  auto cifar = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::vector<Example> training_set = GetFlower();

  WCGAN wcgan;

  wcgan.latent_size = {20};

  wcgan.generator = [](Allocator& a, Node* X) {
    X = a.Linear(X, {6, 6, 4});
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {4, 4}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {4, 4}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {3, 3}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Deconvolution2D(X, {8, 8}, 3, 2);
    X = a.Sigmoid(X);
    return X;
  };

  int i = 0;
  wcgan.input = [&]() {
    i = (i+1) % training_set.size();
    return training_set[i].input;
  };

  // (8-4)/2+1=15
  // (15-3)/2+1=7
  // (7-3)/2+1=3
  //(128-6)/2 + 1 = 62;
  //(62-4)/2+1 = 30
  //(30-4)/2+1 = 14
  //(14-4)/2+1 = 6
  //(6-4)/2+1 = 3
  
  //32-5+1 = 28
  //(28-4)/2+1=13
  //(13-3)/2+1=6
  
  //(128-8)/2+1 = 61
  //(61-3)/2+1= 30
  //(30-4)/2+1=14
  //(14-4)/2+1=6

  wcgan.discriminator = [](Allocator& a, Node* X) {
    X = a.Convolution2D(X, {8, 8}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {3, 3}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {4, 4}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Convolution2D(X, {4, 4}, 4, 2);
    X = a.LeakyRelu(X);

    X = a.Linear(X, {1, 1, 1});
    return X;
  };

  wcgan.Init();

  wcgan.LoadFromFile("network");
  for(int j = 1; j<10000; ++j) {
    wcgan.learning_rate = 0.001f / std::pow(j, 1.0/7.0);
    wcgan.batch_size = 64;
    wcgan.Train();

    std::vector<Tensor> img;
    for(int i = 0; i < 10; ++i) {
      img.push_back(wcgan.input());
      img.push_back(wcgan.Generate());
    }
    std::ofstream("live.ppm") << image_PPM(Tensor::Merge(img), 0.0, 1.0);
    wcgan.SaveToFile("network");
  }
}

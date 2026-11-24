#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input) {
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

std::vector<uint8_t> input_js(27 * 27 * 4);
std::vector<uint8_t> output_js(27 * 27 * 4);

auto mnist_images =
    mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
        MNIST_DATA_LOCATION);
std::vector<Example> training_set =
#ifndef Web
    GetExamples(mnist_images.training_images);
#else
    GetExamples(mnist_images.test_images);
#endif

Allocator a;
auto compress = [](Node* X) {
  X = a.Convolution2D(X, {5, 5}, 12, 2);  // 12x12x12
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {4, 4}, 24, 2);  // 5x5x24
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {3, 3}, 48, 2);  // 2x2x48
  X = a.LeakyRelu(X);

  X = a.Linear(X, {10});
  X = a.Sigmoid(X);
  return X;
};

auto decompress = [](Node* X) {
  X = a.Noise(X, 0.02f);                  // Latent regularization (training only)
  X = a.Linear(X, {2, 2, 48});
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {3, 3}, 24, 2);  // 5x5x24
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {4, 4}, 12, 2);  // 12x12x12
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {5, 5}, 1, 2);   // 27x27x1
  X = a.Sigmoid(X);
  return X;
};

auto input = a.Input({27, 27, 1});
auto latent = compress(input);
auto output = decompress(latent);

Model model(input, output, training_set);

void Train(float lambda, int iterations = 10) {
  model.loss_function = LossFunction::BinaryCrossEntropy;
  model.Train(lambda, iterations);
}

void Import(Tensor& tensor, double* input) {
  for (auto& it : tensor.values) {
    it = *(input++);
  }
}

void Export(const Tensor& tensor, uint8_t* output) {
  for (auto& it : tensor.values) {
    float v = std::max(0.f, std::min(1.f, it));
    (*output++) = 255 * v;
    (*output++) = 255 * v;
    (*output++) = 255 * v;
    (*output++) = 255;
  }
}

void Save() { model.SerializeParamsToFile("save.bin"); }
void Load() { model.DeserializeParamsFromFile("save.bin"); }

extern "C" {

float lambda = 0.003f;
void Train() {
  lambda = std::max(lambda * 0.999f, 0.0001f);
  Train(lambda, 13);
}

void LastInput(uint8_t* _input) {
  Export(input->output[0], _input);
}

void LastOutput(uint8_t* _output) {
  Export(output->output[0], _output);
}

void Predict(double* _input, uint8_t* _output) {
  Import(latent->output[0], _input);
  Range(latent->next, output).Apply([](Node* node) {
    node->training = false;
    node->Forward(1);
  });
  Export(output->output[0], _output);
}

void LoadPretrainedModel() {
  lambda = 0.0001f;
  Load();
}

};

#ifndef Web
int main(int argc, const char* argv[]) {
  bool scratch = false;
  bool preview = false;
  int epochs = 10;
  float lr = 0.002f;
  size_t batch = 16;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--scratch") {
      scratch = true;
    } else if (arg == "--preview") {
      preview = true;
    } else if (arg == "--epochs" && i + 1 < argc) {
      epochs = std::stoi(argv[++i]);
    } else if (arg == "--lr" && i + 1 < argc) {
      lr = std::stof(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      batch = static_cast<size_t>(std::stoul(argv[++i]));
    }
  }

  model.batch_size = batch;
  if (!scratch) {
    Load();
  }

  if (preview) {
    auto test_set = GetExamples(mnist_images.test_images);
    const size_t eval_count = std::min<size_t>(test_set.size(), 1000);
    float total_error = 0.f;
    std::vector<Tensor> pairs;
    for (size_t i = 0; i < eval_count; ++i) {
      Tensor reconstructed = model.Predict(test_set[i].input);
      total_error += (reconstructed - test_set[i].output).Error();
      if (i < 64) {
        pairs.push_back(
            Tensor::ConcatenateHorizontal(test_set[i].input, reconstructed));
      }
    }
    std::cout << "test_error (" << eval_count
              << " images) = " << (total_error / eval_count) << std::endl;
    std::ofstream("preview.pgm") << image_PGM(Tensor::Merge(pairs, 8), 0.f, 1.f);
    std::cout << "Wrote preview.pgm (left: input, right: reconstruction)"
              << std::endl;
    return 0;
  }

  std::mt19937 rng(42);
  const size_t steps_per_epoch = model.examples.size();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    std::shuffle(model.examples.begin(), model.examples.end(), rng);
    Train(lr, static_cast<int>(steps_per_epoch));
    std::cout << "epoch " << epoch << "/" << epochs
              << " | lr=" << lr
              << " | bce=" << model.LastError() << std::endl;
    Save();
    lr = std::max(lr * 0.9f, 0.0001f);
  }
  return 0;
}
#endif

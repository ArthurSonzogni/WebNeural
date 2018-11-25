#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

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
  X = a.Convolution2D(X, {7, 7}, 6, 2);
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {3, 3}, 16, 2);
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {3, 3}, 32, 2);
  X = a.LeakyRelu(X);

  X = a.Linear(X, {10});
  X = a.Sigmoid(X);
  return X;
};

auto decompress = [](Node* X) {
  X = a.Linear(X, {2, 2, 32});
  X = a.Deconvolution2D(X, {3, 3}, 16, 2);

  X = a.LeakyRelu(X);
  X = a.Deconvolution2D(X, {3, 3}, 6, 2);

  X = a.LeakyRelu(X);
  X = a.Deconvolution2D(X, {7, 7}, 1, 2);

  X = a.Sigmoid(X);
  return X;
};

auto input = a.Input({27, 27, 1});
auto latent = compress(input);
auto output = decompress(latent);

Model model(input, output, training_set);

void Train(float lambda, int iterations = 10) {
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
  Load();
  Train(0.001f, 640);
  std::cout << "error " << model.LastError() << std::endl;
  Save();
  return 0;
}
#endif

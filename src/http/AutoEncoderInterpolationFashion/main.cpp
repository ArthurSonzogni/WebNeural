#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
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

std::vector<uint8_t> input_js(27 * 27 * 4);
std::vector<uint8_t> output_js(27 * 27 * 4);

auto mnist_images =
    mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
        MNIST_DATA_LOCATION);
std::vector<Example> training_set =
#ifndef Web
    GetExamples(mnist_images.training_images, mnist_images.training_labels);
#else
    GetExamples(mnist_images.test_images, mnist_images.test_labels);
#endif

class Demo {
 public:
  Demo() {
    auto compress = [&](Node* X) {
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

    auto decompress = [&](Node* X) {
      X = a.Linear(X, {2, 2, 32});
      X = a.Deconvolution2D(X, {3, 3}, 16, 2);
      X = a.LeakyRelu(X);
      X = a.Deconvolution2D(X, {3, 3}, 6, 2);
      X = a.LeakyRelu(X);
      X = a.Deconvolution2D(X, {7, 7}, 1, 2);
      X = a.Sigmoid(X);
      return X;
    };

    input = a.Input({27, 27, 1});
    latent = compress(input);
    output = decompress(latent);

    model = std::make_unique<Model>(input, output, training_set);
  }

  void Train(float lambda, int iterations = 10) {
    model->Train(lambda, iterations);
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

  void Save() { model->SerializeParamsToFile("save.bin"); }
  void Load() { model->DeserializeParamsFromFile("save.bin"); }

  Allocator a;
  Node* input;
  Node* latent;
  Node* output;
  std::unique_ptr<Model> model;
};

Demo demo;

extern "C" {

float lambda = 0.003f;
void Train() {
  lambda = std::max(lambda * 0.999f, 0.0001f);
  demo.Train(lambda, 13);
}

void LastInput(uint8_t* input) {
  demo.Export(demo.input->output[0], input);
}

void LastOutput(uint8_t* output) {
  demo.Export(demo.output->output[0], output);
}

void Predict(double* input, uint8_t* output) {
  demo.Import(demo.latent->output[0], input);
  Range(demo.latent->next, demo.output).Apply([](Node* node) {
    node->Forward(1);
  });
  demo.Export(demo.output->output[0], output);
}

void LoadPretrainedModel() {
  lambda = 0.0001f;
  demo.Load();
}
};

#ifndef Web
int main(int argc, const char* argv[]) {
  demo.Load();
  demo.Train(0.0001f, 6400);
  std::cout << "error " << demo.model->LastError() << std::endl;
  demo.Save();
  return 0;
}
#endif

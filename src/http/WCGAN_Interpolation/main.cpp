#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"

std::random_device seed;
std::mt19937 random_generator(seed());

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({29, 29, 1});
    for (size_t y = 0; y < 28; ++y)
      for (size_t x = 0; x < 28; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y) = input[i][index] / 255.0f;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<uint8_t> input_js(29 * 29 * 4);
std::vector<uint8_t> output_js(29 * 29 * 4);

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

auto generator = [](Node* X) {
  X = a.Linear(X, {4, 4, 16});
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {5, 5}, 8, 2);
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {5, 5}, 8, 2);
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {5, 5}, 1, 1);
  X = a.Sigmoid(X);
  return X;
};

auto discriminator = [](Node* X) {
  X = a.Convolution2D(X, {5, 5}, 8, 1);
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {5, 5}, 8, 2);
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {5, 5}, 16, 2);
  X = a.LeakyRelu(X);

  X = a.Linear(X, {1, 1, 1});
  return X;
};

auto generator_input = a.Input({10});
auto generator_output = generator(generator_input);

auto discriminator_input = a.Input({29, 29, 1});
auto discriminator_output = discriminator(discriminator_input);

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

Model model(generator_input, generator_output);

float learning_rate = 0.0002f;

std::vector<float> SaveModelToVector() {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  return model.SerializeParams();
}
std::vector<float> initial_model = SaveModelToVector();

void LoadModelFromString(const std::vector<float>& data) {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  model.DeserializeParams(data);
}

void Load() {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  model.DeserializeParamsFromFile("save.bin");
  learning_rate = 0.0002f;
}

//void Save() { model.SerializeParamsToFile("save.bin"); }

extern "C" {

void Predict(double* _input, uint8_t* _output) {
  Import(generator_input->output[0], _input);
  Range(generator_input->next, generator_output).Apply([](Node* node) {
    node->Forward(1);
  });
  Export(generator_output->output[0], _output);
}

void LoadPretrainedModel() {
  Load();
}

void ResetModelWeight() {
  LoadModelFromString(initial_model);
  learning_rate = 0.006f;
};

void Train() {
  std::vector<Example> examples;

  std::vector<Tensor> generated;

  // Train the generative network ----------------------------------------------

  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);

  // Do not train on the discriminative network.
  Range(discriminator_input, discriminator_output).Apply(&Node::Lock);
  Range(generator_input, discriminator_output).Apply(&Node::Clear);

  // Generate some examples.
  for (size_t i = 0; i < 4; ++i) {
    auto input = Tensor::Random(generator_input->output[0].sizes);
    auto output = Tensor(1);
    output.values = {1.0};
    examples.push_back({input, output});
  }

  // Train the network.
  auto generator_model = Model(generator_input, discriminator_output, examples);
  generator_model.loss_function = LossFunction::WasserStein;
  generator_model.Train(learning_rate, examples.size());

  // Train the generative network ----------------------------------------------

  // Disconnect the two networks.
  Node::Link(discriminator_input, discriminator_input->next);
  Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);
  Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

  // Generate some examples.
  examples.clear();
  std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
  for (auto& input : generator_output->output) {
    auto output = Tensor(1);
    // One fake
    output.values = {-1.f};
    examples.push_back({input, output});

    // One real
    input = training_set[random_index(random_generator)].input;
    output.values = {1.0};
    examples.push_back({input, output});
  }

  // Train the network.
  auto discriminator_model =
      Model(discriminator_input, discriminator_output, examples);
  discriminator_model.loss_function = LossFunction::WasserStein;
  discriminator_model.post_update_function = PostUpdateFunction::ClipWeight(
      discriminator_input->next, discriminator_output);

  // std::shuffle(examples.begin(), examples.end(), random_generator);
  discriminator_model.Train(learning_rate * 5.f, examples.size());
}

}; // Extern C

#ifndef Web
int main(int argc, const char* argv[]) {}
#endif

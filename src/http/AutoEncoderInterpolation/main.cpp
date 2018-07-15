#include <iostream>
#include <cmath>
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"
#include "node/Convolution2D.hpp"
#include "node/Linear.hpp"
#include "node/Relu.hpp"
#include "node/LeakyRelu.hpp"
#include "node/Sigmoid.hpp"
#include "node/Deconvolution2D.hpp"
#include "node/Input.hpp"
#include <cstdint>

#include <chrono>

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
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
  Input input;
  Convolution2D conv_1;
  Relu relu_1;
  Convolution2D conv_2;
  Relu relu_2;
  Convolution2D conv_3;
  Relu relu_3;
  Linear linear_1;
  Sigmoid sigmoid_1;
  Linear linear_2;
  Deconvolution2D deconv_1;
  Relu relu_4;
  Deconvolution2D deconv_2;
  Relu relu_5;
  Deconvolution2D deconv_3;
  Sigmoid output;

  Model model;

  Demo()
      : input({27, 27, 1}),
        conv_1(input, {7, 7}, 6, 2),
        relu_1(conv_1),
        conv_2(relu_1, {3, 3}, 16, 2),
        relu_2(conv_2),
        conv_3(relu_2, {3, 3}, 32, 2),
        relu_3(conv_3),
        linear_1(relu_3, {10}),
        sigmoid_1(linear_1),
        linear_2(sigmoid_1, linear_1.input[0]->sizes),
        deconv_1(linear_2, {3, 3}, 32, 2),
        relu_4(deconv_1),
        deconv_2(relu_4, {3, 3}, 16, 2),
        relu_5(deconv_2),
        deconv_3(relu_5, {7, 7}, 1, 2),
        output(deconv_3),
        model(input, output, training_set) {
    model.batch_size = 100;
  }

  void Train(float lambda, int iterations = 10) {
    model.Train(lambda, iterations);
  }

  void Predict();

  void Import(Tensor& tensor, double* input) {
    for (auto& it : tensor.values) {
      it = *(input++);
    }
  }

  void Export(const Tensor& tensor, uint8_t* output) {
    for (auto& it : tensor.values) {
      (*output++) = 255 * it;
      (*output++) = 255 * it;
      (*output++) = 255 * it;
      (*output++) = 255;
    }
  }

  void Save() {
    model.SerializeParamsToFile("save.bin");
  }

  void Load() {
    model.DeserializeParamsFromFile("save.bin");
  }
};

Demo demo;

extern "C" {

  float lambda = 0.003f;
  void Train() {
    lambda = std::max(lambda*0.999f, 0.0001f);
    demo.Train(lambda, 13);
  }

  void LastInput(uint8_t* input) {
    demo.Export(demo.input.output[0], input);
  }

  void LastOutput(uint8_t* output) {
    demo.Export(demo.output.output[0], output);
  }

  void Predict(double* input, uint8_t* output) {
    demo.Import(demo.sigmoid_1.output[0], input);
    Range(demo.linear_2, demo.output).Apply([](Node& node) {
      node.Forward(1);
    });
    demo.Export(demo.output.output[0], output);
  }

  void LoadPretrainedModel() {
    lambda = 0.0001f;
    demo.Load();
  }
};

#ifndef Web
int main(int argc, const char *argv[])
{
  demo.Load();
  //demo.Train(0.f, 200);
  demo.Train(0.00001f, 6400);
  std::cout << "error " << demo.model.LastError() << std::endl;
  demo.Save();
  return 0;
}
#endif

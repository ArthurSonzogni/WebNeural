#include <iostream>
#include <cmath>
#include "Image.hpp"
#include "Model.hpp"
#include "mnist/mnist_reader.hpp"
#include "node/Convolution2D.hpp"
#include "node/Linear.hpp"
#include "node/Relu.hpp"
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
  Linear linear_1;
  Sigmoid sigmoid_1;
  Linear linear_2;
  Deconvolution2D deconv_1;
  Relu relu_4;
  Deconvolution2D deconv_2;
  Sigmoid output;

  Model model;

  Demo()
      : input({27, 27, 1}),
        conv_1(input, {7, 7}, 6, 2),
        relu_1(conv_1),
        conv_2(relu_1, {5, 5}, 16, 1),
        relu_2(conv_2),
        linear_1(relu_2, {10}),
        sigmoid_1(linear_1),
        linear_2(sigmoid_1, linear_1.input->sizes),
        deconv_1(linear_2, {5, 5}, 6, 1),
        relu_4(deconv_1),
        deconv_2(relu_4, {7, 7}, 1, 2),
        output(deconv_2),
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

  void Train() {
    demo.model.batch_size = 23;
    demo.Train(0.01f / demo.model.batch_size, 13);
  }

  void LastInput(uint8_t* input) {
    demo.Export(demo.input.output, input);
  }

  void LastOutput(uint8_t* output) {
    demo.Export(demo.output.output, output);
  }

  void Predict(double* input, uint8_t* output) {
    demo.Import(demo.sigmoid_1.output, input);
    Range(demo.linear_2, demo.output).Apply(&Node::Forward);
    demo.Export(demo.output.output, output);
  }

  void LoadPretrainedModel() {
    demo.Load();
  }
};

#ifndef WEB
int main(int argc, const char *argv[])
{
  demo.Load();
  demo.model.batch_size = 111;
  demo.Train(0.f, 200);
  demo.Train(0.001f / demo.model.batch_size, 4000);
  std::cout << "error " << demo.model.LastError() << std::endl;
  demo.Save();
  return 0;
}
#endif

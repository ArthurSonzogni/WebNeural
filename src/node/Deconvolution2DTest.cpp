#include <cmath>
#include "Image.hpp"
#include "Model.hpp"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"
#include "node/Convolution2D.hpp"
#include "node/Linear.hpp"
#include "node/Relu.hpp"
#include "node/Sigmoid.hpp"
#include "node/Deconvolution2D.hpp"
#include "node/Input.hpp"

#include <chrono>

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input,
                                 const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({27, 27, 1});
    for(size_t y = 0; y<27; ++y)
    for(size_t x = 0; x<27; ++x) {
      size_t index = x + 28 *y;
      input_example.at(x,y) = 0.1 + 0.9*input[i][index] / 255.0f;
    }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

TEST(Deconvolution2D, Deconvolution2D) {
  return;
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_set =
      GetExamples(mnist.training_images, mnist.training_labels);


  // Build a neural network.
  auto input = Input({27, 27, 1});
  auto conv_1 = Convolution2D(&input, {5, 5}, 4, 2);  // {12,12,8}
  auto relu_1 = Relu(&conv_1);
  auto conv_2 = Convolution2D(&relu_1, {7, 7}, 8, 1);  // {4,4,16}
  auto relu_2 = Relu(&conv_2);
  auto linear = Linear(&relu_2, {10});
  auto relu_3 = Relu(&linear);
  auto new_base = Linear(&relu_3, linear.input[0]->sizes);
  auto deco_1 = Deconvolution2D(&new_base, {7, 7}, 4, 1); // {4,4,64}
  auto relu_4 = Relu(&deco_1);
  auto deco_2 = Deconvolution2D(&relu_4, {5, 5}, 1, 2);  // {4,4,64}
  auto output = Sigmoid(&deco_2);

  Model model(&input, &output, training_set);

  for (int i = 0; i < 1000; ++i) {
    model.batch_size = 400;
    model.Train(0.00001f, 5000);

    // Check for new predictions.
    //float error = model.Error();
    if (model.LastError() < 0.0005)
      break;

    for (int i = 0; i < 100; ++i) {
      if (180*i < model.LastError() * 100)
        std::cout << "#";
      else
        std::cout << "-";
    }
    std::cout << " " << model.LastError() << " " << model.iteration << "/"
              << model.examples.size() << std::endl;

    //std::ofstream("conv_1.pgm") << image_PGM(conv_1.params);
    //std::ofstream("conv_2.pgm") << image_PGM(conv_2.params);
    //std::ofstream("deco_1.pgm") << image_PGM(deco_1.params);
    //std::ofstream("deco_2.pgm") << image_PGM(deco_2.params);
    std::ofstream("output.pgm") << image_PGM(output.output[0], 0.f, 1.f);

    {
      std::stringstream ss;
      ss << "generated_" << std::setw(4) << std::setfill('0') << i
         << ".pgm";
      std::ofstream(ss.str()) << image_PGM(output.output[0]);
    }
  }
}

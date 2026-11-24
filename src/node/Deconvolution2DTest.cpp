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
  // Learn a known 3x3 stride-2 transposed convolution filter on synthetic data.
  Tensor target_filter({3, 3, 1, 1});
  target_filter.values = {
      0.5f,  1.0f,  0.5f,
      1.0f, -2.0f,  1.0f,
      0.5f,  1.0f,  0.5f,
  };

  auto apply_deconv = [&](const Tensor& in) {
    const size_t iw = in.sizes[0];
    const size_t ih = in.sizes[1];
    const size_t stride = 2;
    Tensor out({(iw - 1) * stride + 3, (ih - 1) * stride + 3, 1});
    for (size_t y = 0; y < ih; ++y) {
      for (size_t x = 0; x < iw; ++x) {
        const float v = in.at(x, y, 0);
        for (size_t dy = 0; dy < 3; ++dy) {
          for (size_t dx = 0; dx < 3; ++dx) {
            out.at(stride * x + dx, stride * y + dy, 0) +=
                v * target_filter.values[dx + 3 * dy];
          }
        }
      }
    }
    return out;
  };

  std::vector<Example> examples;
  for (int i = 0; i < 200; ++i) {
    Tensor in = Tensor::Random({4, 4, 1});
    examples.push_back({in, apply_deconv(in)});
  }

  auto input = Input({4, 4, 1});
  auto output = Deconvolution2D(&input, {3, 3}, 1, 2);

  Model model(&input, &output, examples);
  for (int i = 1; i <= 200; ++i) {
    model.Train(0.05f / std::sqrt(static_cast<float>(i)), 200);
    if (model.Error() < 1e-4f)
      break;
  }

  EXPECT_LT(model.Error(), 1e-4f);
  EXPECT_LT((target_filter - output.params).Error(), 1e-3f);
}

#include "Allocator.hpp"
#include "node/BatchNormalization.hpp"
#include "node/Bias.hpp"
#include "node/BilinearUpsampling.hpp"
#include "node/Border.hpp"
#include "node/Convolution2D.hpp"
#include "node/Deconvolution2D.hpp"
#include "node/Dropout.hpp"
#include "node/Input.hpp"
#include "node/LeakyRelu.hpp"
#include "node/Linear.hpp"
#include "node/MaxPooling.hpp"
#include "node/Node.hpp"
#include "node/Noise.hpp"
#include "node/Relu.hpp"
#include "node/Sigmoid.hpp"
#include "node/Softmax.hpp"
#include "node/Tanh.hpp"

Allocator::Allocator() = default;

Node* Allocator::BatchNormalization(Node* input) {
  nodes.emplace_back(new ::BatchNormalization(input));
  return nodes.back().get();
}

Node* Allocator::Bias(Node* input) {
  nodes.emplace_back(new ::Bias(input));
  return nodes.back().get();
}

Node* Allocator::BilinearUpsampling(Node* input) {
  nodes.emplace_back(new ::BilinearUpsampling(input));
  return nodes.back().get();
}

Node* Allocator::Convolution2D(Node* input,
                               const std::vector<size_t> filter_size,
                               size_t num_features,
                               size_t stride) {
  nodes.emplace_back(
      new ::Convolution2D(input, filter_size, num_features, stride));
  return nodes.back().get();
}

Node* Allocator::Deconvolution2D(Node* input,
                                 std::vector<size_t> filter_size,
                                 size_t num_filters,
                                 size_t stride) {
  nodes.emplace_back(
      new ::Deconvolution2D(input, filter_size, num_filters, stride));
  return nodes.back().get();
}

Node* Allocator::Dropout(Node* input, float ratio) {
  nodes.emplace_back(new ::Dropout(input, ratio));
  return nodes.back().get();
}

Node* Allocator::Input(const std::vector<size_t>& size) {
  nodes.emplace_back(new ::Input(size));
  return nodes.back().get();
}

Node* Allocator::LeakyRelu(Node* input) {
  nodes.emplace_back(new ::LeakyRelu(input));
  return nodes.back().get();
}

Node* Allocator::Linear(Node* input, std::vector<size_t> output_sizes) {
  nodes.emplace_back(new ::Linear(input, output_sizes));
  return nodes.back().get();
}

Node* Allocator::MaxPooling(Node* input) {
  nodes.emplace_back(new ::MaxPooling(input));
  return nodes.back().get();
}

Node* Allocator::Relu(Node* input) {
  nodes.emplace_back(new ::Relu(input));
  return nodes.back().get();
}

Node* Allocator::Sigmoid(Node* input) {
  nodes.emplace_back(new ::Sigmoid(input));
  return nodes.back().get();
}

Node* Allocator::Tanh(Node* input) {
  nodes.emplace_back(new ::Tanh(input));
  return nodes.back().get();
}

Node* Allocator::Softmax(Node* input) {
  nodes.emplace_back(new ::Softmax(input));
  return nodes.back().get();
}

Node* Allocator::Noise(Node* input, float sigma) {
  nodes.emplace_back(new ::Noise(input, sigma));
  return nodes.back().get();
}

Node* Allocator::Border(Node* input, size_t border_size, float value) {
  nodes.emplace_back(new ::Border(input, border_size, value));
  return nodes.back().get();
}

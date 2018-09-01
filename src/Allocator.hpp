#ifndef NODEALLOCATOR_H
#define NODEALLOCATOR_H

#include "node/Node.hpp"
#include <memory>

class Allocator {
 public:

  // Input
  Node* Input(const std::vector<size_t>& size);

  // Linear
  Node* Linear(Node* input, std::vector<size_t> output_sizes);
  Node* Bias(Node* input);
  Node* Convolution2D(Node* input,
                      const std::vector<size_t> filter_size,
                      size_t num_features,
                      size_t stride = 1);
  Node* Deconvolution2D(Node* input,
                        std::vector<size_t> filter_size,
                        size_t num_filters,
                        size_t stride);

  // Activations.
  Node* LeakyRelu(Node* input);
  Node* Relu(Node* input);
  Node* Sigmoid(Node* input);
  Node* Softmax(Node* input);
  Node* Tanh(Node* input);

  // Upsampling/Downsampling.
  Node* MaxPooling(Node* input);
  Node* BilinearUpsampling(Node* input);

  // Random.
  Node* Noise(Node* input, float sigma);
  Node* Dropout(Node* input, float ratio);

  // Regularisation
  Node* BatchNormalization(Node* input);

 private:
  std::vector<std::unique_ptr<Node>> nodes;
};

#endif /* end of include guard: NODEALLOCATOR_H */

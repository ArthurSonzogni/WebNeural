#ifndef NODEALLOCATOR_H
#define NODEALLOCATOR_H

#include "node/Node.hpp"
#include <memory>

class Allocator {
 public:
  Node* BatchNormalization(Node* input);
  Node* Bias(Node* input);
  Node* BilinearUpsampling(Node* input);
  Node* Convolution2D(Node* input,
                      const std::vector<size_t> filter_size,
                      size_t num_features,
                      size_t stride = 1);
  Node* Deconvolution2D(Node* input,
                        std::vector<size_t> filter_size,
                        size_t num_filters,
                        size_t stride);
  Node* Dropout(Node* input, float ratio);
  Node* Input(const std::vector<size_t>& size);
  Node* LeakyRelu(Node* input);
  Node* Linear(Node* input, std::vector<size_t> output_sizes);
  Node* MaxPooling(Node* input);
  Node* Relu(Node* input);
  Node* Sigmoid(Node* input);
  Node* Softmax(Node* input);

 private:
  std::vector<std::unique_ptr<Node>> nodes;
};

#endif /* end of include guard: NODEALLOCATOR_H */

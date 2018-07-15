#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "node/Node.hpp"

class LeakyRelu : public Node {
 public:
  LeakyRelu(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: LEAKY_RELU_H */

#ifndef RELU_H
#define RELU_H

#include "node/Node.hpp"

class Relu : public Node {
 public:
  Relu(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: RELU_H */

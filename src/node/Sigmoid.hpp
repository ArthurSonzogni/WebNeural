#ifndef SIGMOID_H
#define SIGMOID_H

#include "node/Node.hpp"

class Sigmoid : public Node {
 public:
  Sigmoid(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: SIGMOID_H */

#ifndef RELU_H
#define RELU_H

#include "node/Node.hpp"

class Relu : public Node {
 public:
  Relu(Node& input);
  void Forward() override;
  void Backward() override;
};

#endif /* end of include guard: RELU_H */

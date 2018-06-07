#ifndef SIGMOID_H
#define SIGMOID_H

#include "node/Node.hpp"

class Sigmoid : public Node {
 public:
  Sigmoid(Node& input);
  void Forward() override;
  void Backward() override;
};

#endif /* end of include guard: SIGMOID_H */

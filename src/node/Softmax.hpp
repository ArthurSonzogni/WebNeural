#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "node/Node.hpp"

class Softmax : public Node {
 public:
  Softmax(Node& input);
  void Forward() override;
  void Backward() override;
};

#endif /* end of include guard: SOFTMAX_H */

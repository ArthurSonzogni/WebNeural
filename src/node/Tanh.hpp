#ifndef TANH_H
#define TANH_H

#include "node/Node.hpp"

class Tanh : public Node {
 public:
  Tanh(Node* input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: TANH_H */

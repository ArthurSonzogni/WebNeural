#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "node/Node.hpp"

class Softmax : public Node {
 public:
  Softmax(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: SOFTMAX_H */

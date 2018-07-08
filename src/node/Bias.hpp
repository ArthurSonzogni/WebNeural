#ifndef BIAS_H
#define BIAS_H

#include "node/Node.hpp"

class Bias : public Node {
 public:
  Bias(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: BIAS_H */

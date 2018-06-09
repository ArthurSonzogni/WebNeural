#ifndef BIAS_H
#define BIAS_H

#include "node/Node.hpp"

class Bias : public Node {
 public:
  Bias(Node& input);
  void Forward() override;
  void Backward() override;
};

#endif /* end of include guard: BIAS_H */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "node/Node.hpp"

class Dropout : public Node {
 public:
  Dropout(Node& input, float ratio);
  void Forward() override;
  void Backward() override;
  float ratio;
  Tensor random;
};

#endif /* end of include guard: DROPOUT_H */

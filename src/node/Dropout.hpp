#ifndef DROPOUT_H
#define DROPOUT_H

#include "node/Node.hpp"

class Dropout : public Node {
 public:
  Dropout(Node& input, float ratio);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  float ratio;
  std::vector<Tensor> random;
};

#endif /* end of include guard: DROPOUT_H */

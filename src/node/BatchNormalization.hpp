#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "node/Node.hpp"

class BatchNormalization : public Node {
 public:
  BatchNormalization(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  Tensor pixel_deviation;
};

#endif /* end of include guard: BATCH_NORMALIZATION_H */

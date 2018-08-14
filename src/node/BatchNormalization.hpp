#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "node/Node.hpp"

class BatchNormalization : public Node {
 public:
  BatchNormalization(Node* input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  float inv_dev = 1.f;
};

#endif /* end of include guard: BATCH_NORMALIZATION_H */

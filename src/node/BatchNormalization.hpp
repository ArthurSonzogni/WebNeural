#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "node/Node.hpp"

class BatchNormalization : public Node {
 public:
  BatchNormalization(Node* input);
  ~BatchNormalization() = default;
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  float mean = 0.f;
  float inv_dev = 1.f;
  float running_mean = 0.f;
  float running_var = 1.f;
  bool has_running_stats = false;
};

#endif /* end of include guard: BATCH_NORMALIZATION_H */

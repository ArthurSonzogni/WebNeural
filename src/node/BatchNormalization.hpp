#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "node/Node.hpp"

class BatchNormalization : public Node {
 public:
  BatchNormalization(Node* input);
  ~BatchNormalization() = default;
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
  void EnsureBatchSize(size_t batch_size) override;
  void SerializeWeights(std::vector<float>& value) const override;
  void DeserializeWeights(const std::vector<float>& value,
                          size_t& index) override;
  size_t SerializedWeightCount() const override;

 private:
  size_t channels = 1;
  size_t spatial_size = 1;
  std::vector<float> mean;
  std::vector<float> inv_dev;
  std::vector<float> running_mean;
  std::vector<float> running_var;
  std::vector<Tensor> x_hat;
  bool has_running_stats = false;
};

#endif /* end of include guard: BATCH_NORMALIZATION_H */

#ifndef NODE_H
#define NODE_H

#include <cstddef>
#include <vector>
#include "Tensor.hpp"

class Node {
 public:
  Node() = default;
  virtual ~Node() = default;

  // DAG connectivity.
  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

  // Trainable parameters and accumulated parameter gradient.
  Tensor params;
  Tensor params_grad;

  // Per-batch forward outputs and output gradients (dL / d(output)).
  // Lazily grown via EnsureBatchSize(batch_size); slot 0 is always present.
  std::vector<Tensor> output;
  std::vector<Tensor> grad;

  bool training = true;

  virtual void Forward(size_t batch_size) = 0;
  // Backward MUST accumulate (+=) into each inputs[k]->grad[batch] and
  // into params_grad (when the node has parameters).
  virtual void Backward(size_t batch_size) = 0;

  virtual void EnsureBatchSize(size_t batch_size);
  void ZeroGrad(size_t batch_size);
  void ZeroParamGrad();
  void SetInput(size_t index, Node* new_input);

  // Compact weight-only serialization (params + non-trainable state like BN).
  virtual void SerializeWeights(std::vector<float>& value) const;
  virtual void DeserializeWeights(const std::vector<float>& value,
                                  size_t& index);
  virtual size_t SerializedWeightCount() const;

 protected:
  void AddInput(Node* input_node);
  void InitBuffers(const std::vector<size_t>& output_sizes);
};

#endif /* end of include guard: NODE_H */

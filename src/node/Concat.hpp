#ifndef WEBNEURAL_NODE_CONCAT_HPP
#define WEBNEURAL_NODE_CONCAT_HPP

#include "node/Node.hpp"

// Concatenates two tensors along the last/channel dimension (for U-Nets,
// DenseNets, Conditional GANs).
class Concat : public Node {
 public:
  Concat(Node* a, Node* b);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;

 private:
  size_t size_a_ = 0;
  size_t size_b_ = 0;
};

#endif /* end of include guard: WEBNEURAL_NODE_CONCAT_HPP */

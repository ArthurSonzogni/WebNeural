#ifndef WEBNEURAL_NODE_ADD_HPP
#define WEBNEURAL_NODE_ADD_HPP

#include "node/Node.hpp"

// Element-wise addition of two tensors of identical shape (for skip/residual
// connections).
class Add : public Node {
 public:
  Add(Node* a, Node* b);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: WEBNEURAL_NODE_ADD_HPP */

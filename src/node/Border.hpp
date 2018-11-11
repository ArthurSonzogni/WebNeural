#ifndef BORDER_H
#define BORDER_H

#include "node/Node.hpp"

class Border : public Node {
 public:
  Border(Node* input, size_t border_size, float value);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  size_t border_size;
  float value;
  size_t dim_x;
  size_t dim_y;
};

#endif /* end of include guard: BORDER_H */

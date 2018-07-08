#ifndef INPUT_H
#define INPUT_H

#include "Node.hpp"

class Input : public Node {
 public:
  Input(const std::vector<size_t>& size);

  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: INPUT_H */

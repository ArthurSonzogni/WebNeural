#ifndef LINEAR_H
#define LINEAR_H

#include "Node.hpp"

class Linear : public Node {
  public:
    Linear(Node* node, std::vector<size_t> output_sizes);
    void Forward(size_t batch_size) override;
    void Backward(size_t batch_size) override;
  private:
    size_t input_size;
    size_t output_size;
};

#endif /* end of include guard: LINEAR_H */

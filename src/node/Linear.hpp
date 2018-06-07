#ifndef LINEAR_H
#define LINEAR_H

#include "Node.hpp"

class Linear : public Node {
  public:
    Linear(Node& node, size_t num_output);
    void Forward() override;
    void Backward() override;
  private:
    size_t input_size;
    size_t output_size;
};

#endif /* end of include guard: LINEAR_H */

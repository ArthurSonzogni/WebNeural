#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include "Node.hpp"

class Convolution2D : public Node {
  public:
    Convolution2D(Node& node, const std::vector<size_t> half_size, size_t num_features);
    void Forward() override;
    void Backward() override;
  private:
    std::vector<size_t> size_input;
    std::vector<size_t> size_params;
    std::vector<size_t> size_output;
};

#endif /* end of include guard: CONVOLUTION2D_H */

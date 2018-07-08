#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include "Node.hpp"

class Convolution2D : public Node {
  public:
   Convolution2D(Node& node,
                 const std::vector<size_t> filter_size,
                 size_t num_features,
                 size_t stride = 1);
   void Forward(size_t batch_size) override;
   void Backward(size_t batch_size) override;
  private:
    std::vector<size_t> size_input;
    std::vector<size_t> size_params;
    std::vector<size_t> size_output;
    const size_t stride;
};

#endif /* end of include guard: CONVOLUTION2D_H */

#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include "Node.hpp"

class Convolution2D : public Node {
  public:
    Convolution2D(Node& node, const std::vector<size_t> sizes, size_t num_features);
    void Forward() override;
    void Backward() override;
  private:
    size_t input_dimx;
    size_t input_dimy
    size_t input_channels;

    size_t filter_dimx;
    size_t filter_dimy
    size_t filter_channels;

    size_t output_dim_x;
    size_t output_dim_y;
    size_t output_channels;
};

#endif /* end of include guard: CONVOLUTION2D_H */

#include "node/Convolution2D.hpp"

Convolution2D::Convolution2D(Node& node,
                             const std::vector<size_t> sizes,
                             size_t num_features) {
  Link(node);

  input_dimx = input->sizes[0];
  input_dimy = input->sizes[1];
  input_channels = input->values.size() / (input_dimx * input_dimy);

  filter_dimx = sizes[0];
  filter_dimy = sizes[1];
  filter_channels = input_channels;

  output_dim_x = input_dimx - filter_dimx + 1;
  output_dim_y = output_dimy - filter_dimy + 1;
  output_channels = num_features;

 
}
void Convolution2D::Forward() {}
void Convolution2D::Backward() {}

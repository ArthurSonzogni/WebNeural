#ifndef DECONVOLUTION_2D_HPP
#define DECONVOLUTION_2D_HPP

#include "node/Node.hpp"

class Deconvolution2D : public Node {
 public:
  Deconvolution2D(Node& input,
                  std::vector<size_t> filter_size,
                  size_t num_filters,
                  size_t stride);
  void Forward() override;
  void Backward() override;

 private:
  std::vector<size_t> size_input;
  std::vector<size_t> size_params;
  std::vector<size_t> size_output;
  const size_t stride;
};

#endif /* end of include guard: DECONVOLUTION_2D_HPP */

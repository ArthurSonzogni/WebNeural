#ifndef BILINEAR_UPSAMPLING_H
#define BILINEAR_UPSAMPLING_H

#include "node/Node.hpp"

class BilinearUpsampling : public Node {
 public:
  BilinearUpsampling(Node& input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
};

#endif /* end of include guard: BILINEAR_UPSAMPLING_H */

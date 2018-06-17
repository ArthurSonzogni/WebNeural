#ifndef BILINEAR_UPSAMPLING_H
#define BILINEAR_UPSAMPLING_H

#include "node/Node.hpp"

class BilinearUpsampling : public Node {
 public:
  BilinearUpsampling(Node& input);
  void Forward() override;
  void Backward() override;
};

#endif /* end of include guard: BILINEAR_UPSAMPLING_H */

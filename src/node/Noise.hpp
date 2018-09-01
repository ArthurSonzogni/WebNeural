#ifndef NOISE_H
#define NOISE_H

#include "node/Node.hpp"

class Noise : public Node {
 public:
  Noise(Node* input, float sigma);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  float sigma = 0.f;
};

#endif /* end of include guard: NOISE_H */

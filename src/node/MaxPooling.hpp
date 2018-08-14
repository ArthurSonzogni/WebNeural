#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include "node/Node.hpp"

class MaxPooling : public Node {
 public:
  MaxPooling(Node* input);
  void Forward(size_t batch_size) override;
  void Backward(size_t batch_size) override;
 private:
  void ForwardZ(float* i, float* o);
  void BackwardZ(float* i, float* is, float* o, float* os);
};

#endif /* end of include guard: MAXPOOLING_H */

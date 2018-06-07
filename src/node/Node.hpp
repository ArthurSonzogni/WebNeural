#ifndef NODE_H
#define NODE_H

#include "Tensor.hpp"

class Node {
 public:
  // Forward step
  Tensor* input = nullptr;
  Tensor params;
  Tensor output;

  // Backward
  Tensor input_sensitivity;
  Tensor params_sensitivity;
  Tensor* output_sensitivity = nullptr;

  virtual void Forward() = 0;
  virtual void Backward() = 0;
  void Update(float lambda);

 protected:
  void Link(Node& operation);
};

#endif /* end of include guard: NODE_H */

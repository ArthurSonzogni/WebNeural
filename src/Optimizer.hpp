#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "node/Node.hpp"

struct Example {
  Tensor input;
  Tensor output;
};

class Optimizer {
  public:
   Optimizer(Node& node,
             size_t batch_size,
             const std::vector<Example>& examples);
   void Train(float lambda, size_t iteration);
   Tensor Predict(const Tensor& input);
   float Error();
   float ErrorInteger();
   float LastError();

   std::vector<Example> examples;
   size_t iteration = 0;
   size_t batch_size;

  private:
    void Update(float lambda);
    void Forward();
    void Backward();
    std::vector<Node*> nodes;
    float last_error = 0.f;
};

#endif /* end of include guard: OPTIMIZER_H */

#ifndef MODEL_H
#define MODEL_H

#include "node/Node.hpp"

struct Example {
  Tensor input;
  Tensor output;
};

class Model {
 public:
  Model(Node& input, Node& output, const std::vector<Example>& examples);

  void Train(float lambda, size_t iteration);
  float OptimizeInput(const Tensor& output_target, float lambda);
  Tensor Predict(const Tensor& input);
  float Error();
  float ErrorInteger();
  float LastError();

  // Save/Load model weights.
  std::string SerializeParams();
  void DeserializeParams(const std::string& value);

  Node& input;
  Node& output;
  std::vector<Example> examples;
  size_t iteration = 0;
  size_t batch_size = 20;

 private:
  std::vector<Node*> nodes;  // ]input, output]
  float last_error = 0.f;
};

#endif /* end of include guard: MODEL_H */

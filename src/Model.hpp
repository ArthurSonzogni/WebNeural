#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include "Graph.hpp"
#include "LossFunction.hpp"
#include "Optimizer.hpp"
#include "PostUpdateFunction.hpp"
#include "node/Node.hpp"

struct Example {
  Tensor input;
  Tensor output;
};

class Model {
 public:
  Model(Node* input, Node* output, const std::vector<Example>& examples);
  Model(Node* input, Node* output);

  void Train(float lambda, size_t iteration);
  Tensor Predict(const Tensor& input);
  float Error();
  float ErrorInteger();
  float LastError();

  // Save/Load model weights.
  std::vector<float> SerializeParams();
  void DeserializeParams(const std::vector<float>& value);
  void SerializeParamsToFile(const std::string& filename);
  void DeserializeParamsFromFile(const std::string& filename);

  void PrintGradient();

  Node* input;
  Node* output;
  Graph graph;
  Adam optimizer;
  std::vector<Example> examples;
  size_t iteration = 0;
  size_t batch_size = 64;

  LossFunction::F* loss_function = LossFunction::SquaredDifference;
  PostUpdateFunction::F post_update_function = PostUpdateFunction::None();

 private:
  float last_error = 0.f;
};

#endif /* end of include guard: MODEL_H */

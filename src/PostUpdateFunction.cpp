#include "PostUpdateFunction.hpp"

#include "Graph.hpp"
#include "Model.hpp"
#include "node/Node.hpp"

namespace PostUpdateFunction {

F None() {
  return [](Model*) {};
}

F ClipWeight(Node* output, float c) {
  return [output, c](Model*) {
    Graph g(output);
    for (Node* node : g.TrainableNodes()) {
      node->params.Clip(c);
    }
  };
}

F WeightDecay(Node* output, float factor) {
  return [output, factor](Model*) {
    Graph g(output);
    for (Node* node : g.TrainableNodes()) {
      for (auto& it : node->params.values) {
        it *= factor;
      }
    }
  };
}

}  // namespace PostUpdateFunction

#include "PostUpdateFunction.hpp"

#include "Model.hpp"
#include "node/Node.hpp"

namespace PostUpdateFunction {

F None() {
  return [](Model*) {};
}

F ClipWeight(Node* begin, Node* end) {
  return [=](Model* model) {
    Range(begin, end).Apply([](Node* node) { node->params.Clip(2); });
  };
}

F GradientPenalty(Node* begin, Node* end, float penalty) {
  return [=](Model* model) {
    Range(begin, end).Apply([=](Node* node) {
      for (auto it : node->params.values) {
        it *= penalty;
      }
    });
  };
}

}  // namespace PostUpdateFunction

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

}  // namespace PostUpdateFunction

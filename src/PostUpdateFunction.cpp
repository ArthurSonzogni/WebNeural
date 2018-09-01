#include "PostUpdateFunction.hpp"
#include "Model.hpp"

namespace PostUpdateFunction {

  void None(Model*) {}

  void ClipWeight(Model* model) {
    Range(model->input->next, model->output).Apply([](Node* node) {
      node->params.Clip(1.0);
    });
  }

} // namespace PostUpdateFunction

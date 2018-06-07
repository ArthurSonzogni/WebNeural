#include "Input.hpp"

Input::Input(const std::vector<size_t>& size) {
  output = Tensor(size);
  output.producer = this;
}

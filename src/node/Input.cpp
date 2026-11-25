#include "Input.hpp"

Input::Input(const std::vector<size_t>& size) {
  InitBuffers(size);
}

void Input::Forward(size_t batch_size) {
  (void)batch_size;
}

void Input::Backward(size_t batch_size) {
  (void)batch_size;
}

#include "Input.hpp"

Input::Input(const std::vector<size_t>& size) {
  output = std::vector<Tensor>(T, Tensor(size));
  
  params_sensitivity = std::vector<Tensor>(T, Tensor());
  input_sensitivity = std::vector<Tensor>(T, Tensor());
}

void Input::Forward(size_t batch_size) {
  //#pragma omp parallel for
  //for(size_t batch = 0; batch < batch_size; ++batch) {
    //Tensor& O = output[batch];
    //O = params;
  //}
}

void Input::Backward(size_t batch_size) {
}

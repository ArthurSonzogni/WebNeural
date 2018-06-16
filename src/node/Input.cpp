#include "Input.hpp"

Input::Input(const std::vector<size_t>& size) {
  output = Tensor(size);

  params = Tensor(size);
  params_sensitivity = Tensor(size);
}

void Input::Forward() {
  output.values = params.values;
}

void Input::Backward() {
  for(size_t i = 0; i<params_sensitivity.values.size(); ++i) {
    params_sensitivity[i] += (*output_sensitivity)[i];
  }

  for(size_t y = 0; y<params.sizes[1]; ++y) {
    for(size_t x = 0; x<params.sizes[0]; ++x) {
      float delta = 0;
      delta += params[(x + 1) % params.sizes[0] + params.sizes[0] * ((y + 0)% params.sizes[1])];
      delta += params[(x - 1) % params.sizes[0] + params.sizes[0] * ((y + 0)% params.sizes[1])];
      delta += params[(x + 0) % params.sizes[0] + params.sizes[0] * ((y + 1)% params.sizes[1])];
      delta += params[(x + 0) % params.sizes[0] + params.sizes[0] * ((y - 1)% params.sizes[1])];
      delta -= params[(x + 0) % params.sizes[0] + params.sizes[0] * ((y + 0)% params.sizes[1])] * 4;
      params_sensitivity[x + params.sizes[0] * y] += delta * 0.001f;
    }
  }
}

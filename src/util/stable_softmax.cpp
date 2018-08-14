#include "util/stable_softmax.hpp"
#include <cmath>

void StableSoftmax(const std::vector<float>& input,
                   std::vector<float>& output) {
  float best = input[0];
  for (float v : input)
    best = std::max(best, v);

  for (size_t i = 0; i < output.size(); ++i)
    output[i] = exp(input[i] - best);

  // Normalize probability vector.
  float sum = 0.f;
  for (auto& v : output)
    sum += v;
  for (auto& v : output)
    v /= sum;
}

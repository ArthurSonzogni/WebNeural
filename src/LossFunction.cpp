#include "LossFunction.hpp"
#include <cmath>
#include <iostream>
#include "util/stable_softmax.hpp"

namespace LossFunction {

void SquaredDifference(const Tensor& target,
                       const Tensor& current,
                       float* error,
                       Tensor* derivative) {
  *derivative = current - target;
  *error = derivative->Error();
}

// Sum(target, log(current));
void CrossEntropy(const Tensor& target,
                  const Tensor& current,
                  float* error,
                  Tensor* derivative) {
  *error = 0.f;
  *derivative = Tensor(target.sizes);
  derivative->Fill(0.f);
  for (size_t i = 0; i < target.values.size(); ++i) {
    //if (current[i] * (1.f - current[i]) > 0.000000000001f) {
      ///[>error += -target[i] * log(current[i]);
      if (target[i] > 0.5f) {
        float c = std::max(current[i], 1e-10f);
        *error += -target[i] * log2f(c/target[i]);
        (*derivative)[i] -= target[i] / c;
        if (std::isnan((*derivative)[i])) {
          std::cout << "nan => " << c << std::endl;
        }
      }
      //std::cout << "error = " << *error << " " << target[i] << " " << current[i]
                //<< std::endl;
    //}
    //(*derivative)[i] = 0.f;
  }
  //SquaredDifference(target, current, error, derivative);
}

void SoftmaxCrossEntropy(const Tensor& target,
                         const Tensor& current,
                         float* error,
                         Tensor* derivative) {
  const size_t size = target.values.size();
  *derivative = Tensor(size);
  std::vector<float> softmax(size);
  StableSoftmax(current.values, softmax);

  *error = 0.f;
  for (size_t i = 0; i < size; ++i) {
    if (target[i] > 0.5) {
      //*error += 1-softmax[i];// - target[i];
      *error += -log(std::max(1e-10f, softmax[i]));
    }
  }

  // Compute derivative.
  for (size_t i = 0; i < size; ++i)
    (*derivative)[i] = softmax[i] - target[i];
}

void WasserStein(const Tensor& target,
                  const Tensor& current,
                  float* error,
                  Tensor* derivative) {
  if (target[0] > 0.5f) {
    *error = +current[0];
    (*derivative)[0] = +1.f;
  } else {
    *error = -current[0];
    (*derivative)[0] = -1.f;
  }
}

}  // namespace LossFunction

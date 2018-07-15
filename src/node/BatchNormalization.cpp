#include "node/BatchNormalization.hpp"
#include <cmath>

BatchNormalization::BatchNormalization(Node& node) {
  Link(node);

  output = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  pixel_deviation = Tensor(input[0]->sizes);
  pixel_deviation.Fill(1.0f);

  InitInternalSensitivity();
}

void BatchNormalization::Forward(size_t batch_size) {
  const size_t size = output[0].values.size();

  for(size_t i = 0; i < size; ++i) {
    float one = 0.f;
    float X = 0.f;
    float XX = 0.f;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto x = input[batch]->values[i];
      one += 1.f;
      X += x;
      XX += x*x;
    }
    X /= one;
    XX /= one;
    const float mean = X;
    float inv_dev = 1.0 / std::sqrt(XX - X*X + 1e-2);

    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto& I = input[batch]->values[i];
      auto& O = output[batch].values[i];

      O = (I - mean) * inv_dev;
    }

    pixel_deviation[i] = inv_dev;
  }

  //X /= one;
  //XX /= one;


  //const size_t size = output[0].values.size();
  ////const float deviation = pixel_deviation[batch];
  //for (size_t batch = 0; batch < batch_size; ++batch) {
    //Tensor& O = output[batch];
    //Tensor& I = *(input[batch]);
    //for (size_t i = 0; i < size; ++i) {
      //O[i] = (I[i] - mean) * inv_dev;
    //}
  //}
}

void BatchNormalization::Backward(size_t batch_size) {
  //#pragma omp parallel for
  //for (size_t batch = 0; batch < batch_size; ++batch) {
    //Tensor& OS = *(output_sensitivity[batch]);
    //Tensor& IS = input_sensitivity[batch];

    //const size_t size = IS.values.size();
    ////const float deviation = pixel_deviation[batch];
    //for (size_t i = 0; i < size; ++i) {
      ////IS[i] = OS[i] * deviation;
      //IS[i] = OS[i] * inv_dev;
    //}
  //}
  const size_t size = output[0].values.size();
  for(size_t i = 0; i < size; ++i) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
      auto& IS = input_sensitivity[batch].values[i];
      auto& OS = output_sensitivity[batch]->values[i];

      IS = pixel_deviation[i] * OS;
    }
  }
}

#include "Tensor.hpp"
#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

Tensor::Tensor() : Tensor({}) {}
Tensor::Tensor(size_t size) : Tensor(std::vector<size_t>{size}) {}
Tensor::Tensor(const std::vector<size_t>& sizes)
    : values(Multiply(sizes), 0.f), sizes(sizes) {}

void Tensor::Fill(float value) {
  std::fill(values.begin(), values.end(), value);
}

Tensor Tensor::operator-(const Tensor& tensor) const {
  Tensor output = *this;
  for (size_t i = 0; i < values.size(); ++i) {
    output[i] -= tensor.values.at(i);
  }
  return output;
}

void Tensor::operator*=(float lambda) {
  for (auto& it : values) {
    it *= lambda;
  }
}

void Tensor::operator+=(const Tensor& other) {
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] += other[i];
  }
}

bool Tensor::operator==(const Tensor& other) const {
  return sizes == other.sizes && values == other.values;
}

bool Tensor::operator!=(const Tensor& other) const {
  return !(*this == other);
}

float Tensor::Error() {
  float ret = 0.f;
  for (const auto i : values) {
    ret += i * i;
  }
  return ret;
}

size_t Tensor::ArgMax() {
  size_t i_max = 0;
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] > values[i_max])
      i_max = i;
  }
  return i_max;
}

std::string Tensor::ToString() {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      ss << ", ";
    ss << values[i];
  }
  ss << "]";
  return ss.str();
}

// static
Tensor Tensor::Random(const std::vector<size_t>& sizes) {
  Tensor tensor(sizes);
  tensor.Randomize();
  return tensor;
}

// static
Tensor Tensor::SphericalRandom(const std::vector<size_t>& sizes) {
  Tensor tensor = Tensor::Random(sizes);
  float XX = 0.f;
  for (auto x : tensor.values) {
    XX += x * x;
  }
  const float inv_norm = 1.0 / std::sqrt(XX);
  for (auto& x : tensor.values) {
    x *= inv_norm;
  }
  return tensor;
}

void Tensor::Randomize() {
  static std::mt19937 rng;
  std::normal_distribution<float> random(0.0, 1.0);
  for (auto& i : values)
    i = random(rng);
}

void Tensor::UniformRandom() {
  static std::mt19937 rng;
  std::uniform_real_distribution<float> random(0.0, 1.0);
  for (auto& i : values)
    i = random(rng);
}

float& Tensor::at(size_t x, size_t y) {
  return values[x + sizes[0] * y];
}
float& Tensor::at(size_t x, size_t y, size_t z) {
  return values[x + sizes[0] * (y + sizes[1] * z)];
}
float Tensor::at(size_t x, size_t y) const {
  return values[x + sizes[0] * y];
}
float Tensor::at(size_t x, size_t y, size_t z) const {
  return values[x + sizes[0] * (y + sizes[1] * z)];
}

// static
Tensor Tensor::Merge(std::vector<Tensor> tensors, int dim_x) {
  size_t dx = dim_x ? dim_x : std::sqrt(tensors.size());
  size_t dy = (tensors.size()+dx-1) / dx;
  //size_t dx = tensors.size();
  //size_t dy = 1;
  size_t width = tensors[0].sizes[0];
  size_t height = tensors[0].sizes.size() >= 2 ? tensors[0].sizes[1] : 1;
  size_t component = tensors[0].sizes.size() >= 3 ? tensors[0].sizes[2] : 1;
  Tensor merge({width * dx, height * dy, component});

  size_t i_dx = 0;
  size_t i_dy = 0;
  for (auto& t : tensors) {
    for (size_t c = 0; c < component; ++c) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          merge.at(x + width * i_dx, y + height * i_dy, c) = t.at(x,y,c);
        }
      }
    }

    ++i_dx;
    if (i_dx >= dx) {
      i_dx = 0;
      ++i_dy;
    }
  }
  return merge;
}

void Tensor::Clip(const float c) {
  for (auto& v : values)
    v = std::min(c, std::max(-c, v));
}

void Tensor::Clip(const float v_min, const float v_max) {
  for (auto& v : values)
    v = std::min(v_max, std::max(v_min, v));
}

void Tensor::Rescale(const float min, const float max) {
  float input_min = values[0];
  float input_max = values[0];
  for(const auto& v : values) {
    input_min = std::min(input_min, v);
    input_max = std::max(input_max, v);
  }

  for(auto& v : values) {
    v = (v - input_min) * (max - min) / (input_max - input_min) + min;
  }
}

//static
Tensor Tensor::ConcatenateHorizontal(const Tensor& A, const Tensor& B) {
  const size_t width_A = A.sizes[0];
  const size_t width_B = B.sizes[0];
  const size_t height_A = A.sizes[1];
  const size_t height_B = B.sizes[1];

  if (height_A != height_B)
    throw std::invalid_argument("height doesn't match");
  
  Tensor ret({width_A + width_B, height_A});
  for(size_t y = 0; y<height_A; ++y) {
    size_t x = 0;
    for(size_t x_a = 0; x_a<width_A; ++x_a)
      ret.at(x++, y) = A.at(x_a,y);
    for(size_t x_b = 0; x_b<width_B; ++x_b)
      ret.at(x++, y) = B.at(x_b,y);
  }
  return ret;
}

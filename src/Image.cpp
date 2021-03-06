#include "Image.hpp"
#include "sstream"
#include <cmath>

std::string image_PGM(const Tensor& tensor) {
  float v_min = tensor.values[0];
  float v_max = tensor.values[0];
  for (const auto& it : tensor.values) {
    v_min = std::min(v_min, it);
    v_max = std::max(v_max, it);
  }
  return image_PGM(tensor, v_min, v_max);
}

std::string image_PGM(const Tensor& tensor, float v_min, float v_max) {
  size_t width = tensor.sizes[0];
  size_t height = tensor.sizes[1];
  size_t num_images = tensor.values.size() / (width * height);
  size_t total_height = (height + 1) * num_images - 1;

  std::stringstream ss;
  ss << "P2" << std::endl;
  ss << width << " " << total_height << std::endl;
  ss << "255" << std::endl;

  for (size_t image = 0; image < num_images; ++image) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        ss << int(255 *
                  (tensor.values[x + width * (y + height * image)] - v_min) /
                  (v_max - v_min))
           << " ";
      }
      ss << std::endl;
    }
    if (image != num_images - 1) {
      for (size_t x = 0; x < width; ++x) {
        ss << 0 << " ";
      }
    }
    ss << std::endl;
  }

  return ss.str();
}

std::string image_PPM_centered(const Tensor& tensor) {
  float v = tensor.values[0];
  for (const auto& it : tensor.values) {
    v = std::max(std::abs(v), it);
  }
  return image_PPM(tensor, -v, +v);
}

std::string image_PPM(const Tensor& tensor) {
  float v_min = tensor.values[0];
  float v_max = tensor.values[0];
  for (const auto& it : tensor.values) {
    v_min = std::min(v_min, it);
    v_max = std::max(v_max, it);
  }
  return image_PPM(tensor, v_min, v_max);
}

std::string image_PPM(const Tensor& tensor, float v_min, float v_max) {
  size_t width = tensor.sizes[0];
  size_t height = tensor.sizes[1];
  size_t components = tensor.sizes[2];

  std::stringstream ss;
  ss << "P3" << std::endl;
  ss << width << " " << height << std::endl;
  ss << "255" << std::endl;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for(size_t c = 0; c<components; ++c) {
        float v = tensor.values[x + width * (y + height * c)];
        if (c != 0)
          ss << " ";
        ss << int(255.f * (v - v_min) / (v_max - v_min));
      }
      ss << std::endl;
    }
    ss << std::endl;
  }

  return ss.str();
}

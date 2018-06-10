#include "Image.hpp"
#include "sstream"

std::string image_PGM(const Tensor& tensor) {
  float v_min = tensor.values[0];
  float v_max = tensor.values[0];
  for (const auto& it : tensor.values) {
    v_min = std::min(v_min, it);
    v_max = std::max(v_max, it);
  }

  size_t width = tensor.sizes[0];
  size_t height = tensor.sizes[1];
  size_t num_images = tensor.values.size() / (width * height);
  size_t total_height = (height + 1) * num_images;

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
    for (size_t x = 0; x < width; ++x) {
      ss << 0 << " ";
    }
    ss << std::endl;
  }

  return ss.str();
}

std::string image_PPM(const Tensor& tensor) {
  float v_min = tensor.values[0];
  float v_max = tensor.values[0];
  for (const auto& it : tensor.values) {
    v_min = std::min(v_min, it);
    v_max = std::max(v_max, it);
  }
  float scale = std::max(std::abs(v_min), std::abs(v_max));

  size_t width = tensor.sizes[0];
  size_t height = tensor.sizes[1];
  size_t num_images = tensor.values.size() / (width * height);
  size_t total_height = (height + 1) * num_images;

  std::stringstream ss;
  ss << "P3" << std::endl;
  ss << width << " " << total_height << std::endl;
  ss << "255" << std::endl;

  for (size_t image = 0; image < num_images; ++image) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        float value = tensor.values[x + width * (y + height * image)];
        if (value > 0.f) {
          ss << 0 << " " << 0 << " " << int(255 * value / scale) << " ";
        } else {
          ss << int(-255 * value / scale) << " " << 0 << " " << 0 << " ";
        }
      }
      ss << std::endl;
    }
    for (size_t x = 0; x < width; ++x) {
      ss << 0 << " ";
      ss << 0 << " ";
      ss << 0 << " ";
    }
    ss << std::endl;
  }

  return ss.str();
}

#include "Tensor.hpp"

std::string image_PGM(const Tensor& tensor, float min, float max);
std::string image_PGM(const Tensor& tensor);

std::string image_PPM_centered(const Tensor& tensor);
std::string image_PPM(const Tensor& tensor);
std::string image_PPM(const Tensor& tensor, float vmin, float vmax);

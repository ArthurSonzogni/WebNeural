#include "util.hpp"

size_t Multiply(const std::vector<size_t>& v) {
  size_t ret = 1;
  for(const size_t i : v) {
    ret *= i;
  }
  return ret;
}


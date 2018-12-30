#ifndef WEBNEURAL_ALGORITHM_WCGAN
#define WEBNEURAL_ALGORITHM_WCGAN

#include <functional>
#include <random>
#include "Allocator.hpp"

class WCGAN {
 public:
  // Network description.
  std::vector<size_t> latent_size;
  std::function<Tensor()> input;
  std::function<Node*(Allocator&, Node*)> generator;
  std::function<Node*(Allocator&, Node*)> discriminator;

  // Learning description.
  int batch_size = 32;
  float learning_rate = 0.005f;

  void Init();
  void Train();
  Tensor Generate();

  void SaveToFile(const std::string& filename);
  void LoadFromFile(const std::string& filename);

 private:
  Allocator allocator_;

  Node* generator_input_ = nullptr;
  Node* generator_output_ = nullptr;
  Node* discriminator_input_ = nullptr;
  Node* discriminator_output_ = nullptr;

  std::random_device random;
};

#endif /* end of include guard: WEBNEURAL_ALGORITHM_WCGAN */

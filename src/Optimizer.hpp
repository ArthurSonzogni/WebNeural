#ifndef WEBNEURAL_OPTIMIZER_HPP
#define WEBNEURAL_OPTIMIZER_HPP

#include <cstddef>
#include <vector>
#include "Tensor.hpp"
#include "node/Node.hpp"

class Optimizer {
 public:
  explicit Optimizer(std::vector<Node*> nodes);
  virtual ~Optimizer() = default;

  virtual void Step(size_t batch_size, float lr) = 0;
  void ZeroGrad();

 protected:
  std::vector<Node*> nodes_;
};

class Adam : public Optimizer {
 public:
  Adam(std::vector<Node*> nodes,
       float beta1 = 0.9f,
       float beta2 = 0.99f,
       float epsilon = 1e-8f,
       float weight_decay = 0.f);

  void Step(size_t batch_size, float lr) override;
  void Reset();

 private:
  float beta1_;
  float beta2_;
  float epsilon_;
  float weight_decay_;
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;
  std::vector<size_t> steps_;
};

class SGD : public Optimizer {
 public:
  SGD(std::vector<Node*> nodes, float momentum = 0.f, float weight_decay = 0.f);
  void Step(size_t batch_size, float lr) override;

 private:
  float momentum_;
  float weight_decay_;
  std::vector<Tensor> velocity_;
};

#endif /* end of include guard: WEBNEURAL_OPTIMIZER_HPP */

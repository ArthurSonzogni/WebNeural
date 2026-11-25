#include "Optimizer.hpp"
#include <cmath>

Optimizer::Optimizer(std::vector<Node*> nodes) : nodes_(std::move(nodes)) {}

void Optimizer::ZeroGrad() {
  for (Node* node : nodes_) {
    node->ZeroParamGrad();
  }
}

Adam::Adam(std::vector<Node*> nodes,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay)
    : Optimizer(std::move(nodes)),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay) {
  Reset();
}

void Adam::Reset() {
  m_.clear();
  v_.clear();
  steps_.assign(nodes_.size(), 0);
  m_.reserve(nodes_.size());
  v_.reserve(nodes_.size());
  for (Node* node : nodes_) {
    m_.emplace_back(node->params.sizes);
    v_.emplace_back(node->params.sizes);
  }
}

void Adam::Step(size_t batch_size, float lr) {
  if (batch_size == 0)
    return;

  const float inv_batch = 1.f / static_cast<float>(batch_size);

  for (size_t idx = 0; idx < nodes_.size(); ++idx) {
    Node* node = nodes_[idx];
    const size_t params_size = node->params.values.size();
    if (params_size == 0)
      continue;

    steps_[idx] += 1;
    const float n = static_cast<float>(steps_[idx]);
    const float corr1 = beta1_ > 0.f ? 1.f / (1.f - std::pow(beta1_, n)) : 1.f;
    const float corr2 = 1.f / (1.f - std::pow(beta2_, n));

    float* __restrict__ p_ptr = node->params.values.data();
    float* __restrict__ g_ptr = node->params_grad.values.data();
    float* __restrict__ m_ptr = m_[idx].values.data();
    float* __restrict__ v_ptr = v_[idx].values.data();

    #pragma omp parallel for if(params_size >= 4096)
    for (size_t p = 0; p < params_size; ++p) {
      const float g = g_ptr[p] * inv_batch + weight_decay_ * p_ptr[p];
      g_ptr[p] = 0.f;
      const float m = beta1_ * m_ptr[p] + (1.f - beta1_) * g;
      const float v = beta2_ * v_ptr[p] + (1.f - beta2_) * g * g;
      m_ptr[p] = m;
      v_ptr[p] = v;
      p_ptr[p] -= lr * (m * corr1) / (std::sqrt(v * corr2) + epsilon_);
    }
  }
}

SGD::SGD(std::vector<Node*> nodes, float momentum, float weight_decay)
    : Optimizer(std::move(nodes)),
      momentum_(momentum),
      weight_decay_(weight_decay) {
  velocity_.reserve(nodes_.size());
  for (Node* node : nodes_) {
    velocity_.emplace_back(node->params.sizes);
  }
}

void SGD::Step(size_t batch_size, float lr) {
  if (batch_size == 0)
    return;

  const float inv_batch = 1.f / static_cast<float>(batch_size);
  for (size_t idx = 0; idx < nodes_.size(); ++idx) {
    Node* node = nodes_[idx];
    const size_t params_size = node->params.values.size();
    if (params_size == 0)
      continue;

    float* __restrict__ p_ptr = node->params.values.data();
    float* __restrict__ g_ptr = node->params_grad.values.data();
    float* __restrict__ vel_ptr = velocity_[idx].values.data();

    #pragma omp parallel for if(params_size >= 4096)
    for (size_t p = 0; p < params_size; ++p) {
      const float g = g_ptr[p] * inv_batch + weight_decay_ * p_ptr[p];
      g_ptr[p] = 0.f;
      if (momentum_ > 0.f) {
        vel_ptr[p] = momentum_ * vel_ptr[p] + g;
        p_ptr[p] -= lr * vel_ptr[p];
      } else {
        p_ptr[p] -= lr * g;
      }
    }
  }
}

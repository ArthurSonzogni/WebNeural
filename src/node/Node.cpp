#include <iostream>
#include "Node.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

static constexpr float ADAM_b1 = 0.9f;
static constexpr float ADAM_b2 = 0.99f;
static constexpr float ADAM_epsilon = 1e-8f;
constexpr size_t Node::T;

void Node::Update(size_t batch_size, float lambda) {
  InitIfNeeded();
  if (batch_size == 0)
    return;

  const size_t ps_count = ps_batches_used > 0 ? ps_batches_used : batch_size;
  ps_batches_used = 0;

  if (locked) {
    for (size_t batch = 0; batch < ps_count; ++batch) {
      params_sensitivity[batch].Fill(0.f);
    }
    return;
  }
  n += 1;

  const size_t params_size = params.values.size();
  if (params_size == 0)
    return;

  const float inv_batch = 1.f / static_cast<float>(batch_size);
  const float corr1 = 1.f / (1.f - std::pow(ADAM_b1, static_cast<float>(n)));
  const float corr2 = 1.f / (1.f - std::pow(ADAM_b2, static_cast<float>(n)));
  float* __restrict__ p_ptr = params.values.data();
  float* __restrict__ m_ptr = momentum.values.data();
  float* __restrict__ ssg_ptr = smoothed_squared_gradient.values.data();

  if (ps_count == 1) {
    float* __restrict__ ps_ptr = params_sensitivity[0].values.data();
    #pragma omp parallel for if(params_size >= 4096)
    for (size_t p = 0; p < params_size; ++p) {
      const float g = ps_ptr[p] * inv_batch;
      ps_ptr[p] = 0.f;
      const float m = ADAM_b1 * m_ptr[p] + (1.f - ADAM_b1) * g;
      const float v = ADAM_b2 * ssg_ptr[p] + (1.f - ADAM_b2) * g * g;
      m_ptr[p] = m;
      ssg_ptr[p] = v;
      p_ptr[p] -= lambda * (m * corr1) / (std::sqrt(v * corr2) + ADAM_epsilon);
    }
    return;
  }

  // Gather params_sensitivity into reusable ps_buffer without heap allocation.
  float* __restrict__ ps_buf = ps_buffer.values.data();
  std::fill_n(ps_buf, params_size, 0.f);
  for (size_t batch = 0; batch < ps_count; ++batch) {
    float* __restrict__ src = params_sensitivity[batch].values.data();
    #pragma omp simd
    for (size_t p = 0; p < params_size; ++p) {
      ps_buf[p] += src[p];
      src[p] = 0.f;
    }
  }

  // Full Adam update with bias correction.
  #pragma omp parallel for if(params_size >= 4096)
  for (size_t p = 0; p < params_size; ++p) {
    const float g = ps_buf[p] * inv_batch;
    const float m = ADAM_b1 * m_ptr[p] + (1.f - ADAM_b1) * g;
    const float v = ADAM_b2 * ssg_ptr[p] + (1.f - ADAM_b2) * g * g;
    m_ptr[p] = m;
    ssg_ptr[p] = v;
    p_ptr[p] -= lambda * (m * corr1) / (std::sqrt(v * corr2) + ADAM_epsilon);
  }
}

// static
void Node::Link(Node* previous, Node* next) {
  // Make them refer to each other.
  previous->next = next;
  next->previous = previous;

  // Resize in case they are null.
  previous->output.resize(T);
  previous->output_sensitivity.resize(T);
  next->input.resize(T);
  next->input_sensitivity.resize(T);

  // Link for each batch.
  for (size_t batch = 0; batch < T; ++batch) {
    next->input[batch] = &(previous->output[batch]);
    previous->output_sensitivity[batch] = &(next->input_sensitivity[batch]);
  }
}

void Node::InitInternalSensitivity() {
  input_sensitivity = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  params_sensitivity = std::vector<Tensor>(T, Tensor(params.sizes));
  output_sensitivity = std::vector<Tensor*>(T, nullptr);
  Link(previous);
}

void Node::Link(Node* previous) {
  Link(previous, this);
}

void Node::InitIfNeeded() {
  if (initiated)
    return;
  initiated = true;
  momentum = Tensor(params.sizes);
  momentum.Fill(0.f);
  smoothed_squared_gradient = Tensor(params.sizes);
  smoothed_squared_gradient.Fill(0.f);
  ps_buffer = Tensor(params.sizes);
  ps_buffer.Fill(0.f);
}

void Node::Clear() {
  for (size_t batch = 0; batch < T; ++batch) {
    params_sensitivity[batch].Fill(0.f);
  }
  //momentum.Fill(0.f);
  //smoothed_squared_gradient.Fill(0.f);
}

void Range::Apply(const std::function<void(Node*)>& fun) {
  Node* node = first;
  while (true) {
    fun(node);
    if (node == last)
      break;
    node = node->next;
  }
}

void Range::Apply(void (Node::*f)()) {
  Node* node = first;
  while (true) {
    (node->*f)();
    if (node == last)
      break;
    node = node->next;
  }
}

void ReverseRange::Apply(const std::function<void(Node*)>& fun) {
  Node* node = first;
  while (true) {
    fun(node);
    if (node == last)
      break;
    node = node->previous;
  }
}

void ReverseRange::Apply(void (Node::*f)()) {
  Node* node = first;
  while (true) {
    (node->*f)();
    if (node == last)
      break;
    node = node->previous;
  }
}

void Node::SerializeParams(std::vector<float>& value) {
  InitIfNeeded();

  for (auto& p : params.values)
    value.push_back(p);
  for (auto& p : smoothed_squared_gradient.values)
    value.push_back(p);
  for (auto& p : momentum.values)
    value.push_back(p);
}

void Node::DeserializeParams(const std::vector<float>& value, size_t& index) {
  InitIfNeeded();
  const size_t required = params.values.size() +
                          smoothed_squared_gradient.values.size() +
                          momentum.values.size();
  if (index > value.size() || value.size() - index < required) {
    throw std::out_of_range("Serialized params buffer is too small");
  }
  for (auto& p : params.values)
    p = value[index++];
  for (auto& p : smoothed_squared_gradient.values)
    p = value[index++];
  for (auto& p : momentum.values)
    p = value[index++];
}

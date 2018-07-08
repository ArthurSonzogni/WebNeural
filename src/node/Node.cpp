#include <iostream>
#include "Node.hpp"
#include <algorithm>
#include <cmath>

static constexpr float RMS_decay = 0.95f;
static constexpr float RMS_epsilon = 1e-8f;
static constexpr float Momentum_decay = 0.9f;
static constexpr float Momentum_power = 0.1f;
constexpr size_t Node::T;

void Node::Update(size_t batch_size, float lambda) {
  InitIfNeeded();
  if (locked)
    return;

  // Gather params_sensitivity
  Tensor& PS = params_sensitivity[0];
  for(size_t batch = 1; batch<batch_size; ++batch) {
    PS += params_sensitivity[batch];
  }

  // I am using RMSprop.
  size_t params_size = params.values.size();
  for (size_t p = 0; p < params_size; ++p) {
    smoothed_squared_gradient[p] =
        RMS_decay * smoothed_squared_gradient[p] +
        (1.f - RMS_decay) * PS[p] * PS[p];

    momentum[p] =
        momentum[p] * Momentum_decay + PS[p] * Momentum_power;

    params[p] += lambda * (PS[p] + momentum[p]);
                 std::sqrt(smoothed_squared_gradient[p] + RMS_epsilon);
  }

  for(size_t batch = 0; batch < batch_size; ++batch) {
    params_sensitivity[batch].Fill(0.f);
  }
}


// static
void Node::Link(Node& previous, Node& next) {
  previous.next = &(next);
  next.previous = &(previous);

  // Resize in case they are null.
  next.input.resize(T);
  next.input_sensitivity.resize(T);
  previous.output.resize(T);
  previous.output_sensitivity.resize(T);

  for(size_t batch = 0; batch < T; ++batch) {
    next.input[batch] = &(previous.output[batch]);
    previous.output_sensitivity[batch] = &(next.input_sensitivity[batch]);
  }
}

void Node::InitInternalSensitivity() {
  input_sensitivity = std::vector<Tensor>(T, Tensor(input[0]->sizes));
  params_sensitivity = std::vector<Tensor>(T, Tensor(params.sizes));
  output_sensitivity = std::vector<Tensor*>(T, nullptr);
  Link(*previous);
}

void Node::Link(Node& previous) {
  Link(previous, *this);
}

void Node::InitIfNeeded() {
  if (initiated)
    return;
  initiated = true;
  momentum = Tensor(params.sizes);
  momentum.Fill(0.f);
  smoothed_squared_gradient = Tensor(params.sizes);
  smoothed_squared_gradient.Fill(1.f);
}

void Node::Clear() {
  for(size_t batch = 0; batch < T; ++batch) {
    params_sensitivity[batch].Fill(0.f);
  }
  momentum.Fill(0.f);
  smoothed_squared_gradient.Fill(1.f);
}

void Range::Apply(const std::function<void(Node&)>& fun) {
  Node* node = &first;
  while (true) {
    fun(*node);
    if (node == &last)
      break;
    node = node->next;
  }
}

void Range::Apply(void (Node::*f)()) {
  Node* node = &first;
  while (true) {
    (node->*f)();
    if (node == &last)
      break;
    node = node->next;
  }
}

void ReverseRange::Apply(const std::function<void(Node&)>& fun) {
  Node* node = &first;
  while (true) {
    fun(*node);
    if (node == &last)
      break;
    node = node->previous;
  }
}

void ReverseRange::Apply(void (Node::*f)()) {
  Node* node = &first;
  while (true) {
    (node->*f)();
    if (node == &last)
      break;
    node = node->previous;
  }
}

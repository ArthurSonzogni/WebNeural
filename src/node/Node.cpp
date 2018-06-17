#include "Node.hpp"
#include <algorithm>
#include <cmath>

static constexpr float RMS_decay = 0.95f;
static constexpr float RMS_epsilon = 1e-8f;
static constexpr float Momentum_decay = 0.9f;
static constexpr float Momentum_power = 0.2f;

void Node::Update(float lambda) {
  InitIfNeeded();
  if (locked)
    return;

  // I am using RMSprop.
  size_t params_size = params.values.size();
  for (size_t p = 0; p < params_size; ++p) {
    smoothed_squared_gradient[p] =
        RMS_decay * smoothed_squared_gradient[p] +
        (1.f - RMS_decay) * params_sensitivity[p] * params_sensitivity[p];

    momentum[p] =
        momentum[p] * Momentum_decay + params_sensitivity[p] * Momentum_power;

    params[p] += lambda * (params_sensitivity[p] + momentum[p]);
                 std::sqrt(smoothed_squared_gradient[p] + RMS_epsilon);
  }

  params_sensitivity.Fill(0.f);
}


// static
void Node::Link(Node& previous, Node& next) {
  previous.next = &(next);
  next.previous = &(previous);

  next.input = &(previous.output);
  previous.output_sensitivity = &(next.input_sensitivity);
}

void Node::Link(Node& previous) {
  Link(previous, *this);
}

void Node::InitIfNeeded() {
  if (initiated)
    return;
  initiated = true;
  Clear();
}

void Node::Clear() {
  momentum = Tensor(params.sizes);
  momentum.Fill(0.f);
  smoothed_squared_gradient = Tensor(params.sizes);
  smoothed_squared_gradient.Fill(1.f);
}

void Clear();

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

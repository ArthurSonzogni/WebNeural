#include <iostream>
#include "Node.hpp"
#include <algorithm>
#include <cmath>

static constexpr float ADAM_b1 = 0.f;
static constexpr float ADAM_b2 = 0.9f;
static constexpr float ADAM_epsilon = 1e-4f;
constexpr size_t Node::T;

void Node::Update(size_t batch_size, float lambda) {
  InitIfNeeded();
  if (locked)
    return;
  n += 1;

  // Gather params_sensitivity
  Tensor PS = Tensor(params_sensitivity[0].sizes);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    PS += params_sensitivity[batch];
    params_sensitivity[batch].Fill(0.f);
  }

  // I am using ADAM optimizer.
  const size_t params_size = params.values.size();
  for (size_t p = 0; p < params_size; ++p) {
    // Update first and second order estimate.
    //momentum[p] = ADAM_b1 * momentum[p] + (1.f - ADAM_b1) * PS[p];

    smoothed_squared_gradient[p] = ADAM_b2 * smoothed_squared_gradient[p] +
                                   (1.f - ADAM_b2) * PS[p] * PS[p];

    // Correct the bias.
    //const float vt = momentum[p];// * (1.f - pow(ADAM_b1, n));
    const float mt = smoothed_squared_gradient[p];// * (1.f - pow(ADAM_b2, n));

    params[p] -= lambda * PS[p] / (std::sqrt(mt) + ADAM_epsilon);
    


    //if (std::isnan(params[p]))
      //params[p] = 0.f;
    //if (std::isnan(momentum[p]))
      //momentum[p] = 0.f;
    //if (std::isnan(smoothed_squared_gradient[p]))
      //smoothed_squared_gradient[p] = 0.f;
    //if (std::isnan(PS[p]))
      //PS[p] = 0.f;
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
  for (auto& p : params.values)
    p = value[index++];
  for (auto& p : smoothed_squared_gradient.values)
    p = value[index++];
  for (auto& p : momentum.values)
    p = value[index++];
}

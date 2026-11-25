#include "Node.hpp"
#include <algorithm>
#include <stdexcept>

void Node::AddInput(Node* input_node) {
  inputs.push_back(input_node);
  if (input_node) {
    input_node->outputs.push_back(this);
  }
}

void Node::SetInput(size_t index, Node* new_input) {
  if (index >= inputs.size()) {
    throw std::out_of_range("Node::SetInput index out of range");
  }
  Node* old_input = inputs[index];
  if (old_input) {
    auto& outs = old_input->outputs;
    outs.erase(std::remove(outs.begin(), outs.end(), this), outs.end());
  }
  inputs[index] = new_input;
  if (new_input) {
    new_input->outputs.push_back(this);
  }
}

void Node::InitBuffers(const std::vector<size_t>& output_sizes) {
  output = {Tensor(output_sizes)};
  grad = {Tensor(output_sizes)};
  if (!params.values.empty()) {
    params_grad = Tensor(params.sizes);
  }
}

void Node::EnsureBatchSize(size_t batch_size) {
  if (output.empty())
    return;
  const std::vector<size_t> sizes = output[0].sizes;
  if (output.size() < batch_size) {
    output.reserve(batch_size);
    while (output.size() < batch_size) {
      output.emplace_back(sizes);
    }
  }
  if (grad.size() < batch_size) {
    grad.reserve(batch_size);
    while (grad.size() < batch_size) {
      grad.emplace_back(sizes);
    }
  }
}

void Node::ZeroGrad(size_t batch_size) {
  EnsureBatchSize(batch_size);
  for (size_t b = 0; b < batch_size; ++b) {
    grad[b].Fill(0.f);
  }
}

void Node::ZeroParamGrad() {
  if (!params_grad.values.empty()) {
    params_grad.Fill(0.f);
  }
}

size_t Node::SerializedWeightCount() const {
  return params.values.size();
}

void Node::SerializeWeights(std::vector<float>& value) const {
  value.insert(value.end(), params.values.begin(), params.values.end());
}

void Node::DeserializeWeights(const std::vector<float>& value, size_t& index) {
  const size_t required = params.values.size();
  if (index > value.size() || value.size() - index < required) {
    throw std::out_of_range("Serialized weights buffer is too small");
  }
  for (auto& p : params.values) {
    p = value[index++];
  }
}

#include "Graph.hpp"
#include <fstream>
#include <functional>
#include <stdexcept>
#include <unordered_set>

Graph::Graph(Node* output) {
  if (output) {
    Build({output});
  }
}

Graph::Graph(const std::vector<Node*>& outputs) {
  Build(outputs);
}

void Graph::Build(const std::vector<Node*>& outputs) {
  order_.clear();
  std::unordered_set<Node*> visited;
  std::unordered_set<Node*> visiting;

  std::function<void(Node*)> dfs = [&](Node* node) {
    if (!node || visited.count(node))
      return;
    if (visiting.count(node)) {
      throw std::runtime_error("Cycle detected in computation graph");
    }
    visiting.insert(node);
    for (Node* in : node->inputs) {
      dfs(in);
    }
    visiting.erase(node);
    visited.insert(node);
    order_.push_back(node);
  };

  for (Node* out : outputs) {
    dfs(out);
  }
}

void Graph::Forward(size_t batch_size, bool training) {
  for (Node* node : order_) {
    node->EnsureBatchSize(batch_size);
    node->training = training;
    node->Forward(batch_size);
  }
}

void Graph::ForwardAfter(Node* start, size_t batch_size, bool training) {
  bool started = (start == nullptr);
  for (Node* node : order_) {
    if (!started) {
      if (node == start) {
        started = true;
      }
      continue;
    }
    node->EnsureBatchSize(batch_size);
    node->training = training;
    node->Forward(batch_size);
  }
}

void Graph::ZeroGrad(size_t batch_size) {
  for (Node* node : order_) {
    node->ZeroGrad(batch_size);
    node->ZeroParamGrad();
  }
}

void Graph::Backward(size_t batch_size) {
  for (size_t i = order_.size(); i > 0; --i) {
    order_[i - 1]->Backward(batch_size);
  }
}

std::vector<Node*> Graph::TrainableNodes() const {
  std::vector<Node*> result;
  for (Node* node : order_) {
    if (!node->params.values.empty()) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<float> Graph::SerializeWeights() const {
  std::vector<float> data;
  for (Node* node : order_) {
    node->SerializeWeights(data);
  }
  return data;
}

void Graph::DeserializeWeights(const std::vector<float>& data) {
  size_t total = 0;
  for (Node* node : order_) {
    total += node->SerializedWeightCount();
  }
  if (data.size() != total) {
    throw std::out_of_range("Serialized weights size mismatch");
  }
  size_t index = 0;
  for (Node* node : order_) {
    node->DeserializeWeights(data, index);
  }
}

static std::ifstream::pos_type GraphFileSize(const std::string& filename) {
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

void Graph::SaveWeightsToFile(const std::string& filename) const {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
    return;
  auto data = SerializeWeights();
  if (!data.empty()) {
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(float));
  }
}

void Graph::LoadWeightsFromFile(const std::string& filename) {
  auto file_size = GraphFileSize(filename);
  if (file_size <= 0)
    return;
  const size_t num_floats = static_cast<size_t>(file_size) / sizeof(float);
  if (num_floats == 0)
    return;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    return;
  std::vector<float> data(num_floats);
  file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
  DeserializeWeights(data);
}

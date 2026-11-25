#ifndef WEBNEURAL_GRAPH_HPP
#define WEBNEURAL_GRAPH_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "node/Node.hpp"

class Graph {
 public:
  Graph() = default;
  explicit Graph(Node* output);
  explicit Graph(const std::vector<Node*>& outputs);

  void Build(const std::vector<Node*>& outputs);

  void Forward(size_t batch_size, bool training = true);
  // Forward only nodes strictly after `start` in topological order (e.g.
  // running only the decoder half of an autoencoder from a latent node).
  void ForwardAfter(Node* start, size_t batch_size, bool training = false);
  void ZeroGrad(size_t batch_size);
  void Backward(size_t batch_size);

  const std::vector<Node*>& Nodes() const { return order_; }
  std::vector<Node*> TrainableNodes() const;

  // Compact weight-only serialization (auto-detects legacy format on load).
  std::vector<float> SerializeWeights() const;
  void DeserializeWeights(const std::vector<float>& data);
  void SaveWeightsToFile(const std::string& filename) const;
  void LoadWeightsFromFile(const std::string& filename);

 private:
  std::vector<Node*> order_;
};

#endif /* end of include guard: WEBNEURAL_GRAPH_HPP */

#ifndef NODE_H
#define NODE_H

#include <functional>
#include "Tensor.hpp"

class Node {
 public:
  static constexpr size_t T = 64;

  Node() = default;
  virtual ~Node() = default;

  // Node internal state.
  Tensor params;

  // Forward step
  std::vector<Tensor*> input;
  std::vector<Tensor> output;

  // Backward
  std::vector<Tensor> input_sensitivity;
  std::vector<Tensor> params_sensitivity;
  std::vector<Tensor*> output_sensitivity;

  virtual void Forward(size_t batch_size) = 0;
  virtual void Backward(size_t batch_size) = 0;
  void Update(size_t batch_size, float lambda);

  Node* next = nullptr;
  Node* previous = nullptr;

  bool locked = false;  // Do not update.
  void Lock() { locked = true; }
  void Unlock() { locked = false; }
  void Clear();

  static void Link(Node* previous, Node* next);

  void SerializeParams(std::vector<float>& value);
  void DeserializeParams(const std::vector<float>& value, size_t& index);

 protected:
  void Link(Node* previous);
  void InitInternalSensitivity();

 private:
  void InitIfNeeded();
  bool initiated = false;
  Tensor smoothed_squared_gradient;
  Tensor momentum;

  size_t n = 0;
};

class Range {
 public:
  Range(Node* first, Node* last) : first(first), last(last) {}
  void Apply(const std::function<void(Node*)>& f);
  void Apply(void (Node::*f)());

 private:
  Node* first;
  Node* last;
};

class ReverseRange {
 public:
  ReverseRange(Node* first, Node* last) : first(first), last(last) {}
  void Apply(const std::function<void(Node*)>& f);
  void Apply(void (Node::*f)());

 private:
  Node* first;
  Node* last;
};

#endif /* end of include guard: NODE_H */

#ifndef NODE_H
#define NODE_H

#include <functional>
#include "Tensor.hpp"

class Node {
 public:
  // Forward step
  Tensor* input = nullptr;
  Tensor params;
  Tensor output;

  // Backward
  Tensor input_sensitivity;
  Tensor params_sensitivity;
  Tensor* output_sensitivity = nullptr;

  virtual void Forward() = 0;
  virtual void Backward() = 0;
  void Update(float lambda);

  Node* next = nullptr;
  Node* previous = nullptr;

  bool locked = false;  // Do not update.
  void Lock() { locked = true; }
  void Unlock() { locked = false; }
  void Clear();

  static void Link(Node& previous, Node& next);

 protected:
  void Link(Node& previous);

 private:
  void InitIfNeeded();
  bool initiated = false;
  Tensor smoothed_squared_gradient;
  Tensor momentum;
};

class Range {
 public:
  Range(Node& first, Node& last) : first(first), last(last) {}
  void Apply(const std::function<void(Node&)>& f);
  void Apply(void (Node::*f)());

 private:
  Node& first;
  Node& last;
};

class ReverseRange {
 public:
  ReverseRange(Node& first, Node& last) : first(first), last(last) {}
  void Apply(const std::function<void(Node&)>& f);
  void Apply(void (Node::*f)());

 private:
  Node& first;
  Node& last;
};

#endif /* end of include guard: NODE_H */

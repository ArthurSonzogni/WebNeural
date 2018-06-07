#include "Optimizer.hpp"

#include <iostream>

Optimizer::Optimizer(Node& node,
                     size_t batch_size,
                     const std::vector<Example>& examples)
    : examples(examples), batch_size(batch_size) {
  Node* op = &node;
  while (op) {
    std::cout << __func__ << std::endl;
    nodes.push_back(op);
    op = op->input ? op->input->producer : nullptr;
  }
}

void Optimizer::Train(float lambda, size_t iterations) {
  for (size_t i = 0; i < iterations; ++i) {
    ++iteration;
    iteration %= examples.size();

    // Feed the neural network.
    nodes.back()->output = examples[iteration].input;

    // Make a prediction.
    Forward();

    // Compute the error.
    Tensor error = examples[iteration].output - nodes.front()->output;
    nodes.front()->output_sensitivity = &error;

    // Compute the sensitivity.
    Backward();

    // Update the network once in a while.
    if (i % batch_size == 0) {
      Update(lambda / batch_size);
      nodes.front()->output_sensitivity = &error;
    }
  }
}

Tensor Optimizer::Predict(const Tensor& input) {
  // Feed the neural network.
  nodes.back()->output = input;

  // Make a prediction.
  Forward();

  return nodes.front()->output;
}

float Optimizer::Error() {
  float error = 0;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);
    error += (output - example.output).Error();
  }
  error /= float(examples.size());
  return error;
}

float Optimizer::ErrorInteger() {
  float error = 0;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);

    if (output.ArgMax() != example.output.ArgMax())
      error += 1.f;
  }
  error /= float(examples.size());
  return error;
}

void Optimizer::Update(float lambda) {
  for (size_t i = 0; i < nodes.size() - 1; ++i) {
    nodes[i]->Update(lambda);
  }
}

void Optimizer::Forward() {
  for (size_t i = nodes.size() - 2; i < nodes.size(); --i) {
    nodes[i]->Forward();
  }
}

void Optimizer::Backward() {
  for (size_t i = 0; i < nodes.size() - 1; ++i) {
    nodes[i]->Backward();
  }
}

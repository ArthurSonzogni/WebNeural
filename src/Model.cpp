#include "Model.hpp"

#include <iostream>
#include <algorithm>
#include <sstream>

Model::Model(Node& input, Node& output, const std::vector<Example>& examples)
    : input(input), output(output), examples(examples) {
}

void Model::Train(float lambda, size_t iterations) {
  float sum_error = 0.f;
  for (size_t i = 0; i < iterations; ++i) {
    ++iteration;
    iteration %= examples.size();

    // Feed the neural network.
    input.params = examples[iteration].input;

    // Make a prediction.
    Range(input, output).Apply(&Node::Forward);

    // Compute the error.
    Tensor error = examples[iteration].output - output.output;
    output.output_sensitivity = &error;

    // Compute the sensitivity.
    ReverseRange(output, *input.next).Apply(&Node::Backward);

    // Update the network once in a while.
    if (i % batch_size == 0) {
      Range(*input.next, output).Apply([lambda](Node& node) {
        node.Update(lambda);
      });
    }

    sum_error += error.Error();
  }

  last_error = sum_error / iterations;
}

float Model::OptimizeInput(const Tensor& output_target, float lambda) {
  // Make a prediction.
  Range(input, output).Apply(&Node::Forward);

  // Compute the error.
  Tensor error = output_target - output.output;
  output.output_sensitivity = &error;

  // Compute the sensitivity.
  ReverseRange(output, *input.next).Apply(&Node::Backward);

  // Update the input.
  input.Update(lambda);

  return error.Error();
}

Tensor Model::Predict(const Tensor& input_value) {
  // Feed the neural network.
  input.params = input_value;

  // Make a prediction.
  Range(input, output).Apply(&Node::Forward);

  return output.output;
}

float Model::Error() {
  float error = 0;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);
    error += (output - example.output).Error();
  }
  error /= float(examples.size());
  return error;
}

float Model::ErrorInteger() {
  float error = 0.f;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);

    if (output.ArgMax() != example.output.ArgMax())
      error += 1.f;
  }
  error /= float(examples.size());
  return error;
}

float Model::LastError() {
  return last_error;
}

std::string Model::SerializeParams() {
  std::stringstream ss;
  Range(input, output).Apply([&ss](Node& node) {
    for (auto& p : node.params.values)
      ss << p << ' ';
  });
  return ss.str();
}

void Model::DeserializeParams(const std::string& value) {
  std::stringstream ss(value);
  Range(input, output).Apply([&ss](Node& node) {
    for (auto& p : node.params.values)
      ss >> p;
  });
}

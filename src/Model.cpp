#include "Model.hpp"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

Model::Model(Node& input, Node& output, const std::vector<Example>& examples)
    : input(input), output(output), examples(examples) {
}

Model::Model(Node& input, Node& output) : Model(input, output, {}) {}

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

std::vector<float> Model::SerializeParams() {
  std::vector<float> ret;
  Range(input, output).Apply([&ret](Node& node) {
    for (auto& p : node.params.values)
      ret.push_back(p);
  });
  return ret;
}

void Model::DeserializeParams(const std::vector<float>& value) {
  size_t i = 0;
  Range(input, output).Apply([&value,&i](Node& node) {
    for (auto& p : node.params.values)
      p = value[i++];
  });
}
static std::ifstream::pos_type FileSize(const std::string& filename) {
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

void Model::SerializeParamsToFile(const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  auto data = SerializeParams();
  file.write((const char*)(&data[0]), data.size() * sizeof(float));
}

void Model::DeserializeParamsFromFile(const std::string& filename) {
  auto file_size = FileSize(filename);
  if (file_size > 0) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<float> data(file_size / sizeof(float));
    file.read((char*)(&data[0]), file_size);
    DeserializeParams(data);
  }
}

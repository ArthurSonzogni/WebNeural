#include "Model.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>

Model::Model(Node& input, Node& output, const std::vector<Example>& examples)
    : input(input), output(output), examples(examples) {
}

Model::Model(Node& input, Node& output) : Model(input, output, {}) {}

void Model::Train(float lambda, size_t iterations) {
  const float real_lambda = lambda / batch_size;

  std::vector<Tensor> error(Node::T, Tensor(output.output[0].sizes));
  float sum_error = 0.f;
  for (size_t i = 0; i < iterations;) {
    size_t elements = std::min(Node::T, iterations-i);

    // Feed the neural network.
    for (size_t t = 0; t < elements; ++t) {
      input.output[t] = examples[(i + t) % examples.size()].input;
    }

    // Make a prediction.
    Range(*input.next, output).Apply([&](Node& node) { node.Forward(elements); });

    // Compute the error.
    for (size_t t = 0; t < elements; ++t) {
      error[t] = examples[(i + t) % examples.size()].output - output.output[t];
      sum_error += error[t].Error();
      output.output_sensitivity[t] = &(error[t]);
    }

    // Compute the sensitivity.
    ReverseRange(output, *input.next).Apply([&](Node& node) {
      node.Backward(elements);
    });

    // Update the network once in a while.
    //if (iteration
    Range(*input.next, output).Apply([&](Node& node) {
      node.Update(elements, real_lambda);
    });

    i += elements;
  }

  last_error = sum_error / iterations;
}

Tensor Model::Predict(const Tensor& input_value) {
  // Feed the neural network.
  input.output[0] = input_value;

  // Make a prediction.
  Range(*(input.next), output).Apply([](Node& node) { node.Forward(1); });

  return output.output[0];
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

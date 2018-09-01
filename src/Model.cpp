#include "Model.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>

Model::Model(Node* input, Node* output, const std::vector<Example>& examples)
    : input(input), output(output), examples(examples) {}

Model::Model(Node* input, Node* output) : Model(input, output, {}) {}

void Model::Train(float lambda, size_t iterations) {
  std::vector<Tensor> error_sensitivity(Node::T,
                                        Tensor(output->output[0].sizes));
  float sum_error = 0.f;
  for (size_t i = 0; i < iterations;) {
    size_t elements = std::min(Node::T, iterations - i);

    // Feed the neural network.
    for (size_t t = 0; t < elements; ++t) {
      input->output[t] = examples[(iteration + t) % examples.size()].input;
    }

    // Make a prediction.
    Range(input->next, output).Apply([&](Node* node) {
      node->Forward(elements);
    });

    // Compute the error.
    for (size_t t = 0; t < elements; ++t) {
      const Tensor& target = examples[(iteration + t) % examples.size()].output;
      const Tensor& current = output->output[t];
      float current_error;
      loss_function(target, current, &current_error, &error_sensitivity[t]);
      sum_error += current_error;
      output->output_sensitivity[t] = &(error_sensitivity[t]);
    }

    // Compute the sensitivity.
    ReverseRange(output, input->next).Apply([&](Node* node) {
      node->Backward(elements);
    });

    // Update the network.
    Range(input->next, output).Apply([&](Node* node) {
      node->Update(elements, lambda);
    });

    post_update_function(this);

    i += elements;
    iteration += elements;
  }

  last_error = sum_error / iterations;
}

Tensor Model::Predict(const Tensor& input_value) {
  // Feed the neural network.
  input->output[0] = input_value;

  // Make a prediction.
  Range(input->next, output).Apply([](Node* node) { node->Forward(1); });

  return output->output[0];
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
  size_t error = 0.f;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);

    if (output.ArgMax() != example.output.ArgMax())
      error++;
  }
  return error / float(examples.size());
}

float Model::LastError() {
  return last_error;
}

std::vector<float> Model::SerializeParams() {
  std::vector<float> ret;
  Range(input, output).Apply([&ret](Node* node) {
    for (auto& p : node->params.values)
      ret.push_back(p);
  });
  return ret;
}

void Model::DeserializeParams(const std::vector<float>& value) {
  size_t i = 0;
  Range(input, output).Apply([&value, &i](Node* node) {
    for (auto& p : node->params.values)
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
  
void Model::PrintGradient() {
  Range(input->next, output).Apply([&](Node* node) {
    float sum_X = 0.f;
    float sum_1 = 0.f;
    for(auto& v : node->params.values) {
      sum_X += v*v;
      sum_1 += 1.0;
    }
    std::cerr << "Layer = " << std::sqrt(sum_X / sum_1) << std::endl;
  });
}

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
  if (examples.empty() || iterations == 0) {
    last_error = 0.f;
    return;
  }

  Range(input->next, output).Apply([](Node* node) { node->training = true; });

  const size_t max_batch = std::min(Node::T, std::max<size_t>(1, batch_size));
  std::vector<Tensor> error_sensitivity(Node::T,
                                        Tensor(output->output[0].sizes));
  float sum_error = 0.f;
  for (size_t i = 0; i < iterations;) {
    size_t elements = std::min(max_batch, iterations - i);

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
  Range(input->next, output).Apply([](Node* node) {
    node->training = false;
    node->Forward(1);
  });

  return output->output[0];
}

float Model::Error() {
  if (examples.empty())
    return 0.f;
  float error = 0;
  for (auto& example : examples) {
    Tensor output = Predict(example.input);
    error += (output - example.output).Error();
  }
  error /= float(examples.size());
  return error;
}

float Model::ErrorInteger() {
  if (examples.empty())
    return 0.f;
  size_t error = 0;
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
  std::vector<float> value;
  Range(input, output).Apply([&value](Node* node) {
    node->SerializeParams(value);
  });
  return value;
}

void Model::DeserializeParams(const std::vector<float>& value) {
  size_t index = 0;
  Range(input, output).Apply([&value, &index](Node* node) {
    node->DeserializeParams(value, index);
  });
}

static std::ifstream::pos_type FileSize(const std::string& filename) {
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

void Model::SerializeParamsToFile(const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
    return;
  auto data = SerializeParams();
  if (!data.empty()) {
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(float));
  }
}

void Model::DeserializeParamsFromFile(const std::string& filename) {
  auto file_size = FileSize(filename);
  if (file_size > 0) {
    const size_t num_floats = static_cast<size_t>(file_size) / sizeof(float);
    if (num_floats == 0)
      return;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
      return;
    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
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

#include "Model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

Model::Model(Node* input, Node* output, const std::vector<Example>& examples)
    : input(input),
      output(output),
      graph(output),
      optimizer(graph.TrainableNodes()),
      examples(examples) {}

Model::Model(Node* input, Node* output) : Model(input, output, {}) {}

void Model::Train(float lambda, size_t iterations) {
  if (examples.empty() || iterations == 0) {
    last_error = 0.f;
    return;
  }

  const size_t max_batch = std::max<size_t>(1, batch_size);
  float sum_error = 0.f;
  for (size_t i = 0; i < iterations;) {
    size_t elements = std::min(max_batch, iterations - i);

    // Feed the neural network.
    input->EnsureBatchSize(elements);
    for (size_t t = 0; t < elements; ++t) {
      input->output[t] = examples[(iteration + t) % examples.size()].input;
    }

    // Forward pass.
    graph.Forward(elements, true);

    // Clear gradients and seed output gradient from loss.
    graph.ZeroGrad(elements);
    for (size_t t = 0; t < elements; ++t) {
      const Tensor& target = examples[(iteration + t) % examples.size()].output;
      const Tensor& current = output->output[t];
      float current_error = 0.f;
      loss_function(target, current, &current_error, &output->grad[t]);
      sum_error += current_error;
    }

    // Backward pass and optimizer step.
    graph.Backward(elements);
    optimizer.Step(elements, lambda);

    post_update_function(this);

    i += elements;
    iteration += elements;
  }

  last_error = sum_error / static_cast<float>(iterations);
}

Tensor Model::Predict(const Tensor& input_value) {
  input->output[0] = input_value;
  graph.Forward(1, false);
  return output->output[0];
}

float Model::Error() {
  if (examples.empty())
    return 0.f;
  float error = 0.f;
  for (auto& example : examples) {
    Tensor out = Predict(example.input);
    error += (out - example.output).Error();
  }
  error /= static_cast<float>(examples.size());
  return error;
}

float Model::ErrorInteger() {
  if (examples.empty())
    return 0.f;
  size_t error = 0;
  for (auto& example : examples) {
    Tensor out = Predict(example.input);
    if (out.ArgMax() != example.output.ArgMax())
      error++;
  }
  return static_cast<float>(error) / static_cast<float>(examples.size());
}

float Model::LastError() {
  return last_error;
}

std::vector<float> Model::SerializeParams() {
  return graph.SerializeWeights();
}

void Model::DeserializeParams(const std::vector<float>& value) {
  graph.DeserializeWeights(value);
  optimizer.Reset();
}

void Model::SerializeParamsToFile(const std::string& filename) {
  graph.SaveWeightsToFile(filename);
}

void Model::DeserializeParamsFromFile(const std::string& filename) {
  graph.LoadWeightsFromFile(filename);
  optimizer.Reset();
}

void Model::PrintGradient() {
  for (Node* node : graph.TrainableNodes()) {
    float sum_X = 0.f;
    float sum_1 = 0.f;
    for (auto& v : node->params.values) {
      sum_X += v * v;
      sum_1 += 1.f;
    }
    std::cerr << "Layer = " << std::sqrt(sum_X / sum_1) << std::endl;
  }
}

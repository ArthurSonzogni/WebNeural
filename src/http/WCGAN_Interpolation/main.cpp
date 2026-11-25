#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include "Allocator.hpp"
#include "Graph.hpp"
#include "Image.hpp"
#include "LossFunction.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "mnist/mnist_reader.hpp"

std::random_device seed;
std::mt19937 random_generator(seed());

std::vector<Example> GetExamples(const std::vector<std::vector<float>>& input) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    auto input_example = Tensor({29, 29, 1});
    for (size_t y = 0; y < 28; ++y)
      for (size_t x = 0; x < 28; ++x) {
        size_t index = x + 28 * y;
        input_example.at(x, y) = input[i][index] / 255.0f;
      }

    examples.push_back({input_example, input_example});
  }

  return examples;
}

std::vector<uint8_t> input_js(29 * 29 * 4);
std::vector<uint8_t> output_js(29 * 29 * 4);

auto mnist_images =
    mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
        MNIST_DATA_LOCATION);
std::vector<Example> training_set =
#ifndef Web
    GetExamples(mnist_images.training_images);
#else
    GetExamples(mnist_images.test_images);
#endif

Allocator a;

auto res_block = [](Node* X, size_t channels) {
  Node* h = a.Deconvolution2D(X, {3, 3}, channels, 1);
  h = a.BatchNormalization(h);
  h = a.LeakyRelu(h);
  h = a.Convolution2D(h, {3, 3}, channels, 1);
  h = a.BatchNormalization(h);
  return a.LeakyRelu(a.Add(X, h));
};

auto generator = [](Node* X) {
  X = a.Linear(X, {4, 4, 32});
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = a.BilinearUpsampling(X);            // 9x9x32
  X = a.Convolution2D(X, {2, 2}, 24, 1);  // 8x8x24
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = res_block(X, 24);                   // 8x8x24 residual skip (Add)

  X = a.BilinearUpsampling(X);            // 17x17x24
  X = a.Convolution2D(X, {3, 3}, 16, 1);  // 15x15x16
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = a.BilinearUpsampling(X);            // 31x31x16
  X = a.Convolution2D(X, {3, 3}, 1, 1);   // 29x29x1
  X = a.Sigmoid(X);
  return X;
};

auto discriminator = [](Node* X) {
  X = a.Convolution2D(X, {4, 4}, 16, 1);    // 26x26x16
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {4, 4}, 24, 2);    // 12x12x24
  X = a.LeakyRelu(X);

  X = a.Convolution2D(X, {4, 4}, 32, 2);    // 5x5x32
  X = a.LeakyRelu(X);

  X = a.Linear(X, {1, 1, 1});
  X = a.Sigmoid(X);
  return X;
};

auto generator_input = a.Input({10});
auto generator_output = generator(generator_input);

auto discriminator_input = a.Input({29, 29, 1});
auto discriminator_output = discriminator(discriminator_input);
Node* discriminator_first = discriminator_input->outputs[0];

Graph gen_graph(generator_output);
Graph disc_graph(discriminator_output);

// Use DCGAN Adam beta1 = 0.5 for stability (Radford et al., 2015).
Adam opt_g(gen_graph.TrainableNodes(), 0.5f);
Adam opt_d(disc_graph.TrainableNodes(), 0.5f);

void Import(Tensor& tensor, double* input) {
  for (auto& it : tensor.values) {
    it = *(input++);
  }
}

void Export(const Tensor& tensor, uint8_t* output) {
  for (auto& it : tensor.values) {
    float v = std::max(0.f, std::min(1.f, it));
    (*output++) = 255 * v;
    (*output++) = 255 * v;
    (*output++) = 255 * v;
    (*output++) = 255;
  }
}

float learning_rate = 0.0002f;

std::vector<float> SaveModelToVector() {
  discriminator_first->SetInput(0, generator_output);
  Graph full_graph(discriminator_output);
  auto weights = full_graph.SerializeWeights();
  discriminator_first->SetInput(0, discriminator_input);
  return weights;
}
std::vector<float> initial_model = SaveModelToVector();

void LoadModelFromString(const std::vector<float>& data) {
  discriminator_first->SetInput(0, generator_output);
  Graph full_graph(discriminator_output);
  full_graph.DeserializeWeights(data);
  discriminator_first->SetInput(0, discriminator_input);
  opt_g.Reset();
  opt_d.Reset();
}

void Save() {
  discriminator_first->SetInput(0, generator_output);
  Graph full_graph(discriminator_output);
  full_graph.SaveWeightsToFile("save.bin");
  discriminator_first->SetInput(0, discriminator_input);
}

void Load() {
  discriminator_first->SetInput(0, generator_output);
  Graph full_graph(discriminator_output);
  full_graph.LoadWeightsFromFile("save.bin");
  discriminator_first->SetInput(0, discriminator_input);
  opt_g.Reset();
  opt_d.Reset();
  learning_rate = 0.0002f;
}

static float TrainGeneratorBatch(size_t batch, float lr) {
  discriminator_first->SetInput(0, generator_output);
  Graph full_graph(discriminator_output);

  generator_input->EnsureBatchSize(batch);
  for (size_t b = 0; b < batch; ++b) {
    generator_input->output[b] =
        Tensor::Random(generator_input->output[0].sizes);
  }
  full_graph.Forward(batch, true);
  full_graph.ZeroGrad(batch);

  Tensor target_real(1);
  target_real.values = {1.0f};
  float sum_loss = 0.f;
  for (size_t b = 0; b < batch; ++b) {
    float loss = 0.f;
    LossFunction::BinaryCrossEntropy(
        target_real, discriminator_output->output[b], &loss,
        &discriminator_output->grad[b]);
    sum_loss += loss;
  }

  full_graph.Backward(batch);
  opt_g.Step(batch, lr);
  opt_d.ZeroGrad();

  discriminator_first->SetInput(0, discriminator_input);
  return sum_loss / static_cast<float>(batch);
}

static float TrainDiscriminatorBatch(
    size_t half_batch,
    float lr,
    std::uniform_int_distribution<size_t>& random_index) {
  // Generate fake batch (generator is not updated).
  generator_input->EnsureBatchSize(half_batch);
  for (size_t b = 0; b < half_batch; ++b) {
    generator_input->output[b] =
        Tensor::Random(generator_input->output[0].sizes);
  }
  gen_graph.Forward(half_batch, true);

  const size_t total = half_batch * 2;
  Tensor fake_label(1);
  fake_label.values = {0.0f};
  Tensor real_label(1);
  real_label.values = {0.9f};

  discriminator_input->EnsureBatchSize(total);
  for (size_t i = 0; i < total; ++i) {
    if (i % 2 == 0) {
      discriminator_input->output[i] = generator_output->output[i / 2];
    } else {
      discriminator_input->output[i] =
          training_set[random_index(random_generator)].input;
    }
  }

  disc_graph.Forward(total, true);
  disc_graph.ZeroGrad(total);

  float sum_loss = 0.f;
  for (size_t i = 0; i < total; ++i) {
    const Tensor& label = (i % 2 == 0) ? fake_label : real_label;
    float loss = 0.f;
    LossFunction::BinaryCrossEntropy(
        label, discriminator_output->output[i], &loss,
        &discriminator_output->grad[i]);
    sum_loss += loss;
  }

  disc_graph.Backward(total);
  opt_d.Step(total, lr);
  return sum_loss / static_cast<float>(total);
}

extern "C" {

void Predict(double* _input, uint8_t* _output) {
  Import(generator_input->output[0], _input);
  gen_graph.Forward(1, false);
  Export(generator_output->output[0], _output);
}

void LoadPretrainedModel() {
  Load();
}

void ResetModelWeight() {
  LoadModelFromString(initial_model);
  learning_rate = 0.0002f;
}

void Train() {
  std::uniform_int_distribution<size_t> random_index(0, training_set.size() - 1);
  TrainDiscriminatorBatch(4, learning_rate, random_index);
  TrainGeneratorBatch(4, learning_rate);
}

};  // extern "C"

#ifndef Web
int main(int argc, const char* argv[]) {
  bool scratch = false;
  bool preview = false;
  int steps = 4000;
  int n_critic = 1;
  size_t batch = 32;
  float lr = 0.0002f;
  std::string preview_path = "preview.pgm";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--scratch") {
      scratch = true;
    } else if (arg == "--preview") {
      preview = true;
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if (arg == "--critic" && i + 1 < argc) {
      n_critic = std::stoi(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      batch = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (arg == "--lr" && i + 1 < argc) {
      lr = std::stof(argv[++i]);
    } else if (arg == "--preview-path" && i + 1 < argc) {
      preview_path = argv[++i];
    }
  }

  if (!scratch) {
    Load();
  }

  // Fixed latent vectors so the live preview shows the same 64 digits evolving.
  std::vector<Tensor> preview_latents;
  for (int i = 0; i < 64; ++i) {
    preview_latents.push_back(
        Tensor::Random(generator_input->output[0].sizes));
  }

  // Exponential Moving Average (EMA) of generator weights for cleaner inference.
  std::vector<float> ema_gen_params = gen_graph.SerializeWeights();
  constexpr float ema_decay = 0.995f;

  auto write_preview = [&](const std::string& filename) {
    std::vector<Tensor> samples;
    samples.reserve(preview_latents.size());
    for (const auto& z : preview_latents) {
      generator_input->output[0] = z;
      gen_graph.Forward(1, false);
      samples.push_back(generator_output->output[0]);
    }
    const std::string tmp = filename + ".tmp";
    {
      std::ofstream out(tmp);
      out << image_PGM(Tensor::Merge(samples, 8), 0.f, 1.f);
    }
    std::rename(tmp.c_str(), filename.c_str());
  };

  if (preview) {
    write_preview(preview_path);
    std::cout << "Wrote " << preview_path << " (8x8 generated samples)"
              << std::endl;
    return 0;
  }

  std::uniform_int_distribution<size_t> random_index(0, training_set.size() - 1);
  constexpr float kPi = 3.14159265358979323846f;
  const float lr_min = lr * 0.1f;

  for (int step = 1; step <= steps; ++step) {
    // Cosine learning rate decay.
    const float progress =
        static_cast<float>(step - 1) / static_cast<float>(std::max(1, steps - 1));
    const float cur_lr =
        lr_min + 0.5f * (lr - lr_min) * (1.f + std::cos(kPi * progress));

    // 1) Train discriminator n_critic times.
    float d_loss = 0.f;
    for (int c = 0; c < n_critic; ++c) {
      d_loss = TrainDiscriminatorBatch(batch, cur_lr, random_index);
    }

    // 2) Train generator once.
    float g_loss = TrainGeneratorBatch(batch, cur_lr);

    // Update EMA of generator weights.
    auto cur_gen_params = gen_graph.SerializeWeights();
    for (size_t i = 0; i < ema_gen_params.size(); ++i) {
      ema_gen_params[i] =
          ema_decay * ema_gen_params[i] + (1.f - ema_decay) * cur_gen_params[i];
    }

    if (step % 25 == 0 || step == 1 || step == steps) {
      std::cout << "step " << step << "/" << steps
                << " | lr=" << cur_lr
                << " | d_loss=" << d_loss
                << " | g_loss=" << g_loss << std::endl;
      // Preview & save using the smoother EMA generator weights, then restore.
      gen_graph.DeserializeWeights(ema_gen_params);
      Save();
      write_preview(preview_path);
      if (step < steps) {
        gen_graph.DeserializeWeights(cur_gen_params);
      }
    }
  }
  return 0;
}
#endif

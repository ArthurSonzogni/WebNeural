#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include "Allocator.hpp"
#include "Image.hpp"
#include "Model.hpp"
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

auto generator = [](Node* X) {
  X = a.Linear(X, {5, 5, 32});
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {4, 4}, 24, 2);  // 12x12x24
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {4, 4}, 16, 2);  // 26x26x16
  X = a.BatchNormalization(X);
  X = a.LeakyRelu(X);

  X = a.Deconvolution2D(X, {4, 4}, 1, 1);   // 29x29x1
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

Model model(generator_input, generator_output);

float learning_rate = 0.0002f;

std::vector<float> SaveModelToVector() {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  return model.SerializeParams();
}
std::vector<float> initial_model = SaveModelToVector();

void LoadModelFromString(const std::vector<float>& data) {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  model.DeserializeParams(data);
}

void Save() {
  Node::Link(generator_output, discriminator_input->next);
  Model m(generator_input, discriminator_output);
  m.SerializeParamsToFile("save.bin");
  Node::Link(discriminator_input, discriminator_input->next);
}

void Load() {
  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);
  Model model(generator_input, discriminator_output);
  model.DeserializeParamsFromFile("save.bin");
  Node::Link(discriminator_input, discriminator_input->next);
  learning_rate = 0.0002f;
}

extern "C" {

void Predict(double* _input, uint8_t* _output) {
  Import(generator_input->output[0], _input);
  Range(generator_input->next, generator_output).Apply([](Node* node) {
    node->training = false;
    node->Forward(1);
  });
  Export(generator_output->output[0], _output);
}

void LoadPretrainedModel() {
  Load();
}

void ResetModelWeight() {
  LoadModelFromString(initial_model);
  learning_rate = 0.006f;
};

void Train() {
  std::vector<Example> examples;

  // Use DCGAN Adam beta1 = 0.5 for stability.
  Range(generator_input, generator_output).Apply([](Node* node) {
    node->beta1 = 0.5f;
  });
  Range(discriminator_input, discriminator_output).Apply([](Node* node) {
    node->beta1 = 0.5f;
  });

  // Train the generative network ----------------------------------------------

  // Connect the two network.
  Node::Link(generator_output, discriminator_input->next);

  // Do not train on the discriminative network.
  Range(discriminator_input, discriminator_output).Apply(&Node::Lock);
  Range(generator_input, discriminator_output).Apply(&Node::Clear);

  // Generate some examples.
  for (size_t i = 0; i < 4; ++i) {
    auto input = Tensor::Random(generator_input->output[0].sizes);
    auto output = Tensor(1);
    output.values = {1.0f};
    examples.push_back({input, output});
  }

  // Train the network.
  auto generator_model = Model(generator_input, discriminator_output, examples);
  generator_model.loss_function = LossFunction::BinaryCrossEntropy;
  generator_model.Train(learning_rate, examples.size());

  // Train the discriminative network ------------------------------------------

  // Disconnect the two networks.
  Node::Link(discriminator_input, discriminator_input->next);
  Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);
  Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

  // Generate some examples.
  const size_t num_generated = examples.size();
  examples.clear();
  std::uniform_int_distribution<> random_index(0, training_set.size() - 1);
  for (size_t i = 0; i < num_generated; ++i) {
    auto output = Tensor(1);
    // One fake
    output.values = {0.0f};
    examples.push_back({generator_output->output[i], output});

    // One real (one-sided label smoothing)
    const Tensor& real_input = training_set[random_index(random_generator)].input;
    output.values = {0.9f};
    examples.push_back({real_input, output});
  }

  // Train the network.
  auto discriminator_model =
      Model(discriminator_input, discriminator_output, examples);
  discriminator_model.loss_function = LossFunction::BinaryCrossEntropy;
  discriminator_model.Train(learning_rate, examples.size());
}

}; // Extern C

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

  // Use DCGAN Adam beta1 = 0.5 for stability (Radford et al., 2015).
  Range(generator_input, generator_output).Apply([](Node* node) {
    node->beta1 = 0.5f;
  });
  Range(discriminator_input, discriminator_output).Apply([](Node* node) {
    node->beta1 = 0.5f;
  });

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
  std::vector<float> ema_gen_params =
      Model(generator_input, generator_output).SerializeParams();
  constexpr float ema_decay = 0.995f;

  auto write_preview = [&](const std::string& filename) {
    std::vector<Tensor> samples;
    samples.reserve(preview_latents.size());
    for (const auto& z : preview_latents) {
      generator_input->output[0] = z;
      Range(generator_input->next, generator_output).Apply([](Node* node) {
        node->training = false;
        node->Forward(1);
      });
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
    Node::Link(discriminator_input, discriminator_input->next);
    Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);

    float d_loss = 0.f;
    for (int c = 0; c < n_critic; ++c) {
      Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

      // Generate fake batch (without updating generator).
      for (size_t b = 0; b < batch; ++b) {
        generator_input->output[b] =
            Tensor::Random(generator_input->output[0].sizes);
      }
      Range(generator_input->next, generator_output).Apply([&](Node* node) {
        node->training = true;
        node->Forward(batch);
      });

      std::vector<Example> d_examples;
      d_examples.reserve(batch * 2);
      for (size_t b = 0; b < batch; ++b) {
        Tensor fake_label(1);
        fake_label.values = {0.0f};
        d_examples.push_back({generator_output->output[b], fake_label});

        Tensor real_label(1);
        real_label.values = {0.9f};
        d_examples.push_back(
            {training_set[random_index(random_generator)].input, real_label});
      }

      Model d_model(discriminator_input, discriminator_output, d_examples);
      d_model.batch_size = std::min(Node::T, d_examples.size());
      d_model.loss_function = LossFunction::BinaryCrossEntropy;
      d_model.Train(cur_lr, d_examples.size());
      d_loss = d_model.LastError();
    }

    // 2) Train generator once.
    Node::Link(generator_output, discriminator_input->next);
    Range(discriminator_input, discriminator_output).Apply(&Node::Lock);
    Range(generator_input, discriminator_output).Apply(&Node::Clear);

    std::vector<Example> g_examples;
    g_examples.reserve(batch);
    for (size_t b = 0; b < batch; ++b) {
      Tensor label(1);
      label.values = {1.0f};
      g_examples.push_back(
          {Tensor::Random(generator_input->output[0].sizes), label});
    }

    Model g_model(generator_input, discriminator_output, g_examples);
    g_model.batch_size = std::min(Node::T, batch);
    g_model.loss_function = LossFunction::BinaryCrossEntropy;
    g_model.Train(cur_lr, g_examples.size());
    float g_loss = g_model.LastError();

    // Update EMA of generator weights.
    auto cur_gen_params =
        Model(generator_input, generator_output).SerializeParams();
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
      Model gen_only(generator_input, generator_output);
      gen_only.DeserializeParams(ema_gen_params);
      Save();
      write_preview(preview_path);
      if (step < steps) {
        gen_only.DeserializeParams(cur_gen_params);
      }
    }
  }
  return 0;
}
#endif

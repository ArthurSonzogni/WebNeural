#include "Allocator.hpp"
#include "Image.hpp"
#include "Image.hpp"
#include "Model.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <fstream>
#include <random>

Tensor goal() {
  static std::mt19937 rng;
  static std::uniform_real_distribution<float> length(0.f, 5.0);
  static std::uniform_real_distribution<float> noise(0.f, 0.1f);

  Tensor ret({2});

  float l = length(rng);
  float theta = l * 6;
  float dx = noise(rng);
  float dy = noise(rng);
  ret.values = {l * cosf(theta) + dx, l * sinf(theta) + dy};
  return ret;
}

// The goal is to generate sample (x,y) so that y = exp(-x*x) using a Generative
// adversarial network.
TEST(GAN, spiral) {
  // ┌──────────────────────────┐
  // │Define the neural networks│
  // └──────────────────────────┘
  Allocator a;
  auto generator = [&](Node* X) {
    for(int i = 0; i<2; ++i) {
      X = a.Linear(X, {30});
      X = a.Tanh(X);
      X = a.BatchNormalization(X);
    } 
    X = a.Linear(X, {30});
    X = a.Tanh(X);
    X = a.Linear(X, {2});  // (x,y);
    return X;
  };

  auto discriminator = [&](Node* X) {
    //X = a.Noise(X, 0.2);
    for(int i = 0; i<3; ++i) {
      X = a.Linear(X, {30});
      X = a.Tanh(X);
      X = a.BatchNormalization(X);
    } 
    X = a.Linear(X, {30});
    X = a.Tanh(X);
    X = a.Linear(X, {1});
    return X;
  };

  auto generator_input = a.Input({10});
  auto generator_output = generator(generator_input);

  auto discriminator_input = a.Input({2});
  auto discriminator_output = discriminator(discriminator_input);

  float generator_loss = 0.f;
  float discriminator_loss = 0.f;
  for (size_t iteration = 0; iteration < 1000000000; ++iteration) {
    // ┌───────────────────┐
    // │Train the generator│
    // └───────────────────┘

    // Connect the two network.
    Node::Link(generator_output, discriminator_input->next);

    // Do not learn on the discriminative part.
    Range(discriminator_input, discriminator_output).Apply(&Node::Lock);
    Range(generator_input, discriminator_output).Apply(&Node::Clear);

    // Generate some examples.
    std::vector<Example> generator_examples;
    for (size_t i = 0; i < 64; ++i) {
      auto input = Tensor::Random(generator_input->output[0].sizes);
      auto output = Tensor({1});
      output.values = {1};  // 1 means 'real'
      generator_examples.push_back({input, output});
    }

    // Train
    Model model_generator(generator_input, discriminator_output,
                          generator_examples);
    model_generator.loss_function = LossFunction::WasserStein;
    model_generator.post_update_function = PostUpdateFunction::ClipWeight;
    model_generator.Train(0.00005f, generator_examples.size());
    generator_loss = model_generator.LastError();

    // ┌───────────────────────┐
    // │Train the discriminator│
    // └───────────────────────┘

    // Disconnect the two network.
    Node::Link(discriminator_input, discriminator_input->next);

    // Unlock learning on the discriminator part.
    Range(discriminator_input, discriminator_output).Apply(&Node::Unlock);
    Range(discriminator_input, discriminator_output).Apply(&Node::Clear);

    // Generator some examples.
    std::vector<Example> discriminator_examples;
    for (auto& generator_output : generator_output->output) {
      // Make one fake example.
      auto fake_input = generator_output;

      auto fake_output = Tensor({1});
      fake_output.values = {0.0};  // 0 means 'fake'.
      discriminator_examples.push_back({fake_input, fake_output});
    }

    for (auto& generator_output : generator_output->output) {
      generator_output.Fill(0.f);
      // Make one real example.
      auto real_input = goal(); 
      auto real_output = Tensor({1});
      real_output.values = {1.0};  // 1 means 'real';
      discriminator_examples.push_back({real_input, real_output});
    }

    Model model_discriminator(discriminator_input, discriminator_output,
                              discriminator_examples);
    model_discriminator.loss_function = LossFunction::WasserStein;
    model_discriminator.post_update_function = PostUpdateFunction::ClipWeight;
    model_discriminator.Train(0.001f, discriminator_examples.size());
    discriminator_loss = model_discriminator.LastError();

    generator_loss = discriminator_loss;
    discriminator_loss = generator_loss;
    //std::cerr << " generator_loss = " << generator_loss
              //<< " discriminator_loss = " << discriminator_loss << std::endl;

    static float seuil = 0;
    if (iteration < seuil)
      continue;
    seuil = 1.05 * seuil + 5;

    std::cerr << "seuil = " << seuil << std::endl;
    // ┌───────────────────────┐
    // │Update user screen     │
    // └───────────────────────┘
    
      std::uniform_real_distribution<double> distribution(0.0,1.0);

    static size_t image_index = 0;
    std::stringstream number;
    number << std::setw(6) << std::setfill('0') << image_index++;

    constexpr size_t width = 400;
    constexpr size_t height = 400;
    constexpr float x_min = -5.0;
    constexpr float x_max = +5.0;
    constexpr float y_min = -5.0;
    constexpr float y_max = +5.0;
    constexpr float max_value = 5.f;

    Tensor preview({width, height});
    constexpr size_t n = width * height / 20;
    for(size_t i = 0; i<n; ++i) {
      generator_input->output[0].Randomize();
      Range(generator_input->next, generator_output).Apply([](Node* node) {
          node->Forward(1);
      });

      float x = generator_output->output[0][0];
      float y = generator_output->output[0][1];

      bool in_image = (x > x_min && x < x_max && y > y_min && y < y_max);
      if (!in_image)
        continue;

      //float X = Tensor::Random({1})[0];
      //float Y = gaussian(X);

      // Scale to fix the image coordinates
      x = width * (x-x_min) / (x_max - x_min);
      y = height * (y-y_min) / (y_max - y_min);
      //X = width * (X-x_min) / (x_max - x_min);
      //Y = height * (Y-y_min) / (y_max - y_min);

      // Print
      preview.at(x, y) = std::min(preview.at(x, y) + 1.f, max_value);

      //if (preview.at(X,Y) == 0)
        //preview.at(X,Y) = 5.f;
    }

    Tensor discriminator_map({width, height});
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        // Scale to fit the neural network coordinates.
        float X = float(x) / width * (x_max - x_min) + x_min;
        float Y = float(y) / height * (y_max - y_min) + y_min;
        discriminator_input->output[0].values = {X, Y};

        Range(discriminator_input->next, discriminator_output)
            .Apply([](Node* node) { node->Forward(1); });

        discriminator_map.at(x, y) = discriminator_output->output[0][0];
      }
    }

    preview.Rescale();
    discriminator_map.Rescale();
    Tensor image = Tensor::ConcatenateHorizontal(preview, discriminator_map);
    std::ofstream("image" + number.str() + ".pgm") << image_PGM(image);
  }
}

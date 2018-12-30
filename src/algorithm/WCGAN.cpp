#include "algorithm/WCGAN.hpp"
#include <iomanip>
#include <iostream>
#include "Model.hpp"

void WCGAN::Init() {
  generator_input_ = allocator_.Input(latent_size);
  generator_output_ = generator(allocator_, generator_input_);

  discriminator_input_ = allocator_.Input(input().sizes);
  discriminator_output_ = discriminator(allocator_, discriminator_input_);
}

void WCGAN::Train() {
  //---------------------------------------------------------------------------
  // Train the generator
  //---------------------------------------------------------------------------

  // Connect the two network.
  Node::Link(generator_output_, discriminator_input_->next);

  // Do not train on the discriminative network.
  Range(discriminator_input_, discriminator_output_).Apply(&Node::Lock);
  Range(generator_input_, discriminator_output_).Apply(&Node::Clear);

  // Generate instances from the latent distribution.
  std::vector<Example> examples;
  for (int i = 0; i < batch_size; ++i) {
    auto input = Tensor::Random(latent_size);
    auto output = Tensor({1});
    output.values = {1.0};
    examples.push_back({input, output});
  }

  // Train the generator;
  auto model_train_generator =
      Model(generator_input_, discriminator_output_, examples);
  model_train_generator.loss_function = LossFunction::WasserStein;

  float error = 0.f;
  std::vector<Tensor> generated;
  for (size_t i = 0; i < examples.size(); i += Node::T) {
    model_train_generator.Train(learning_rate, examples.size());
    generated.insert(generated.end(), generator_output_->output.begin(),
                     generator_output_->output.end());
    error += model_train_generator.LastError();
  }
  float generative_error = Node::T * error / examples.size();

  //---------------------------------------------------------------------------
  // Train the discriminator
  //---------------------------------------------------------------------------
  // Disconnect the two networks.
  Node::Link(discriminator_input_, discriminator_input_->next);
  Range(discriminator_input_, discriminator_output_).Apply(&Node::Unlock);
  Range(discriminator_input_, discriminator_output_).Apply(&Node::Clear);

  examples.clear();
  for (size_t i = 0; i < generated.size(); ++i) {
    auto output = Tensor({1});

    output.values = {1.f};
    examples.push_back(Example{input(), output});

    output.values = {-1.f};
    examples.push_back(Example{generated[i], output});
  }

  // Train the network.
  auto model_train_discriminator =
      Model(discriminator_input_, discriminator_output_, examples);
  model_train_discriminator.loss_function = LossFunction::WasserStein;
  //model_train_discriminator.post_update_function =
      //PostUpdateFunction::ClipWeight(discriminator_input_->next,
                                     //discriminator_output_);
  model_train_discriminator.post_update_function =
      PostUpdateFunction::GradientPenalty(discriminator_input_->next,
                                          discriminator_output_, 0.99);

  // std::shuffle(input.begin(), input.end(), random_generator);
  for(int i = 0; i<10; ++i) {
    model_train_discriminator.Train(learning_rate, examples.size());
  }
  float discriminative_error = model_train_discriminator.LastError();

  std::cout << "| g_error = " << std::setw(10) << generative_error
            << "| d_error " << std::setw(10)
            << discriminative_error
            //<< "| epoch = "
            //<< (0.0 * float(Node::T) / training_set.size()) << std::endl;
            << std::endl;
}

Tensor WCGAN::Generate() {
  auto model =
      Model(generator_input_, generator_output_, std::vector<Example>());
  return model.Predict(Tensor::Random(latent_size));
}

void WCGAN::SaveToFile(const std::string& filename) {
  Node::Link(generator_output_, discriminator_input_->next);
  auto model =
      Model(generator_input_, discriminator_output_, std::vector<Example>());
  model.SerializeParamsToFile(filename);
}

void WCGAN::LoadFromFile(const std::string& filename) {
  Node::Link(generator_output_, discriminator_input_->next);
  auto model =
      Model(generator_input_, discriminator_output_, std::vector<Example>());
  model.DeserializeParamsFromFile(filename);
}

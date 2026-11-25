#include "algorithm/WCGAN.hpp"
#include <iomanip>
#include <iostream>
#include "Graph.hpp"
#include "LossFunction.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"

void WCGAN::Init() {
  generator_input_ = allocator_.Input(latent_size);
  generator_output_ = generator(allocator_, generator_input_);

  discriminator_input_ = allocator_.Input(input().sizes);
  discriminator_output_ = discriminator(allocator_, discriminator_input_);
  discriminator_first_ = discriminator_input_->outputs[0];
}

void WCGAN::Train() {
  Graph gen_graph(generator_output_);
  Graph disc_graph(discriminator_output_);
  Adam opt_g(gen_graph.TrainableNodes());
  Adam opt_d(disc_graph.TrainableNodes());

  //---------------------------------------------------------------------------
  // Train the generator
  //---------------------------------------------------------------------------
  discriminator_first_->SetInput(0, generator_output_);
  Graph full_graph(discriminator_output_);

  const size_t batch = static_cast<size_t>(std::max(1, batch_size));
  generator_input_->EnsureBatchSize(batch);
  for (size_t i = 0; i < batch; ++i) {
    generator_input_->output[i] = Tensor::Random(latent_size);
  }
  full_graph.Forward(batch, true);
  full_graph.ZeroGrad(batch);

  Tensor target_real(1);
  target_real.values = {1.0f};
  float error = 0.f;
  for (size_t i = 0; i < batch; ++i) {
    float cur = 0.f;
    LossFunction::WasserStein(target_real, discriminator_output_->output[i],
                              &cur, &discriminator_output_->grad[i]);
    error += cur;
  }
  full_graph.Backward(batch);
  opt_g.Step(batch, learning_rate);
  opt_d.ZeroGrad();

  std::vector<Tensor> generated(generator_output_->output.begin(),
                                generator_output_->output.begin() + batch);
  float generative_error = error / static_cast<float>(batch);

  //---------------------------------------------------------------------------
  // Train the discriminator
  //---------------------------------------------------------------------------
  discriminator_first_->SetInput(0, discriminator_input_);

  std::vector<Example> examples;
  examples.reserve(generated.size() * 2);
  for (size_t i = 0; i < generated.size(); ++i) {
    Tensor out(1);
    out.values = {1.f};
    examples.push_back(Example{input(), out});
    out.values = {-1.f};
    examples.push_back(Example{generated[i], out});
  }

  auto model_train_discriminator =
      Model(discriminator_input_, discriminator_output_, examples);
  model_train_discriminator.loss_function = LossFunction::WasserStein;
  model_train_discriminator.post_update_function =
      PostUpdateFunction::WeightDecay(discriminator_output_, 0.99f);

  for (int i = 0; i < 10; ++i) {
    model_train_discriminator.Train(learning_rate, examples.size());
  }
  float discriminative_error = model_train_discriminator.LastError();

  std::cout << "| g_error = " << std::setw(10) << generative_error
            << "| d_error " << std::setw(10) << discriminative_error
            << std::endl;
}

Tensor WCGAN::Generate() {
  Graph gen_graph(generator_output_);
  generator_input_->output[0] = Tensor::Random(latent_size);
  gen_graph.Forward(1, false);
  return generator_output_->output[0];
}

void WCGAN::SaveToFile(const std::string& filename) {
  discriminator_first_->SetInput(0, generator_output_);
  Graph full_graph(discriminator_output_);
  full_graph.SaveWeightsToFile(filename);
  discriminator_first_->SetInput(0, discriminator_input_);
}

void WCGAN::LoadFromFile(const std::string& filename) {
  discriminator_first_->SetInput(0, generator_output_);
  Graph full_graph(discriminator_output_);
  full_graph.LoadWeightsFromFile(filename);
  discriminator_first_->SetInput(0, discriminator_input_);
}

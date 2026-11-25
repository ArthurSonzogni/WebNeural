#include <cmath>
#include "Allocator.hpp"
#include "Graph.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "gtest/gtest.h"

TEST(Graph, ResidualAdd) {
  // Learn y = 2 * x using a residual skip connection: y = x + Linear(x).
  Allocator a;
  Node* x = a.Input({4});
  Node* branch = a.Linear(x, {4});
  Node* y = a.Add(x, branch);

  std::vector<Example> examples;
  for (int i = 0; i < 64; ++i) {
    Tensor in = Tensor::Random({4});
    Tensor out = in;
    for (float& v : out.values)
      v *= 2.f;
    examples.push_back({in, out});
  }

  Model model(x, y, examples);
  model.Train(0.05f, 4000);
  EXPECT_LT(model.Error(), 1e-2f);
}

TEST(Graph, FanOutGradientAccumulation) {
  // Two parallel branches from the same Input merged with Add:
  // y = Linear1(x) + Linear2(x). Verify gradients flow to both branches
  // and back to x cleanly.
  Allocator a;
  Node* x = a.Input({3});
  Node* b1 = a.Linear(x, {3});
  Node* b2 = a.Linear(x, {3});
  Node* y = a.Add(b1, b2);

  Graph g(y);
  x->output[0].values = {1.f, -2.f, 0.5f};
  g.Forward(1, true);
  g.ZeroGrad(1);
  y->grad[0].values = {1.f, 1.f, 1.f};
  g.Backward(1);

  // Input gradient should be the sum of gradients from both Linear branches.
  // Check against finite differences on x.
  const float eps = 1e-3f;
  for (size_t i = 0; i < 3; ++i) {
    float orig = x->output[0].values[i];
    x->output[0].values[i] = orig + eps;
    g.Forward(1, false);
    float sum_plus =
        y->output[0].values[0] + y->output[0].values[1] + y->output[0].values[2];

    x->output[0].values[i] = orig - eps;
    g.Forward(1, false);
    float sum_minus =
        y->output[0].values[0] + y->output[0].values[1] + y->output[0].values[2];

    x->output[0].values[i] = orig;
    float numerical = (sum_plus - sum_minus) / (2.f * eps);
    EXPECT_NEAR(x->grad[0].values[i], numerical, 1e-2f);
  }
}

TEST(Graph, Concat1DAnd3D) {
  // 1D Concat: [3] + [2] -> [5]
  {
    Allocator a;
    Node* x = a.Input({3});
    Node* b1 = a.Linear(x, {3});
    Node* b2 = a.Linear(x, {2});
    Node* c = a.Concat(b1, b2);
    Node* y = a.Linear(c, {3});

    std::vector<Example> examples;
    for (int i = 0; i < 32; ++i) {
      Tensor in = Tensor::Random({3});
      examples.push_back({in, in});
    }
    Model model(x, y, examples);
    model.Train(0.05f, 3000);
    EXPECT_LT(model.Error(), 1e-2f);
  }

  // 3D channel Concat: [4, 4, 2] + [4, 4, 3] -> [4, 4, 5]
  {
    Allocator a;
    Node* x = a.Input({4, 4, 2});
    Node* b1 = a.Relu(x);
    Node* b2 = a.Convolution2D(x, {1, 1}, 3, 1);
    Node* c = a.Concat(b1, b2);
    EXPECT_EQ(c->output[0].sizes, (std::vector<size_t>{4, 4, 5}));

    Graph g(c);
    x->output[0] = Tensor::Random({4, 4, 2});
    g.Forward(1, true);
    g.ZeroGrad(1);
    c->grad[0].Fill(1.f);
    g.Backward(1);
    // All elements in x->grad[0] should be non-zero or finite.
    for (float v : x->grad[0].values) {
      EXPECT_TRUE(std::isfinite(v));
    }
  }
}

TEST(Graph, SGDOptimizerAndCompactSerialization) {
  Allocator a;
  Node* x = a.Input({2});
  Node* y = a.Linear(x, {2});

  Graph g(y);
  SGD opt(g.TrainableNodes(), 0.9f);

  Tensor in({2});
  in.values = {1.f, -1.f};
  Tensor target({2});
  target.values = {0.5f, 0.25f};

  for (int step = 0; step < 200; ++step) {
    x->output[0] = in;
    g.Forward(1, true);
    g.ZeroGrad(1);
    float err = 0.f;
    LossFunction::SquaredDifference(target, y->output[0], &err, &y->grad[0]);
    g.Backward(1);
    opt.Step(1, 0.05f);
  }

  x->output[0] = in;
  g.Forward(1, false);
  EXPECT_LT((y->output[0] - target).Error(), 1e-3f);

  // Compact weight serialization round-trip.
  auto weights = g.SerializeWeights();
  EXPECT_EQ(weights.size(), y->params.values.size());

  y->params.Fill(0.f);
  g.DeserializeWeights(weights);
  x->output[0] = in;
  g.Forward(1, false);
  EXPECT_LT((y->output[0] - target).Error(), 1e-3f);
}

TEST(Graph, DynamicBatchSizeBeyond64) {
  // Verify nodes lazily grow beyond the old 64-element cap (including BN + Dropout).
  Allocator a;
  Node* x = a.Input({4});
  Node* h = a.Linear(x, {4});
  h = a.BatchNormalization(h);
  h = a.Dropout(h, 0.8f);
  Node* y = a.LeakyRelu(h);

  // Initially only slot 0 is allocated.
  EXPECT_EQ(x->output.size(), 1u);
  EXPECT_EQ(y->output.size(), 1u);

  const size_t batch = 96;
  Graph g(y);
  x->EnsureBatchSize(batch);
  for (size_t b = 0; b < batch; ++b) {
    x->output[b] = Tensor::Random({4});
  }
  g.Forward(batch, true);
  g.ZeroGrad(batch);
  for (size_t b = 0; b < batch; ++b) {
    y->grad[b].Fill(1.f);
  }
  g.Backward(batch);

  EXPECT_EQ(y->output.size(), batch);
  EXPECT_EQ(x->grad.size(), batch);
}

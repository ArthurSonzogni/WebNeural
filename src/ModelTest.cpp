#include "gtest/gtest.h"

#include "node/Input.hpp"
#include "node/Linear.hpp"
#include "node/Sigmoid.hpp"
#include "Model.hpp"

TEST(Model, Serialize) {
  auto input = Input({5,5});
  auto linear = Linear(&input, {5,5});
  auto output = Sigmoid(&linear);

  Model model(&input, &output, {});

  auto serialized_params = model.SerializeParams();

  Tensor old_linear_params = linear.params;
  linear.params.Fill(0.f);
  EXPECT_TRUE((old_linear_params - linear.params).Error() > 1e-5);
  model.DeserializeParams(serialized_params);
  EXPECT_TRUE((old_linear_params - linear.params).Error() < 1e-5);
}

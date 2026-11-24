#include "node/BatchNormalization.hpp"
#include <cmath>
#include <stdexcept>

BatchNormalization::BatchNormalization(Node* node) {
  Link(node);

  const auto& in_sizes = input[0]->sizes;
  if (in_sizes.size() >= 3) {
    channels = in_sizes[2];
    spatial_size = input[0]->values.size() / channels;
  } else if (in_sizes.size() == 1) {
    channels = in_sizes[0];
    spatial_size = 1;
  } else {
    channels = 1;
    spatial_size = input[0]->values.size();
  }

  output = std::vector<Tensor>(T, Tensor(in_sizes));
  x_hat = std::vector<Tensor>(T, Tensor(in_sizes));

  // Learnable scale (gamma = 1) and shift (beta = 0) per channel.
  params = Tensor({2, channels});
  for (size_t c = 0; c < channels; ++c) {
    params[c] = 1.f;
    params[channels + c] = 0.f;
  }

  mean.assign(channels, 0.f);
  inv_dev.assign(channels, 1.f);
  running_mean.assign(channels, 0.f);
  running_var.assign(channels, 1.f);

  InitInternalSensitivity();
}

void BatchNormalization::Forward(size_t batch_size) {
  if (batch_size == 0 || spatial_size == 0 || channels == 0)
    return;

  const float total = static_cast<float>(spatial_size * batch_size);

  #pragma omp parallel for if(channels >= 8)
  for (size_t c = 0; c < channels; ++c) {
    const size_t offset = c * spatial_size;
    float m = 0.f;
    float inv = 1.f;

    if (training) {
      float sum = 0.f;
      for (size_t batch = 0; batch < batch_size; ++batch) {
        const float* __restrict__ in = input[batch]->values.data() + offset;
        for (size_t s = 0; s < spatial_size; ++s) {
          sum += in[s];
        }
      }
      m = sum / total;

      float var = 0.f;
      for (size_t batch = 0; batch < batch_size; ++batch) {
        const float* __restrict__ in = input[batch]->values.data() + offset;
        for (size_t s = 0; s < spatial_size; ++s) {
          const float d = in[s] - m;
          var += d * d;
        }
      }
      var /= total;
      inv = 1.f / std::sqrt(var + 1e-5f);

      mean[c] = m;
      inv_dev[c] = inv;

      if (!has_running_stats) {
        running_mean[c] = m;
        running_var[c] = var;
      } else {
        constexpr float momentum = 0.9f;
        running_mean[c] = momentum * running_mean[c] + (1.f - momentum) * m;
        running_var[c] = momentum * running_var[c] + (1.f - momentum) * var;
      }
    } else {
      m = running_mean[c];
      inv = 1.f / std::sqrt(running_var[c] + 1e-5f);
      mean[c] = m;
      inv_dev[c] = inv;
    }

    const float gamma = params[c];
    const float beta = params[channels + c];
    for (size_t batch = 0; batch < batch_size; ++batch) {
      const float* __restrict__ in = input[batch]->values.data() + offset;
      float* __restrict__ xh = x_hat[batch].values.data() + offset;
      float* __restrict__ out = output[batch].values.data() + offset;
      #pragma omp simd
      for (size_t s = 0; s < spatial_size; ++s) {
        const float norm = (in[s] - m) * inv;
        xh[s] = norm;
        out[s] = gamma * norm + beta;
      }
    }
  }

  if (training) {
    has_running_stats = true;
  }
}

void BatchNormalization::Backward(size_t batch_size) {
  if (batch_size == 0 || spatial_size == 0 || channels == 0)
    return;

  const float total = static_cast<float>(spatial_size * batch_size);
  float* __restrict__ PS0 = params_sensitivity[0].values.data();
  ps_batches_used = 1;

  #pragma omp parallel for if(channels >= 8)
  for (size_t c = 0; c < channels; ++c) {
    const size_t offset = c * spatial_size;
    const float gamma = params[c];
    const float inv = inv_dev[c];

    float dgamma = 0.f;
    float dbeta = 0.f;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      const float* __restrict__ xh = x_hat[batch].values.data() + offset;
      const float* __restrict__ os = output_sensitivity[batch]->values.data() + offset;
      #pragma omp simd reduction(+:dgamma,dbeta)
      for (size_t s = 0; s < spatial_size; ++s) {
        dgamma += os[s] * xh[s];
        dbeta += os[s];
      }
    }
    PS0[c] += dgamma;
    PS0[channels + c] += dbeta;

    if (!training) {
      const float scale = gamma * inv;
      for (size_t batch = 0; batch < batch_size; ++batch) {
        const float* __restrict__ os = output_sensitivity[batch]->values.data() + offset;
        float* __restrict__ is = input_sensitivity[batch].values.data() + offset;
        #pragma omp simd
        for (size_t s = 0; s < spatial_size; ++s) {
          is[s] = os[s] * scale;
        }
      }
      continue;
    }

    const float scale = gamma * inv;
    const float mean_dbeta = dbeta / total;
    const float mean_dgamma = dgamma / total;
    for (size_t batch = 0; batch < batch_size; ++batch) {
      const float* __restrict__ xh = x_hat[batch].values.data() + offset;
      const float* __restrict__ os = output_sensitivity[batch]->values.data() + offset;
      float* __restrict__ is = input_sensitivity[batch].values.data() + offset;
      #pragma omp simd
      for (size_t s = 0; s < spatial_size; ++s) {
        is[s] = scale * (os[s] - mean_dbeta - xh[s] * mean_dgamma);
      }
    }
  }
}

void BatchNormalization::SerializeParams(std::vector<float>& value) {
  Node::SerializeParams(value);
  for (size_t c = 0; c < channels; ++c)
    value.push_back(running_mean[c]);
  for (size_t c = 0; c < channels; ++c)
    value.push_back(running_var[c]);
}

void BatchNormalization::DeserializeParams(const std::vector<float>& value,
                                           size_t& index) {
  Node::DeserializeParams(value, index);
  if (index > value.size() || value.size() - index < 2 * channels) {
    throw std::out_of_range("Serialized BatchNormalization stats too small");
  }
  for (size_t c = 0; c < channels; ++c)
    running_mean[c] = value[index++];
  for (size_t c = 0; c < channels; ++c)
    running_var[c] = value[index++];
  has_running_stats = true;
}

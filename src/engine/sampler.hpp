#pragma once
// Sampler: temperature / top-k / top-p / min-p / repetition penalty.
// Logits are fp32: the head gemm produces fp32 and greedy parity at temperature 0 is an
// argmax over exactly those values, so narrowing to fp16 first would be a needless
// rounding between the model and the decision.
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>

namespace helios {

struct GenParams {
  int max_tokens = 256;
  float temperature = 0.7f;
  int top_k = 40;
  float top_p = 0.95f;
  float min_p = 0.0f;
  float rep_penalty = 1.0f;
  int rep_window = 256;
  uint64_t seed = 0;
  bool greedy = false;
  // Token ids the repetition penalty must NOT touch: the stop/EOS set, which for this checkpoint
  // includes the token that closes the think block. Penalising them traps a reasoning model - it
  // cannot bring itself to emit </think>, so it thinks until max_tokens and returns an EMPTY answer.
  // Measured: at rep_penalty 1.0 the answer arrives after 266 characters of reasoning; at 1.2 it is
  // still reasoning at 1,887 characters and content is "". Any client defaulting a mild
  // repetition_penalty got blank responses.
  std::vector<int> rep_exempt;
  std::vector<std::string> stop;   // stop strings
  std::string grammar;             // unused (reserved)
};

class Sampler {
public:
  void reset(uint64_t seed);
  // logits: fp32 host array [vocab]; recent: token ids for repetition penalty
  int sample(const float* logits, int vocab, const GenParams& p, const std::vector<int>& recent);

private:
  uint64_t state_ = 0x853c49e6748fea9bull;
  float next_f();
  std::vector<float> buf_;
  std::vector<int> idx_;
};

}  // namespace helios
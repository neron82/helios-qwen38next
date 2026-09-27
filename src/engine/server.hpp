#pragma once
// OpenAI-compatible HTTP surface over a single-sequence Runner.
#include "engine/runner.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>

namespace helios {

int run_server(Runner& runner, Tokenizer& tk, const std::string& host, int port, int n_threads,
               const std::string& api_key, int default_max_tokens,
               const std::string& reasoning_effort, const std::string& model_dir);

}  // namespace helios

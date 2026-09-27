// Qwen3.8 chat-template cases. test/chat_parity_qwen.py re-renders the same message lists from the
// checkpoint's own chat_template.jinja (via transformers) and diffs the bytes.
//
// Keep the case list in sync with that script. The cases that matter are the ones where the Qwen and
// GLM formats actually differ: the synthetic reasoning-instruction system turn, the per-effort
// instruction text, the tools preamble, the assistant tool-call shape and the generation prompt.
#include "core/chat.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace helios;
using json = nlohmann::ordered_json;

static void pc(const char* name, const std::string& t) {
  printf("%s %s\n", name, json(t).dump().c_str());
}

int main() {
  json tools = json::array({{{"type", "function"},
                            {"function",
                             {{"name", "get_weather"},
                              {"description", "Get weather for a city"},
                              {"parameters",
                               {{"type", "object"},
                                {"properties",
                                 {{"city", {{"type", "string"}, {"description", "City name"}}}}},
                                {"required", json::array({"city"})}}}}}}});

  std::vector<ChatMsg> c1 = {{"user", "Weather in Paris?", "", {}, ""}};

  // Effort levels: xhigh injects a long instruction, low a short one, medium none at all (which also
  // drops the synthetic system turn). Getting these wrong changes the prompt silently.
  pc("QWEN1", render_chat_qwen(c1, json::array(), "xhigh", true));
  pc("QWEN2", render_chat_qwen(c1, json::array(), "low", true));
  pc("QWEN3", render_chat_qwen(c1, json::array(), "medium", true));

  // Tools preamble + the system message merged into the same turn.
  pc("QWEN4", render_chat_qwen(c1, tools, "xhigh", true));

  // Assistant tool call followed by its response.
  std::vector<ChatMsg> c5 = {
      {"user", "Weather in Paris?", "", {}, ""},
      {"assistant", "", "", {{"call_1", "get_weather", "{\"city\": \"Paris\"}"}}, ""},
      {"tool", "18C and sunny", "", {}, "call_1"}};
  pc("QWEN5", render_chat_qwen(c5, tools, "xhigh", true));

  // Multi-turn: reasoning is kept on the newest assistant turn and cleared on older ones.
  std::vector<ChatMsg> c6 = {{"user", "hi", "", {}, ""},
                             {"assistant", "hello", "step 1", {}, ""},
                             {"user", "bye", "", {}, ""}};
  pc("QWEN6", render_chat_qwen(c6, json::array(), "xhigh", true));

  // Explicit system message.
  std::vector<ChatMsg> c7 = {{"system", "You are terse.", "", {}, ""}, {"user", "hi", "", {}, ""}};
  pc("QWEN7", render_chat_qwen(c7, json::array(), "xhigh", true));

  // enable_thinking = false: closed empty think block in the generation prompt.
  pc("QWEN8", render_chat_qwen(c1, json::array(), "xhigh", true, false));
  return 0;
}

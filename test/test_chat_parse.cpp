// OutputParser tool-call parsing, in BOTH dialects this checkpoint family uses.
//
// The renderer in src/core/chat.cpp emits Qwen3's dialect - an inner
//     <function=NAME> ... <parameter=KEY>\nVALUE\n</parameter> ... </function>
// nested in <tool_call> - and the system prompt tells the model to produce the same. The parser
// originally understood only the older `NAME<arg_key>K</arg_key><arg_value>V</arg_value>` form, so a
// real tool call came back with name = the whole "<function=...>..." blob and arguments = "{}", and a
// client would have invoked the tool with no arguments at all. Nothing caught it because the chat
// tests only exercised the RENDERER; there was no parser coverage anywhere in the tree.
//
// The legacy dialect is still accepted here, because a checkpoint that follows the prompt literally
// can produce it.
#include "core/chat.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace helios;
using json = nlohmann::ordered_json;

static int failures = 0;

// One tool call, fed a few characters at a time so the streaming path is exercised too - the parser
// holds back partial markers and must reassemble the same result.
static void expect_call(const char* name, const std::string& body, const std::string& want_name,
                        const json& want_args, size_t chunk = 7) {
  OutputParser p(/*start_in_think=*/false);
  std::vector<OutputParser::Delta> deltas;
  for (size_t i = 0; i < body.size(); i += chunk) {
    std::string piece = body.substr(i, chunk);
    for (auto& d : p.feed(piece)) deltas.push_back(d);
  }
  for (auto& d : p.feed("")) deltas.push_back(d);
  p.finish();
  const ParsedOutput& out = p.parsed();

  if (out.tool_calls.size() != 1) {
    printf("FAIL %s: expected 1 tool call, got %zu\n", name, out.tool_calls.size());
    ++failures;
    return;
  }
  const ToolCall& tc = out.tool_calls[0];
  if (tc.name != want_name) {
    printf("FAIL %s: name was %s, wanted %s\n", name, tc.name.c_str(), want_name.c_str());
    ++failures;
    return;
  }
  json got = json::object();
  try {
    got = json::parse(tc.arguments);
  } catch (...) {
  }
  if (got != want_args) {
    printf("FAIL %s: arguments were %s, wanted %s\n", name, tc.arguments.c_str(),
           want_args.dump().c_str());
    ++failures;
    return;
  }
  printf("pass %s: %s(%s)\n", name, want_name.c_str(), want_args.dump().c_str());
}

// Prose and a tool call in the same turn: the prose must survive and the call must still be found.
static void expect_mixed(const char* name) {
  const std::string body =
      "Let me check that for you.\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
  OutputParser p(/*start_in_think=*/false);
  for (auto& d : p.feed(body)) {
  }
  p.finish();
  const ParsedOutput& out = p.parsed();
  if (out.tool_calls.size() != 1) {
    printf("FAIL %s: expected 1 tool call, got %zu\n", name, out.tool_calls.size());
    ++failures;
    return;
  }
  if (out.content.find("Let me check that for you") == std::string::npos) {
    printf("FAIL %s: prose lost, content=%s\n", name, out.content.c_str());
    ++failures;
    return;
  }
  printf("pass %s: prose kept, one call\n", name);
}

// A model that hallucinates a ChatML turn boundary must not have the template tokens reach the
// caller. This checkpoint's markers are <|im_start|>/<|im_end|> (its eos_token is <|im_end|>), and
// the parser's holdback/cut sets originally carried only the other dialect's, so these leaked into
// user-visible content - reliably so once a tool call was in play.
static void expect_no_template_leak(const char* name, const std::string& body) {
  OutputParser p(/*start_in_think=*/false);
  for (auto& d : p.feed(body)) {
  }
  p.finish();
  const std::string& out = p.parsed().content;
  const char* bad[] = {"<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>",
                       "<|observation|>", "<|endoftext|>",
                       "<tool_response>", "</tool_response>"};
  for (const char* b : bad) {
    if (out.find(b) != std::string::npos) {
      printf("FAIL %s: leaked %s into content: %s\n", name, b, out.c_str());
      ++failures;
      return;
    }
  }
  printf("pass %s: no template token in content\n", name);
}

int main() {
  expect_no_template_leak("im_start hallucinated",
                          "Here is the answer.\n<|im_start|>user\nwhat about tomorrow?<|im_end|>\n"
                          "And that is all.");
  expect_no_template_leak("im_start with a tool call",
                          "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
                          "</parameter>\n</function>\n</tool_call>\n"
                          "<|im_start|>user\n<tool_response>\n{\"type\": 0}\n</tool_response>");
  expect_call("qwen3 single", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
                              "</parameter>\n</function>\n</tool_call>",
              "get_weather", json{{"city", "Paris"}});
  expect_call("qwen3 two params", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
                                   "</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n"
                                   "</function>\n</tool_call>",
              "get_weather", json{{"city", "Paris"}, {"unit", "celsius"}});
  // A value containing a newline and the delimiter text must survive verbatim.
  expect_call("qwen3 multiline value", "<tool_call>\n<function=note>\n<parameter=text>\nline one\nline "
                                        "two\n</parameter>\n</function>\n</tool_call>",
              "note", json{{"text", "line one\nline two"}});
  // Numbers and booleans that look like JSON are typed, not stringified.
  expect_call("qwen3 typed args", "<tool_call>\n<function=set>\n<parameter=n>\n42\n</parameter>\n"
                                  "<parameter=on>\ntrue\n</parameter>\n</function>\n</tool_call>",
              "set", json{{"n", 42}, {"on", true}});
  // The legacy dialect this parser used to be the only one for.
  expect_call("legacy single", "<tool_call>get_weather<arg_key>city</arg_key>"
                               "<arg_value>Paris</arg_value></tool_call>",
              "get_weather", json{{"city", "Paris"}});
  expect_mixed("prose plus call");
  printf(failures ? "\n%d FAILURE(S)\n" : "\nall tool-call parser cases pass\n", failures);
  return failures ? 1 : 0;
}

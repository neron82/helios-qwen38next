// Chat template rendering + output parsing. See chat.hpp for the validation notes.
#include "core/chat.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace helios {

using json = nlohmann::ordered_json;

namespace {

// json.dumps(&, separators=(', ', ': ')) - what jinja's tojson produces and what the reference
// renders contain. nlohmann's dump() is compact, so rebuild the string with those separators while
// escaping scalars through nlohmann (identical escaping rules).
void dump_pretty(const json& j, std::string& out) {
  if (j.is_object()) {
    out += '{';
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      out += json(it.key()).dump();
      out += ": ";
      dump_pretty(it.value(), out);
    }
    out += '}';
  } else if (j.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& v : j) {
      if (!first) out += ", ";
      first = false;
      dump_pretty(v, out);
    }
    out += ']';
  } else {
    out += j.dump();
  }
}

std::string to_json_like_jinja(const json& j) {
  std::string s;
  dump_pretty(j, s);
  return s;
}

// A tool object may be {"type":"function","function":{...}} or the function object directly.
json tool_function(const json& tool) {
  if (tool.is_object() && tool.contains("function")) return tool["function"];
  return tool;
}

std::string cap_first(const std::string& s) {
  std::string c = s;
  if (!c.empty()) c[0] = (char)toupper((unsigned char)c[0]);
  return c;
}

std::string strip_ws(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Argument values: strings go in raw, everything else as JSON (matches the reference renders).
void append_args(const json& args, std::string& out) {
  if (!args.is_object()) {
    if (!args.is_null()) out += to_json_like_jinja(args);
    return;
  }
  for (auto it = args.begin(); it != args.end(); ++it) {
    out += "<arg_key>" + it.key() + "</arg_key><arg_value>";
    if (it.value().is_string()) out += it.value().get<std::string>();
    else out += to_json_like_jinja(it.value());
    out += "</arg_value>";
  }
}

}  // namespace

std::string render_chat(const std::vector<ChatMsg>& msgs, const json& tools,
                        const std::string& reasoning_effort, bool add_generation_prompt,
                        bool thinking) {
  std::string out = "[gMASK]<sop>";
  std::string eff = (reasoning_effort == "low" || reasoning_effort == "high") ? reasoning_effort
                                                                             : "max";
  out += "<|system|>Reasoning Effort: " + cap_first(eff);

  // Tools block (only when tools were supplied and any survives the defer_loading filter).
  if (tools.is_array() && !tools.empty()) {
    std::vector<json> shown;
    for (const auto& t : tools) {
      json fn = tool_function(t);
      if (fn.contains("defer_loading") && fn["defer_loading"].is_boolean() &&
          fn["defer_loading"].get<bool>())
        continue;
      shown.push_back(fn);
    }
    if (!shown.empty()) {
      out += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user "
             "query.\n\nYou are provided with function signatures within <tools></tools> XML "
             "tags:\n<tools>\n";
      for (const json& fn : shown) {
        json clean = json::object();
        for (auto it = fn.begin(); it != fn.end(); ++it) {
          // The reference template's tool_to_json skips exactly these two keys; a deferred tool was
          // already dropped above, so a surviving defer_loading:false must not reach the model.
          if (it.key() == "strict" || it.key() == "defer_loading") continue;
          clean[it.key()] = it.value();
        }
        out += to_json_like_jinja(clean);
        out += "\n";
      }
      out += "</tools>\n\nFor each function call, output the function name and arguments within the "
             "following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
             "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
             "<arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
  }

  int last_user = -1;
  for (size_t i = 0; i < msgs.size(); i++)
    if (msgs[i].role == "user") last_user = (int)i;

  for (size_t i = 0; i < msgs.size(); i++) {
    const ChatMsg& m = msgs[i];
    if (m.role == "user") {
      out += "<|user|>" + m.content;
    } else if (m.role == "system") {
      out += "<|system|>" + m.content;
    } else if (m.role == "assistant") {
      out += "<|assistant|>";
      // Reasoning is rendered only for an assistant turn after the last user turn; history is
      // cleared (<think></think>), which is what the reference template does.
      bool keep_reasoning = (int)i > last_user && !m.reasoning.empty();
      out += keep_reasoning ? ("<think>" + m.reasoning + "</think>") : "<think></think>";
      std::string c = strip_ws(m.content);
      if (!c.empty()) out += c;
      for (const ToolCall& tc : m.tool_calls) {
        json args = json::object();
        try {
          json parsed = json::parse(tc.arguments.empty() ? "{}" : tc.arguments);
          if (parsed.is_object()) args = parsed;
        } catch (...) {
        }
        out += "<tool_call>" + tc.name;
        append_args(args, out);
        out += "</tool_call>";
      }
    } else if (m.role == "tool") {
      // A run of consecutive tool messages becomes one <|observation|> block.
      if (i == 0 || msgs[i - 1].role != "tool") {
        out += "<|observation|>";
        size_t end = i;
        while (end + 1 < msgs.size() && msgs[end + 1].role == "tool") end++;
        bool have_ids = true;
        for (size_t k = i; k <= end; k++) if (msgs[k].tool_call_id.empty()) have_ids = false;
        const std::vector<ToolCall>* order = nullptr;
        if (have_ids && i > 0 && msgs[i - 1].role == "assistant" &&
            !msgs[i - 1].tool_calls.empty()) {
          bool all_match = true;
          for (size_t k = i; k <= end; k++) {
            bool found = false;
            for (const ToolCall& tc : msgs[i - 1].tool_calls)
              if (tc.id == msgs[k].tool_call_id) found = true;
            if (!found) all_match = false;
          }
          if (all_match) order = &msgs[i - 1].tool_calls;
        }
        if (order) {
          for (const ToolCall& tc : *order)
            for (size_t k = i; k <= end; k++)
              if (msgs[k].tool_call_id == tc.id)
                out += "<tool_response>" + msgs[k].content + "</tool_response>";
        } else {
          for (size_t k = i; k <= end; k++)
            out += "<tool_response>" + msgs[k].content + "</tool_response>";
        }
      }
    }
  }
  if (add_generation_prompt) out += thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
  return out;
}

// ------------------------------------------------------------------ Qwen3.8 chat format
// Transcribed from the checkpoint's own chat_template.jinja (the authoritative spec - what the model
// saw in training is whatever that file renders). Every literal below is copied verbatim from it; see
// test/chat_parity_qwen.py, which diffs this against transformers' apply_chat_template.
namespace {

const char* kQwenReasonXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
const char* kQwenReasonLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

// The template's tool-use preamble, reproduced exactly (it begins with the blank-line separator).
const char* kQwenToolInstr = R"JINJA(

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)JINJA";

std::string qwen_trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

}  // namespace

ChatFormat detect_chat_format(const std::string& model_dir) {
  FILE* f = fopen((model_dir + "/chat_template.jinja").c_str(), "rb");
  if (!f) return ChatFormat::Glm53;
  std::string t;
  char buf[1 << 16];
  size_t got;
  while ((got = fread(buf, 1, sizeof(buf), f)) > 0) t.append(buf, got);
  fclose(f);
  if (t.find("<|im_start|>") != std::string::npos && t.find("[gMASK]") == std::string::npos)
    return ChatFormat::Qwen38;
  return ChatFormat::Glm53;
}

std::string render_chat_qwen(const std::vector<ChatMsg>& msgs, const helios_json& tools,
                             const std::string& reasoning_effort, bool add_generation_prompt,
                             bool thinking) {
  // The template defaults to xhigh and rejects anything outside {xhigh, medium, low}. medium maps to
  // an empty instruction, which also suppresses the synthetic system turn entirely.
  std::string eff = reasoning_effort;
  if (eff != "xhigh" && eff != "medium" && eff != "low") eff = "xhigh";
  std::string reason_instr;
  if (thinking) reason_instr = eff == "xhigh" ? kQwenReasonXhigh : (eff == "low" ? kQwenReasonLow : "");

  const bool have_tools = tools.is_array() && !tools.empty();
  const bool first_is_system = !msgs.empty() && msgs[0].role == "system";
  std::string out;

  if (have_tools) {
    // Unlike GLM, the tools block is unconditional and every tool is dumped as-is (no filtering).
    out += "<|im_start|>system\n";
    if (!reason_instr.empty()) out += reason_instr + "\n\n";
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto& t : tools) out += "\n" + to_json_like_jinja(t);
    out += "\n</tools>";
    out += kQwenToolInstr;
    if (first_is_system) {
      std::string c = qwen_trim(msgs[0].content);
      if (!c.empty()) out += "\n\n" + c;
    }
    out += "<|im_end|>\n";
  } else if (first_is_system) {
    std::string c = qwen_trim(msgs[0].content);
    if (!c.empty())
      out += "<|im_start|>system\n" + (reason_instr.empty() ? "" : reason_instr + "\n\n") + c +
             "<|im_end|>\n";
    else if (!reason_instr.empty())
      out += "<|im_start|>system\n" + reason_instr + "<|im_end|>\n";
  } else if (!reason_instr.empty()) {
    out += "<|im_start|>system\n" + reason_instr + "<|im_end|>\n";
  }

  // Every assistant turn carries a think block. The template's condition is
  //   `preserve_thinking is undefined or preserve_thinking is true or loop.index0 > last_query_index`
  // and the server never passes preserve_thinking, so the first clause always holds and history is
  // never cleared. (GLM is the opposite - it clears older reasoning - which is why this cannot share
  // the GLM renderer's last_query logic.) An assistant turn with no reasoning renders an empty block.
  for (size_t i = 0; i < msgs.size(); i++) {
    const ChatMsg& m = msgs[i];
    if (m.role == "system") {
      continue;  // the template raises for a non-leading system message; skip rather than abort
    } else if (m.role == "user") {
      out += "<|im_start|>user\n" + qwen_trim(m.content) + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      std::string c = qwen_trim(m.content);
      out += "<|im_start|>assistant\n<think>\n" + qwen_trim(m.reasoning) + "\n</think>\n\n" + c;
      for (size_t k = 0; k < m.tool_calls.size(); k++) {
        const ToolCall& tc = m.tool_calls[k];
        if (k == 0 && c.empty()) out += "<tool_call>\n<function=" + tc.name + ">\n";
        else if (k == 0) out += "\n\n<tool_call>\n<function=" + tc.name + ">\n";
        else out += "\n<tool_call>\n<function=" + tc.name + ">\n";
        helios_json args = helios_json::object();
        try {
          helios_json parsed = helios_json::parse(tc.arguments.empty() ? "{}" : tc.arguments);
          if (parsed.is_object()) args = parsed;
        } catch (...) {
        }
        for (auto it = args.begin(); it != args.end(); ++it) {
          out += "<parameter=" + it.key() + ">\n";
          out += it.value().is_string() ? it.value().get<std::string>() : to_json_like_jinja(it.value());
          out += "\n</parameter>\n";
        }
        out += "</function>\n</tool_call>";
      }
      out += "<|im_end|>\n";
    } else if (m.role == "tool") {
      if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
      out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
      const bool last = (i + 1 == msgs.size());
      if (last || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
    }
  }

  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return out;
}

// ---------------------------------------------------------------- output parsing
namespace {

// This checkpoint closes its thinking with `</think>` (token 154842 in tokenizer.json). There is no
// DeepSeek-style alternative to accept here: nothing in the checkpoint's chat_template.jinja or
// tokenizer_config.json mentions one, and a previous constant for it was both unreferenced and
// malformed.
const char* kThinkEnd = "</think>";
const char* kToolBegin = "<tool_call>";
const char* kToolEnd = "</tool_call>";
const char* kObservation = "<|observation|>";
const char* kEos = "<|endoftext|>";
const char* kEndS = "<|end|>";
const char* kUser = "<|user|>";
const char* kAsst = "<|assistant|>";
const char* kSys = "<|system|>";
// This checkpoint is ChatML: its turn markers are <|im_start|>/<|im_end|> and its eos_token is
// <|im_end|> (tokenizer_config.json). The markers above are the other dialect's and were the only
// ones being held back, so a model that hallucinated a turn boundary - which it does readily once a
// tool call is in play - had its "<|im_start|>user" emitted straight into user-visible content. These
// are joined into the holdback set below so no template token can leak, whichever dialect produced
// it; the per-dialect constants above still drive boundary DETECTION.
const char* kImStart = "<|im_start|>";
const char* kImEnd = "<|im_end|>";
// render_chat_qwen wraps a tool RESULT in <|im_start|>user ... <tool_response>...</tool_response>
// ... <|im_end|> (chat.cpp, the m.role == "tool" branch), so a model echoing the template back
// produced these verbatim in content. They are output-only markers, never something to parse, so
// they belong in the strip set alongside the turn markers.
const char* kToolRespOpen = "<tool_response>";
const char* kToolRespClose = "</tool_response>";

// Longest suffix of `s` that is a proper prefix of one of the markers (so we never emit text that
// could still turn into a tag).
size_t holdback_len(const std::string& s) {
  static const std::vector<std::string> marks = {kThinkEnd, kToolBegin, kToolEnd, kObservation,
                                                 kEos, kEndS, kUser, kAsst, kSys,
                                                 kImStart, kImEnd,
                                                 kToolRespOpen, kToolRespClose};
  size_t best = 0;
  for (const std::string& m : marks)
    for (size_t len = 1; len < m.size() && len <= s.size(); len++)
      if (s.compare(s.size() - len, len, m, 0, len) == 0) best = std::max(best, len);
  return best;
}

// The model emits Qwen3's dialect:
//     <function=get_weather>
//     <parameter=location>
//     Paris
//     </parameter>
//     </function>
// and the assistant renderer in this file emits the same. The older `NAME<arg_key>K</arg_key>
// <arg_value>V</arg_value>` dialect is still accepted, because a checkpoint that follows the
// prompt literally can produce it. Before this, ONLY the legacy form was parsed: `tool_name_of`
// takes everything before the first `<arg_key>` as the name, so a Qwen3 call came back with
// name = "<function=get_weather>\n<parameter=location>..." and arguments = {} - which a client
// would dutifully invoke with no arguments.
bool qwen3_name(const std::string& body, std::string& name) {
  const size_t f = body.find("<function=");
  if (f == std::string::npos) return false;
  const size_t fe = body.find('>', f);
  if (fe == std::string::npos) return false;
  name = strip_ws(body.substr(f + 10, fe - f - 10));
  return !name.empty();
}

void put_arg(json& args, const std::string& key, const std::string& val) {
  json v;
  bool ok = true;
  try {
    v = json::parse(val);
  } catch (...) {
    ok = false;
  }
  args[key] = ok ? v : json(val);
}

std::string args_to_json(const std::string& body) {
  json args = json::object();
  if (body.find("<parameter=") != std::string::npos) {

    // <parameter=KEY>\nVALUE\n</parameter>, value may span lines.
    size_t pos = 0;
    while (true) {
      const size_t p = body.find("<parameter=", pos);
      if (p == std::string::npos) break;
      const size_t pe = body.find('>', p);
      const size_t close = body.find("</parameter>", pe == std::string::npos ? p : pe);
      if (pe == std::string::npos || close == std::string::npos) break;
      const std::string key = strip_ws(body.substr(p + 11, pe - p - 11));
      if (!key.empty()) put_arg(args, key, strip_ws(body.substr(pe + 1, close - pe - 1)));
      pos = close + 12;
    }
    return args.dump();
  }
  // legacy: NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
  size_t pos = 0;
  while (true) {
    size_t ak = body.find("<arg_key>", pos);
    if (ak == std::string::npos) break;
    size_t ak_end = body.find("</arg_key>", ak);
    size_t av = body.find("<arg_value>", ak_end);
    size_t av_end = body.find("</arg_value>", av);
    if (ak_end == std::string::npos || av == std::string::npos || av_end == std::string::npos) break;
    const std::string key = body.substr(ak + 9, ak_end - ak - 9);
    put_arg(args, key, body.substr(av + 11, av_end - av - 11));
    pos = av_end + 12;
  }
  return args.dump();
}

std::string tool_name_of(const std::string& body) {
  std::string qn;
  if (qwen3_name(body, qn)) return qn;
  size_t p = body.find("<arg_key>");
  return strip_ws(body.substr(0, p == std::string::npos ? body.size() : p));
}

// Clients match a tool result to its call by id, so ids must be unique across the turn - an
// unterminated final call included.
std::string tool_id_for(size_t n) {
  char buf[32];
  snprintf(buf, sizeof(buf), "call_%d", (int)n);
  return buf;
}

}  // namespace

std::vector<OutputParser::Delta> OutputParser::feed(const std::string& piece) {
  pending_.clear();
  if (done_) return pending_;   // after a turn boundary the rest of the stream is hallucination
  buf_ += piece;
  drain(false);
  return pending_;
}

std::vector<OutputParser::Delta> OutputParser::finish() {
  pending_.clear();
  drain(true);
  if (in_tool_ && !tool_body_.empty()) {   // unterminated tool call: emit what we have
    Delta d;
    d.tool_begin = true;
    d.tool_name = tool_name_of(tool_body_);
    d.tool_args_fragment = args_to_json(tool_body_);
    ToolCall tc;
    tc.id = tool_id_for(out_.tool_calls.size());
    tc.name = d.tool_name;
    tc.arguments = d.tool_args_fragment;
    d.tool_call_id = tc.id;
    out_.tool_calls.push_back(tc);
    pending_.push_back(d);
    tool_body_.clear();
    in_tool_ = false;
  }
  return pending_;
}

void OutputParser::drain(bool flush) {
  size_t pos = 0;
  while (pos < buf_.size()) {
    size_t keep = flush ? 0 : holdback_len(buf_.substr(pos));
    size_t limit = buf_.size() - keep;
    if (in_tool_) {
      size_t end = buf_.find(kToolEnd, pos);
      if (end == std::string::npos) {
        size_t take = limit > pos ? limit - pos : 0;
        tool_body_ += buf_.substr(pos, take);
        pos += take;
        break;
      }
      tool_body_ += buf_.substr(pos, end - pos);
      pos = end + strlen(kToolEnd);
      std::string name = tool_name_of(tool_body_);
      Delta d;
      d.tool_begin = true;
      d.tool_name = name;
      d.tool_args_fragment = args_to_json(tool_body_);
      ToolCall tc;
      tc.id = tool_id_for(out_.tool_calls.size());
      tc.name = name;
      tc.arguments = d.tool_args_fragment;
      d.tool_call_id = tc.id;
      out_.tool_calls.push_back(tc);
      pending_.push_back(d);
      tool_body_.clear();
      in_tool_ = false;
      continue;
    }
    // find the earliest marker in [pos, limit)
    size_t best = std::string::npos;
    const char* which = nullptr;
    size_t which_len = 0;
    struct M {
      const char* tag;
      size_t len;
    } marks[] = {{kThinkEnd, strlen(kThinkEnd)}, {kToolBegin, strlen(kToolBegin)},
                 {kObservation, strlen(kObservation)}, {kEos, strlen(kEos)}, {kEndS, strlen(kEndS)},
                 {kUser, strlen(kUser)}, {kAsst, strlen(kAsst)}, {kSys, strlen(kSys)},
                 {kImStart, strlen(kImStart)}, {kImEnd, strlen(kImEnd)},
                 {kToolRespOpen, strlen(kToolRespOpen)}, {kToolRespClose, strlen(kToolRespClose)}};
    for (const M& m : marks) {
      size_t f = buf_.find(m.tag, pos);
      if (f != std::string::npos && f < limit && f < best) { best = f; which = m.tag; which_len = m.len; }
    }
    if (best == std::string::npos) {
      size_t take = limit > pos ? limit - pos : 0;
      std::string text = buf_.substr(pos, take);
      pos += take;
      if (!text.empty()) {
        Delta d;
        if (in_think_) { d.reasoning = text; out_.reasoning += text; }
        else { d.content = text; out_.content += text; }
        pending_.push_back(d);
      }
      break;
    }
    if (best > pos) {
      std::string text = buf_.substr(pos, best - pos);
      Delta d;
      if (in_think_) { d.reasoning = text; out_.reasoning += text; }
      else { d.content = text; out_.content += text; }
      pending_.push_back(d);
    }
    pos = best + which_len;
    if (which == std::string(kThinkEnd)) {
      in_think_ = false;
    } else if (which == std::string(kToolBegin)) {
      in_tool_ = true;
      tool_body_.clear();
    } else if (which == std::string(kUser) || which == std::string(kAsst) ||
               which == std::string(kSys)) {
      // The model started a new turn: end generation here instead of streaming hallucinated
      // conversation turns to the client.
      Delta d;
      d.stop = true;
      pending_.push_back(d);
      done_ = true;
      buf_.clear();
      return;
    } else if (which == std::string(kObservation) || which == std::string(kEos) ||
               which == std::string(kEndS)) {
      // End of the assistant turn. <|observation|> is what the *template* writes before tool
      // responses, so a model that emits it has run past the end of its turn and is now inventing
      // the observation it is owed - everything it writes next (the "tool response", its own
      // retries, further tool calls) is hallucination that must not reach the client. Stop here,
      // exactly as for <|user|>: the client supplies the real tool results and asks again.
      Delta d;
      d.stop = true;
      pending_.push_back(d);
      done_ = true;
      buf_.clear();
      return;
    }
  }
  buf_.erase(0, pos);
}

ParsedOutput parse_output(const std::string& text, bool start_in_think) {
  OutputParser p(start_in_think);
  p.feed(text);
  p.finish();
  return p.parsed();
}

}  // namespace helios

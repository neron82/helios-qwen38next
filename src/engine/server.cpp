// OpenAI-compatible HTTP server for the Helios engine.
//
// Production surface: /health, /v1/models, /v1/metrics (and /metrics), /v1/completions,
// /v1/chat/completions. Chat requests support: system/user/assistant/tool messages, OpenAI `tools`
// (rendered into the model's <tools> block and parsed back out of <tool_call> XML), `tool_choice`,
// reasoning_effort, stop sequences, and SSE streaming that separates <think> content into
// `reasoning_content` deltas.
//
// The engine is a single-sequence runner, so one generation runs at a time: the request lock is held
// for the whole generation (including while streaming). httplib still serves other connections
// concurrently, and a client that disconnects cancels its generation.
#include "engine/runner.hpp"
#include "core/chat.hpp"
#include "tokenizer/tokenizer.hpp"
#include "json.hpp"
#include "engine/utf8.hpp"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <functional>
#include <vector>

namespace helios {

using json = nlohmann::ordered_json;

namespace {

constexpr const char* kModelId = "glm-5.3-flash-exl3";

struct ServerOpts {
  std::string api_key;          // empty = no auth
  int default_max_tokens = 32768;
  // Default reasoning effort for requests that do not set one. The model ships with "max", but at
  // that level it will spend an entire output budget deliberating and can return no answer at all,
  // for very little gain over "high" - so this engine defaults to "high" deliberately. The template
  // accepts low/high/max and coerces anything else to max, hence the validation below.
  std::string default_reasoning_effort = "high";
  int n_threads = 4;
  ChatFormat format = ChatFormat::Glm53;   // set from the checkpoint's chat_template.jinja at startup
  std::string model_id = kModelId;         // advertised in /v1/models
};
ServerOpts g_opts;
int g_ctx_cap = 0;      // KV capacity in tokens, from the runner; advertised in /v1/models
int g_slots = 1;        // sequence slots (HELIOS_SEQUENCES), after the runner's VRAM clamp
int g_slot_ctx = 0;     // context a single conversation may use, i.e. g_ctx_cap / g_slots

std::string new_id(const char* prefix) {
  static std::atomic<uint64_t> ctr{0};
  auto now = std::chrono::system_clock::now().time_since_epoch();
  uint64_t ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  char buf[64];
  snprintf(buf, sizeof(buf), "%s-%llx%04llx", prefix, (unsigned long long)ms,
           (unsigned long long)(ctr++ & 0xffff));
  return buf;
}

void error_response(httplib::Response& res, int status, const std::string& msg, const char* type) {
  res.status = status;
  json j{{"error", {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
  res.set_content(j.dump(), "application/json");
}

bool authorized(const httplib::Request& req) {
  if (g_opts.api_key.empty()) return true;
  auto it = req.headers.find("Authorization");
  if (it == req.headers.end()) return false;
  return it->second == "Bearer " + g_opts.api_key;
}

// Incremental UTF-8-safe decoding: byte-level BPE can split a codepoint across tokens, so hold back
// any incomplete trailing sequence until the next token arrives.
// Turns the token stream into text without ever emitting a partial UTF-8 character.
//
// Tokenizer::decode concatenates the byte string of each token, so decoding the whole id list is
// append-only and diffing against the previous length is sound. What is not sound is emitting a
// prefix: a byte-level BPE vocabulary can split a character across tokens, so the tail of the
// decoded bytes may be an incomplete sequence. Those bytes are held back here and emitted once
// their continuations arrive.
class Utf8Streamer {
public:
  std::string push(Tokenizer& tk, const std::vector<int>& all_ids) {
    const std::string bytes = tk.decode(all_ids, /*keep_special=*/true);
    if (bytes.size() <= prev_) return {};
    const std::string_view fresh(bytes.data() + prev_, bytes.size() - prev_);
    const size_t take = utf8_complete_prefix(fresh);
    std::string out(fresh.substr(0, take));
    prev_ += take;
    return out;
  }

private:
  size_t prev_ = 0;
};

struct ChatRequest {
  std::vector<ChatMsg> msgs;
  json tools = json::array();
  std::string reasoning_effort = "high";   // replaced from the server default by parse_chat
  bool thinking = true;
  bool tools_off = false;
  GenParams gen;
  bool stream = false;
};

// Which sequence slot a request runs on. Absent means round-robin, which is right for unrelated
// one-shot requests. Present means the client PINS the slot, and that is the only way a
// conversation stays itself: the slot holds that conversation's KV rows and recurrent state, so a
// follow-up turn has to name the same one or it is answered from a different thread's context.
// Accepted as a body field ("slot") or a header (X-Helios-Slot), because a client that already
// streams cannot easily add a body field to every call but can set a header.
//
// A pin outside [0, sequences) is a 400, never folded into range: folding would silently run the
// turn against whichever conversation happened to occupy that slot, which produces a confident
// answer to the wrong question. That is exactly the class of failure this whole feature is
// structured to make impossible elsewhere, so it must not be introduced here.
int parse_slot(const json& body, const httplib::Request& hreq, std::string& err) {
  int want = -1;
  const std::string hdr = hreq.get_header_value("X-Helios-Slot");
  if (!hdr.empty()) want = atoi(hdr.c_str());
  if (body.contains("slot")) {
    if (!body["slot"].is_number_integer()) { err = "slot must be an integer"; return -2; }
    want = body["slot"].get<int>();
  }
  return want;
}

// Why a slot pin was refused. Spells out both the request and the valid range, because "invalid
// request" on a field the client believes is optional is not an error message, it is a puzzle.
std::string slot_range_error(int want) {
  char buf[256];
  snprintf(buf, sizeof(buf),
           "slot %d is out of range: this server runs %d sequence slot%s. Omit \"slot\" to "
           "round-robin, or pin 0..%d to keep a conversation's state.",
           want, g_slots, g_slots == 1 ? "" : "s", g_slots - 1);
  return buf;
}

bool parse_common(const json& body, GenParams& p, std::string& err) {
  p.max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number()
                     ? body["max_tokens"].get<int>()
                     : g_opts.default_max_tokens;
  if (p.max_tokens <= 0) p.max_tokens = g_opts.default_max_tokens;
  p.temperature = body.value("temperature", 0.7f);
  p.top_p = body.value("top_p", 0.95f);
  p.top_k = body.value("top_k", 40);
  p.min_p = body.value("min_p", 0.0f);
  p.rep_penalty = body.value("repetition_penalty", 1.0f);
  // `seed` was never read from the body, so p.seed kept its default and every request sampled from
  // the same RNG state: two different seeds produced identical output, and a client asking for
  // reproducible sampling silently did not get it. Absent means "do not reset the sampler", which is
  // what generate() already branches on (p.seed ? p.seed : 1234).
  if (body.contains("seed") && body["seed"].is_number())
    p.seed = (uint64_t)body["seed"].get<long long>();
  p.greedy = p.temperature <= 0.01f;
  if (body.contains("stop")) {
    const json& s = body["stop"];
    if (s.is_string()) p.stop.push_back(s.get<std::string>());
    else if (s.is_array())
      for (const auto& v : s) if (v.is_string()) p.stop.push_back(v.get<std::string>());
  }
  if (body.contains("n") && body["n"].is_number() && body["n"].get<int>() > 1) {
    err = "n > 1 is not supported";
    return false;
  }
  return true;
}

std::string content_to_text(const json& c) {
  if (c.is_string()) return c.get<std::string>();
  if (c.is_array()) {   // OpenAI content parts: keep the text parts
    std::string out;
    for (const auto& part : c) {
      if (part.is_object() && part.contains("text") && part["text"].is_string())
        out += part["text"].get<std::string>();
    }
    return out;
  }
  return "";
}

bool parse_chat(const json& body, ChatRequest& out, std::string& err) {
  if (!body.contains("messages") || !body["messages"].is_array()) {
    err = "messages is required";
    return false;
  }
  for (const auto& m : body["messages"]) {
    ChatMsg msg;
    msg.role = m.value("role", "user");
    msg.content = content_to_text(m.contains("content") ? m["content"] : json(""));
    if (m.contains("reasoning_content") && m["reasoning_content"].is_string())
      msg.reasoning = m["reasoning_content"].get<std::string>();
    if (m.contains("tool_call_id") && m["tool_call_id"].is_string())
      msg.tool_call_id = m["tool_call_id"].get<std::string>();
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
      for (const auto& tc : m["tool_calls"]) {
        ToolCall c;
        c.id = tc.value("id", "");
        if (tc.contains("function")) {
          c.name = tc["function"].value("name", "");
          const json& a = tc["function"].contains("arguments") ? tc["function"]["arguments"]
                                                              : json("");
          c.arguments = a.is_string() ? a.get<std::string>() : a.dump();
        }
        msg.tool_calls.push_back(c);
      }
    }
    out.msgs.push_back(std::move(msg));
  }
  if (body.contains("tools") && body["tools"].is_array()) out.tools = body["tools"];
  if (out.tools.empty()) out.tools_off = true;
  if (body.contains("tool_choice")) {
    const json& tc = body["tool_choice"];
    if (tc.is_string() && tc.get<std::string>() == "none") out.tools_off = true;
  }
  out.reasoning_effort = g_opts.default_reasoning_effort;   // server default; request wins below
  if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string())
    out.reasoning_effort = body["reasoning_effort"].get<std::string>();
  // An unrecognised level would be coerced to the template's default, which for GLM is "max" - the
  // most expensive setting - so fall back to the server's default instead of letting a typo escalate.
  // The two formats accept disjoint sets: GLM low|high|max, Qwen xhigh|medium|low.
  const bool glm = g_opts.format == ChatFormat::Glm53;
  const bool known = glm ? (out.reasoning_effort == "low" || out.reasoning_effort == "high" ||
                            out.reasoning_effort == "max")
                         : (out.reasoning_effort == "low" || out.reasoning_effort == "medium" ||
                            out.reasoning_effort == "xhigh");
  if (!known) {
    fprintf(stderr, "[server] unknown reasoning effort '%s' (want %s); using %s\n",
            out.reasoning_effort.c_str(), glm ? "low|high|max" : "low|medium|xhigh",
            g_opts.default_reasoning_effort.c_str());
    out.reasoning_effort = g_opts.default_reasoning_effort;
  }
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
    const json& kw = body["chat_template_kwargs"];
    if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string())
      out.reasoning_effort = kw["reasoning_effort"].get<std::string>();
    if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean())
      out.thinking = kw["enable_thinking"].get<bool>();
    if (kw.contains("thinking") && kw["thinking"].is_boolean())
      out.thinking = kw["thinking"].get<bool>();
  }
  if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
    out.thinking = body["enable_thinking"].get<bool>();
  if (body.contains("thinking") && body["thinking"].is_boolean())
    out.thinking = body["thinking"].get<bool>();
  return parse_common(body, out.gen, err);
}

struct GenOutcome {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;
  int completion_tokens = 0;
  bool hit_stop = false;
  int slot = -1;              // the sequence slot this request ran on, echoed back to the client
};

}  // namespace

int run_server(Runner& runner, Tokenizer& tk, const std::string& host, int port, int n_threads,
               const std::string& api_key, int default_max_tokens,
               const std::string& default_reasoning_effort, const std::string& model_dir) {
  httplib::Server srv;
  g_opts.api_key = api_key;
  g_opts.n_threads = n_threads;
  // Pick the renderer from the checkpoint itself. Rendering a prompt in the wrong family's format
  // does not fail loudly - the model just produces degenerate output - so this must not be assumed.
  g_opts.format = detect_chat_format(model_dir);
  {
    size_t slash = model_dir.find_last_of('/');
    std::string base = slash == std::string::npos ? model_dir : model_dir.substr(slash + 1);
    if (!base.empty()) g_opts.model_id = base;
    // The effort sets are disjoint, so the default has to follow the format. "high" (the GLM default,
    // chosen because "max" burns the whole budget deliberating) is not a Qwen level at all.
    g_opts.default_reasoning_effort = g_opts.format == ChatFormat::Glm53 ? "high" : "xhigh";
    fprintf(stderr, "[server] chat format=%s model_id=%s default_effort=%s\n",
            g_opts.format == ChatFormat::Glm53 ? "glm53" : "qwen38", g_opts.model_id.c_str(),
            g_opts.default_reasoning_effort.c_str());
  }
  if (default_max_tokens > 0) g_opts.default_max_tokens = default_max_tokens;
  if (!default_reasoning_effort.empty()) g_opts.default_reasoning_effort = default_reasoning_effort;
  g_ctx_cap = runner.total_context_cap();
  g_slots = runner.sequences();
  g_slot_ctx = runner.slot_context_cap();
  // Both request paths must prepend the model's begin token. The checkpoint declares no bos_token,
  // so HF-style tokenisation of the rendered prompt omits it - but the reference engine feeds it
  // (add_bos = True), and without it the model degrades into repetition: on the same rendered chat
  // prompt the baseline answers coherently while this engine emitted "We we we we ...". The 6-token
  // parity case did not catch it because short prompts are insensitive to the begin token.
  auto encode_prompt = [&tk](const std::string& s) {
    std::vector<int> ids = tk.encode(s);
    const int begin = tk.eos_id();
    if (begin >= 0 && (ids.empty() || ids.front() != begin)) ids.insert(ids.begin(), begin);
    return ids;
  };
  static std::mutex gen_mu;          // one generation at a time (single-sequence engine)
  static std::atomic<uint64_t> served{0};

  // A stop caused by running out of KV capacity is reported as "length", like a max_tokens stop:
  // the caller must be able to tell truncation from a natural end.
  auto finish_reason = [&](int completion_tokens, int max_tokens) -> const char* {
    if (completion_tokens >= max_tokens) return "length";
    if (runner.context_cap() > 0 && runner.pos() >= runner.context_cap()) return "length";
    return "stop";
  };

  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
  });

  srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
    // Advertise the limits as well: clients that read them can size their own request controls
    // instead of guessing, and `max_tokens` here is what an omitted request field will use.
    json j{{"object", "list"},
           {"data", json::array({{{"id", g_opts.model_id},
                                  {"object", "model"},
                                  {"created", 0},
                                  {"owned_by", "helios"},
                                  {"root", g_opts.model_id},
                                  {"max_tokens", g_opts.default_max_tokens},
                                  // Per CONVERSATION, not the whole cache: with N slots the cache is
                                  // divided N ways, so advertising the undivided figure would invite
                                  // a prompt that runs off the end of a slot's own rows. On a
                                  // single-sequence server the two are the same number, so this is
                                  // unchanged from what the field has always reported.
                                  {"context_length", g_slot_ctx},
                                  // Also the undivided total, so a client that manages several
                                  // conversations can do the arithmetic itself.
                                  {"total_context_length", g_ctx_cap},
                                  // Concurrency here is INTERLEAVED, not simultaneous: N
                                  // conversations stay resident and are served one request at a
                                  // time. Reported explicitly because "parallel: 4" on its own
                                  // reads as four requests decoding at once, which this engine
                                  // cannot do.
                                  {"parallel", g_slots},
                                  {"concurrency_mode", "interleaved"},
                                  {"reasoning_effort", g_opts.default_reasoning_effort}}})}};
    res.set_content(j.dump(), "application/json");
  });

  auto metrics = [&](const httplib::Request&, httplib::Response& res) {
    const auto& t = runner.timings();
    json j{{"requests", served.load()},
           {"prefill_ms", t.prefill_ms}, {"prefill_tokens", t.prefill_tokens},
           {"decode_ms", t.decode_ms}, {"decode_tokens", t.decode_tokens},
           {"decode_tps", t.decode_ms > 0 ? t.decode_tokens * 1000.0 / t.decode_ms : 0.0},
           {"prefill_tps", t.prefill_ms > 0 ? t.prefill_tokens * 1000.0 / t.prefill_ms : 0.0},
           // Cross-request prefix cache: how much prompt work the resident history absorbed.

           {"prefix_last_resume", runner.prefix_resume()},
           {"prefix_snapshots", runner.prefix_snapshot_count()},
           {"prefix_snapshot_interval", runner.prefix_snapshot_interval()}};
    res.set_content(j.dump(), "application/json");
  };
  srv.Get("/metrics", metrics);
  srv.Get("/v1/metrics", metrics);

  // Shared generation driver: picks the sequence slot, runs the model, feeds the parser, and
  // streams or accumulates. The emit callback returns false to stop generation - that is how a
  // client that went away cancels its request instead of leaving the engine decoding a whole
  // output budget for nobody.
  //
  // The slot is acquired HERE, inside the driver, rather than in each handler: a slot is a
  // conversation, and acquiring one is what makes a follow-up turn land on its own state. Doing it
  // in one place is also what keeps the two handlers from drifting apart, since a handler that
  // forgot to switch would run a turn against whatever conversation the last request left bound.
  // `should_abort` is polled once per prefill chunk, while the model is producing nothing at all.
  // The per-token emit callback cannot cover that window: a 120k prefill is ~50 s of silent GPU
  // work, so a client that hangs up in the middle of it is not noticed until the first token is
  // produced. The caller supplies a liveness probe - for SSE that means a heartbeat write, which is
  // what makes httplib notice a dead peer at all.
  auto generate = [&](const std::vector<int>& prompt, const GenParams& p, ChatRequest& req,
                      GenOutcome& out, int want_slot,
                      std::function<bool(OutputParser::Delta&)> emit,
                      std::function<bool()> should_abort = nullptr) -> bool {
    // -1 here would collide with "round-robin", so the runner's out-of-range answer is mapped back
    // onto it explicitly rather than passed through.
    const int slot = runner.acquire_slot(want_slot);
    if (slot < 0) return false;
    out.slot = slot;
    OutputParser parser(/*start_in_think=*/req.thinking);
    Utf8Streamer utf8;
    std::vector<int> ids;
    uint64_t tok_count = 0;
    std::string visible;   // for stop-string matching on client-visible text
    bool stop_hit = false;
    runner.generate(prompt, p, [&](int tok) {
      ids.push_back(tok);
      tok_count++;
      std::string piece = utf8.push(tk, ids);
      if (piece.empty()) return true;
      std::vector<OutputParser::Delta> deltas = parser.feed(piece);
      // Mirror the think-block state into the runner: it suspends the repetition penalty while the
      // model is reasoning, because a penalty over the reasoning window leaves a reasoning model
      // circling instead of closing (content "" at rep_penalty >= 1.1).
      runner.set_in_think(parser.in_think());
      for (auto& d : deltas) {
        if (d.stop) { stop_hit = true; if (emit) emit(d); return false; }
        if (!d.content.empty()) {
          visible += d.content;
          for (const auto& s : p.stop)
            if (!s.empty() && visible.size() >= s.size() &&
                visible.compare(visible.size() - s.size(), s.size(), s) == 0) {
              stop_hit = true;
              return false;
            }
        }
        if (emit && !emit(d)) return false;   // client gone: stop decoding for nobody
      }
      return true;
    }, should_abort);
    std::vector<OutputParser::Delta> rest = parser.finish();
    for (auto& d : rest) { if (d.stop) { out.hit_stop = true; } if (emit) emit(d); }
    const ParsedOutput& parsed = parser.parsed();
    // No further bytes will arrive, so a truncated final character must be replaced rather than
    // held back - otherwise serialising it would still throw.
    out.reasoning = utf8_sanitize(parsed.reasoning);
    out.content = utf8_sanitize(parsed.content);
    out.tool_calls = parsed.tool_calls;
    out.completion_tokens = (int)tok_count;
    out.hit_stop = stop_hit;
    return true;
  };

  auto chat_handler = [&](const httplib::Request& hreq, httplib::Response& res) {
    if (!authorized(hreq)) { error_response(res, 401, "invalid api key", "invalid_request_error"); return; }
    json body;
    try {
      body = json::parse(hreq.body);
    } catch (...) {
      error_response(res, 400, "could not parse JSON body", "invalid_request_error");
      return;
    }
    ChatRequest req;
    std::string err;
    if (!parse_chat(body, req, err)) { error_response(res, 400, err, "invalid_request_error"); return; }
    req.gen.stop.clear();
    parse_common(body, req.gen, err);
    req.stream = body.value("stream", false);
    int want_slot = parse_slot(body, hreq, err);
    if (want_slot == -2) { error_response(res, 400, err, "invalid_request_error"); return; }
    const bool want_usage = body.contains("stream_options") && body["stream_options"].is_object() &&
                            body["stream_options"].value("include_usage", false);
    json tools = req.tools_off ? json::array() : req.tools;
    std::string prompt_text =
        g_opts.format == ChatFormat::Qwen38
            ? render_chat_qwen(req.msgs, tools, req.reasoning_effort,
                               /*add_generation_prompt=*/true, req.thinking)
            : render_chat(req.msgs, tools, req.reasoning_effort,
                          /*add_generation_prompt=*/true, req.thinking);
    std::vector<int> prompt = encode_prompt(prompt_text);
    const std::string id = new_id("chatcmpl");
    const int64_t created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();

    if (!req.stream) {
      GenOutcome out;
      std::lock_guard<std::mutex> lock(gen_mu);
      if (!generate(prompt, req.gen, req, out, want_slot, nullptr)) {
        error_response(res, 400, slot_range_error(want_slot), "invalid_request_error");
        return;
      }
      json msg{{"role", "assistant"}, {"content", out.content}};
      if (!out.reasoning.empty()) msg["reasoning_content"] = out.reasoning;
      if (!out.tool_calls.empty()) {
        msg["tool_calls"] = json::array();
        for (const auto& tc : out.tool_calls) {
          msg["tool_calls"].push_back({{"id", tc.id},
                                       {"type", "function"},
                                       {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
        }
      }
      const char* finish = !out.tool_calls.empty() ? "tool_calls"
                           : out.hit_stop ? "stop"
                           : finish_reason(out.completion_tokens, req.gen.max_tokens);
      json outj{{"id", id},         {"object", "chat.completion"},
                {"created", created}, {"model", g_opts.model_id},
                {"choices", json::array({{{"index", 0},
                                          {"message", msg},
                                          {"logprobs", nullptr},
                                          {"finish_reason", finish}}})},
                {"usage", {{"prompt_tokens", (int)prompt.size()},
                           {"completion_tokens", out.completion_tokens},
                           {"total_tokens", (int)prompt.size() + out.completion_tokens}}}};
      // Echoed so a client can pin its follow-up turn to the same conversation without having to
      // track the round-robin itself. Always slot 0 on a single-sequence server, which is the
      // shipped default.
      if (runner.sequences() > 1) outj["slot"] = out.slot;
      served++;
      res.set_content(outj.dump(), "application/json");
      return;
    }

    // ---- SSE streaming
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [&, prompt, req, id, created, want_usage, want_slot](size_t, httplib::DataSink& sink) mutable {
          std::lock_guard<std::mutex> lock(gen_mu);
          long long last_hb = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count();
          // A write can keep succeeding for a long time after the client is gone: the kernel
          // accepts bytes into the socket send buffer before the RST arrives, so `write`'s return
          // value alone does not notice a dead client. Measured: aborting a 3000-token stream 1.5 s
          // in left the generation running to completion and blocked the NEXT request behind it for
          // 93 s, which on a shared server means one user pressing ctrl-C stalls everyone.
          // `is_writable()` is httplib's own liveness check on the connection, and it reflects the
          // closed socket as soon as the RST lands, so consult it before each token rather than
          // relying on the write to fail.
          bool wrote_once = false;
          auto send = [&](const json& j) {
            if (!sink.is_writable()) return false;
            std::string s = "data: " + j.dump() + "\n\n";
            if (!sink.write(s.data(), s.size())) return false;
            wrote_once = true;
            return true;
          };
          // Liveness DURING prefill. `is_writable()` only learns the peer is gone when the socket is
          // polled or written, and a prefill writes nothing - so a client that cancels while a long
          // prompt is being ingested is not noticed until the first token. An SSE comment line is a
          // legal no-op that forces a write, so a heartbeat both keeps the connection warm and makes
          // the dead peer visible to is_writable() a chunk from now. The probe runs once per
          // 1024-token chunk, so a 50 s prefill is ~50 chances to notice.
          //
          // `wrote_once` matters: httplib reports is_writable() false for a sink that has not
          // written yet, so trusting it before the first write aborts the request immediately. The
          // first SSE chunk is sent before generate() is called, so by prefill time this is
          // normally already true - the guard is what keeps a reordering from silently turning
          // "cancel a long request" into "refuse every request".
          auto prefill_probe = [&]() {
            if (!wrote_once) return false;
            const char* hb = getenv("HELIOS_HB_MS");
            const long hb_ms = hb ? atol(hb) : 500;
            if (!sink.is_writable()) return true;
            const auto t = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                t.time_since_epoch()).count();
            if (ms - last_hb < hb_ms) return false;
            last_hb = ms;
            // ": keepalive\n\n" is an SSE comment: ignored by every conformant client, including
            // OpenAI SDKs, so this is invisible to the caller.
            static const char kBeat[] = ": keepalive\n\n";
            sink.write(kBeat, sizeof(kBeat) - 1);
            return !sink.is_writable();
          };
          json base{{"id", id}, {"object", "chat.completion.chunk"},
                    {"created", created}, {"model", g_opts.model_id}};
          json first = base;
          first["choices"] = json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}},
                                           {"finish_reason", nullptr}}});
          // Echoed on the FIRST chunk, not the last. A client streaming a conversation needs to know
          // which slot it is on before the first token arrives, so it can pin its follow-up turn -
          // waiting for the terminal chunk would mean holding the whole stream to find out which
          // conversation it was talking to. The buffered path reports the same field.
          // The slot is acquired HERE, before the first chunk is built, so the first chunk can
          // already say which conversation this is. A client streaming a turn needs to know its slot
          // before the first token arrives so it can pin the follow-up; reporting it on the terminal
          // chunk instead would mean holding the whole stream to find out which conversation it was
          // talking to. The buffered path reports the same field, from the same acquisition.
          const int bound = runner.acquire_slot(want_slot);
          if (bound < 0) {
            json e = base;
            e["choices"] = json::array({{{"index", 0}, {"delta", json::object()},
                                         {"finish_reason", nullptr}}});
            e["error"] = json{{"message", slot_range_error(want_slot)},
                              {"type", "invalid_request_error"}};
            const std::string se = "data: " + e.dump() + "\n\n";
            sink.write(se.data(), se.size());
            const std::string sd = "data: [DONE]\n\n";
            sink.write(sd.data(), sd.size());
            sink.done();
            return true;
          }
          if (runner.sequences() > 1) first["slot"] = bound;
          if (!send(first)) { sink.done(); return true; }
          bool alive = true;
          int tool_index = -1;
          // Already bound above, so the driver keeps this slot rather than re-deciding it.
          GenOutcome out;
          out.slot = bound;
          // `bound`, not `want_slot`: the slot is already decided and bound above, and re-deciding
          // it here would be a second round-robin step - putting a pinned request on one slot and
          // its first chunk on another.
          if (!generate(prompt, req.gen, req, out, bound, [&](OutputParser::Delta& d) -> bool {
            if (!alive) return false;   // sink already dead: let the runner wind down
            if (!d.reasoning.empty()) {
              json j = base;
              j["choices"] = json::array({{{"index", 0},
                                           {"delta", {{"reasoning_content", d.reasoning}}},
                                           {"finish_reason", nullptr}}});
              alive = send(j);
            }
            if (!d.content.empty()) {
              json j = base;
              j["choices"] = json::array({{{"index", 0}, {"delta", {{"content", d.content}}},
                                           {"finish_reason", nullptr}}});
              alive = send(j);
            }
            if (d.tool_begin) {
              tool_index++;
              json j = base;
              j["choices"] = json::array(
                  {{{"index", 0},
                    {"delta", {{"tool_calls", json::array({{{"index", tool_index},
                                                            {"id", d.tool_call_id},
                                                            {"type", "function"},
                                                            {"function", {{"name", d.tool_name},
                                                                          {"arguments", ""}}}}})}}},
                    {"finish_reason", nullptr}}});
              alive = send(j);
              if (!d.tool_args_fragment.empty() && alive) {
                json a = base;
                a["choices"] = json::array(
                    {{{"index", 0},
                      {"delta", {{"tool_calls", json::array({{{"index", tool_index},
                                                              {"function",
                                                               {{"arguments", d.tool_args_fragment}}}}})}}},
                      {"finish_reason", nullptr}}});
                alive = send(a);
              }
            }
            return alive;
          }, prefill_probe)) {
            // The slot pin was out of range. Headers are already committed by the chunked provider
            // at this point, so the error is delivered in-band as SSE rather than as an HTTP status
            // that can no longer be sent - the client sees an error event instead of a hang.
            json e = base;
            e["choices"] = json::array({{{"index", 0}, {"delta", json::object()},
                                         {"finish_reason", nullptr}}});
            e["error"] = json{{"message", slot_range_error(want_slot)},
                              {"type", "invalid_request_error"}};
            const std::string se = "data: " + e.dump() + "\n\n";
            sink.write(se.data(), se.size());
            const std::string sd = "data: [DONE]\n\n";
            sink.write(sd.data(), sd.size());
            sink.done();
            return true;
          }
          const char* finish = !out.tool_calls.empty() ? "tool_calls"
                               : out.hit_stop ? "stop"
                               : finish_reason(out.completion_tokens, req.gen.max_tokens);
          if (alive) {
            json last = base;
            last["choices"] = json::array({{{"index", 0}, {"delta", json::object()},
                                            {"finish_reason", finish}}});
            alive = send(last);
          }
          if (alive && want_usage) {
            json u = base;
            u["choices"] = json::array();
            u["usage"] = {{"prompt_tokens", (int)prompt.size()},
                          {"completion_tokens", out.completion_tokens},
                          {"total_tokens", (int)prompt.size() + out.completion_tokens}};
            alive = send(u);
          }
          if (alive) {
            const char* done = "data: [DONE]\n\n";
            sink.write(done, strlen(done));
          }
          sink.done();
          served++;
          return true;
        });
  };

  auto text_handler = [&](const httplib::Request& hreq, httplib::Response& res) {
    if (!authorized(hreq)) { error_response(res, 401, "invalid api key", "invalid_request_error"); return; }
    json body;
    try {
      body = json::parse(hreq.body);
    } catch (...) {
      error_response(res, 400, "could not parse JSON body", "invalid_request_error");
      return;
    }
    GenParams p;
    std::string err;
    if (!parse_common(body, p, err)) { error_response(res, 400, err, "invalid_request_error"); return; }
    int want_slot = parse_slot(body, hreq, err);
    if (want_slot == -2) { error_response(res, 400, err, "invalid_request_error"); return; }
    std::string text;
    if (body.contains("prompt")) {
      if (body["prompt"].is_string()) text = body["prompt"].get<std::string>();
      else if (body["prompt"].is_array() && !body["prompt"].empty() && body["prompt"][0].is_string())
        text = body["prompt"][0].get<std::string>();
    }
    std::vector<int> prompt = encode_prompt(text);
    GenOutcome out;
    std::lock_guard<std::mutex> lock(gen_mu);
    // The slot is bound before anything is generated, so a rejected pin costs no tokens and cannot
    // leave the engine mid-generation against a conversation the client did not ask for.
    if (runner.acquire_slot(want_slot) < 0) {
      error_response(res, 400, slot_range_error(want_slot), "invalid_request_error");
      return;
    }
    out.slot = runner.active_slot();
    // raw completions carry no chat template, so the model's output is taken as literal text
    OutputParser parser(/*start_in_think=*/false);
    Utf8Streamer utf8;

    // SSE. This handler previously had no streaming path at all, so `stream: true` was silently
    // ignored and the client got one buffered JSON body - which a streaming client reads as "no
    // output yet" until the whole generation finishes. The chat endpoint has always streamed; this
    // brings the completions endpoint in line with it. Deltas are text only: a raw completion has no
    // chat template and no `<think>` contract, so there is no reasoning/content split to make.
    if (body.value("stream", false)) {
      const std::string id = new_id("cmpl");
      const int64_t created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();
      const bool want_usage = body.contains("stream_options") && body["stream_options"].is_object() &&
                              body["stream_options"].value("include_usage", false);
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      // Capture BY VALUE. httplib invokes the content provider while writing the response, i.e.
      // AFTER this handler has returned, so anything captured by reference to a local here is a
      // dangling reference by the time it is read. `p` in particular is a GenParams local; reading
      // it from the provider gave a generation with a garbage max_tokens that produced no tokens.
      const GenParams gp = p;
      const int slot_id = out.slot;
      res.set_chunked_content_provider(
          "text/event-stream",
          [&, prompt, gp, slot_id, id, created, want_usage](size_t, httplib::DataSink& sink) mutable {
            // Per-provider: Utf8Streamer carries decode-position state, and the buffered path below
            // has its own instance.
            Utf8Streamer utf8;
            std::lock_guard<std::mutex> lock(gen_mu);
            // A write keeps succeeding into the socket send buffer long after the client is gone, so
            // the write's return value alone does not notice a dead peer. `is_writable()` is
            // httplib's own liveness probe on the connection; consult it per chunk.
            auto send = [&](const json& j) {
              if (!sink.is_writable()) return false;
              std::string s = "data: " + j.dump() + "\n\n";
              return sink.write(s.data(), s.size());
            };
            json base{{"id", id}, {"object", "text_completion"}, {"created", created},
                      {"model", g_opts.model_id}};
            bool alive = true;
            std::vector<int> ids;
            size_t sent = 0;   // bytes already streamed, so the held-back tail is flushed exactly once
            runner.generate(prompt, gp, [&](int tok) {
              if (!alive) return false;
              ids.push_back(tok);
              std::string piece = utf8.push(tk, ids);
              if (piece.empty()) return true;
              sent += piece.size();
              json j = base;
              j["choices"] = json::array({{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}}});
              alive = send(j);
              return true;
            });
            // Utf8Streamer holds back a trailing partial character rather than emitting half of one.
            // At end of generation that tail is still owed to the client, so emit whatever the full
            // decode holds beyond what has been sent.
            {
              const std::string all = tk.decode(ids, /*keep_special=*/true);
              if (all.size() > sent) {
                json j = base;
                j["choices"] = json::array(
                    {{{"index", 0}, {"text", all.substr(sent)}, {"finish_reason", nullptr}}});
                alive = send(j);
              }
            }
            json fin = base;
            fin["choices"] = json::array({{{"index", 0}, {"text", ""},
                                           {"finish_reason", finish_reason((int)ids.size(), gp.max_tokens)}}});
            if (runner.sequences() > 1) fin["slot"] = slot_id;
            alive = send(fin);
            if (want_usage) {
              json u = base;
              u["choices"] = json::array();
              u["usage"] = {{"prompt_tokens", (int)prompt.size()},
                            {"completion_tokens", (int)ids.size()},
                            {"total_tokens", (int)prompt.size() + (int)ids.size()}};
              alive = send(u) && alive;
            }
            const std::string sd = "data: [DONE]\n\n";
            sink.write(sd.data(), sd.size());
            sink.done();
            served++;
            return true;
          });
      return;
    }
    std::vector<int> ids;
    runner.generate(prompt, p, [&](int tok) {
      ids.push_back(tok);
      std::string piece = utf8.push(tk, ids);
      if (!piece.empty()) parser.feed(piece);
      return true;
    });
    parser.finish();
    const ParsedOutput& parsed = parser.parsed();
    out.content = parsed.content;
    out.reasoning = parsed.reasoning;
    out.completion_tokens = (int)ids.size();
    const char* finish = finish_reason(out.completion_tokens, p.max_tokens);
    json outj{{"id", new_id("cmpl")},
              {"object", "text_completion"},
              {"created", (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count()},
              {"model", g_opts.model_id},
              {"choices", json::array({{{"index", 0}, {"text", out.content}, {"finish_reason", finish}}})},
              {"usage", {{"prompt_tokens", (int)prompt.size()},
                         {"completion_tokens", out.completion_tokens},
                         {"total_tokens", (int)prompt.size() + out.completion_tokens}}}};
    if (runner.sequences() > 1) outj["slot"] = out.slot;
    served++;
    res.set_content(outj.dump(), "application/json");
  };

  srv.Post("/v1/chat/completions", chat_handler);
  srv.Post("/v1/completions", text_handler);
  srv.set_read_timeout(600, 0);
  srv.set_write_timeout(600, 0);
  srv.set_payload_max_length(32 * 1024 * 1024);
  printf("[server] model=%s listening on %s:%d (%d threads)%s\n", g_opts.model_id.c_str(), host.c_str(), port,
         n_threads, api_key.empty() ? "" : " [api key required]");
  fflush(stdout);
  if (!srv.listen(host.c_str(), port)) {
    fprintf(stderr, "[server] failed to listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}

}  // namespace helios

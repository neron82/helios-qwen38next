// helios — Qwen3.8-Flash-Next EXL3 engine
// CLI: `helios inspect <model_dir>` dumps tensor inventory + bit analysis.
#include "core/model.hpp"
#include "core/device.hpp"
#include "engine/runner.hpp"
#include "engine/sampler.hpp"
#include "engine/server.hpp"
#include "tokenizer/tokenizer.hpp"
#include <cstring>
#include "core/safetensors.hpp"
#include <cstdio>
#include <regex>
#include <map>

using namespace helios;

static const char* dtype_name(Dtype d) {
  switch (d) { case Dtype::F16: return "f16"; case Dtype::BF16: return "bf16"; case Dtype::F32: return "f32";
    case Dtype::I32: return "i32"; case Dtype::I16: return "i16"; default: return "?"; }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: helios inspect <model_dir>\n"); return 1; }

  std::string cmd = argv[1], dir = argv[2];
  if (cmd == "load" || cmd == "serve" || cmd == "gen" || cmd == "bench-m" || cmd == "bench-gdn") {
    // Prefill chunk. Every one of the 512 experts is read once per chunk regardless of how many
    // tokens the chunk holds (top-10 of 512 saturates at 2560 assignments), so the expert traffic
    // is (tokens/chunk) x 29 GB and a bigger chunk reads strictly less. Measured on a 6,802-token
    // prompt: 296.0/295.7/294.7 at 256 against 324.3/324.0/324.4 at 1024, +9.7% with a 0.5% spread.
    // 2048 is the wall, not a preference: the grouped MoE's per-chunk temp buffers then OOM at
    // 92,160 B of dynamic smem, and gpu0 is down to 1.16 GB free at 1024.
    int cap = 262144, port = 8080, max_chunk = 1024;
    // Default output length for requests that omit `max_tokens`. The engine imposes no output cap of
    // its own beyond the KV capacity, so this only bounds what an omission produces.
    int default_max_tokens = getenv("HELIOS_MAX_TOKENS") ? atoi(getenv("HELIOS_MAX_TOKENS")) : 0;
    // Default reasoning effort (low|high|max) for requests that do not send one.
    std::string reasoning_effort = getenv("HELIOS_REASONING_EFFORT") ? getenv("HELIOS_REASONING_EFFORT") : "";
    // MTP speculative decoding: ON by default, --no-mtp / HELIOS_MTP=0 turns it off. The measured
    // basis is in Runner::init (the 8k A/B table); in short, K=1 accepts 65% of draft slots and
    // lifts 8k decode from 47.9 to 56.5 tok/s, and the output is byte-identical run to run.
    //
    // The one thing it changes is bit-exactness against the sequential path: a batched verify
    // forward reduces differently from a width-1 one, so greedy text drifts off the non-speculative
    // stream after a few dozen tokens. HELIOS_MTP=0 (or --no-mtp) restores it, which is also what
    // the reference-digest regression check needs.
    const char* me = getenv("HELIOS_MTP");
    bool use_mtp = me ? atoi(me) != 0 : true;
    const char* host = "127.0.0.1";
    std::string api_key;
    bool g_pair = false;
    bool ram_only = false;
    bool raw_prompt = false;   // BOS + raw text, no chat template
    std::vector<int> gen_ids;
    std::string gen_text;
    int max_tokens = 64;
    float temperature = 0.7f;
    // --repeat N runs the same prompt N times in ONE process, which is what the cross-request prefix
    // cache (HELIOS_PREFIX_CACHE=1) needs to be exercised at all: it reuses across requests, and a
    // fresh engine per invocation has no history to reuse. Iterations after the first append
    // --repeat-suffix, so a long shared prefix with a DIFFERENT tail can be measured - the shape a
    // chat turn actually has. Absent (the default), the command is unchanged: one request.
    int repeat = 1;
    std::string repeat_suffix;
    // Sequence slot for `gen`. --slot N pins every request to slot N; absent, requests round-robin
    // over HELIOS_SEQUENCES slots. --slots-file runs an explicit plan instead, one
    // "<slot> <prompt-file>" per line, which is how the interleaving is driven: it is the only way
    // to put several conversations through ONE process in a chosen order, and a fresh process per
    // request cannot see the state a previous conversation left behind - which is the entire class
    // of bug a multi-request test exists to catch.
    int slot = -1;
    std::string slots_file;
    for (int i = 3; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--ram-only") ram_only = true;
      else if (a == "--cap" && i + 1 < argc) cap = atoi(argv[++i]);
      else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
      else if (a == "--host" && i + 1 < argc) host = argv[++i];
      else if (a == "--api-key" && i + 1 < argc) api_key = argv[++i];
      else if (a == "--max-tokens" && i + 1 < argc) default_max_tokens = atoi(argv[++i]);
      else if (a == "--reasoning-effort" && i + 1 < argc) reasoning_effort = argv[++i];
      else if (a == "--no-mtp") use_mtp = false;
      else if (a == "--mtp") use_mtp = true;
      else if (a == "--chunk" && i + 1 < argc) max_chunk = atoi(argv[++i]);
      else if (a == "--tokens" && i + 1 < argc) max_tokens = atoi(argv[++i]);
      else if (a == "--temp" && i + 1 < argc) temperature = atof(argv[++i]);
      else if (a == "--prompt" && i + 1 < argc) gen_text = argv[++i];
      else if (a == "--raw") raw_prompt = true;
      else if (a == "--pair") g_pair = true;
      else if (a == "--ids" && i + 1 < argc) {
        // Raw token ids, bypassing the chat template. Needed for parity: the reference is driven
        // with a bare BOS + text, while `--prompt` renders the model's chat template.
        std::string cur;
        for (const char* c = argv[++i]; *c; c++) {
          if (*c == ',') { if (!cur.empty()) { gen_ids.push_back(atoi(cur.c_str())); cur.clear(); } }
          else cur += *c;
        }
        if (!cur.empty()) gen_ids.push_back(atoi(cur.c_str()));
      }
      else if (a == "--prompt-file" && i + 1 < argc) {
        std::string path = argv[++i];
        FILE* pf = fopen(path.c_str(), "rb");
        if (!pf) { fprintf(stderr, "cannot open prompt file %s\n", path.c_str()); return 2; }
        std::string raw; char buf[1 << 16]; size_t got;
        while ((got = fread(buf, 1, sizeof(buf), pf)) > 0) raw.append(buf, got);
        fclose(pf);
        size_t t = raw.find_first_not_of(" \t\r\n");
        if (t != std::string::npos) raw = raw.substr(t);
        gen_text = raw;
      }
      else if (a == "--slot" && i + 1 < argc) slot = atoi(argv[++i]);
      else if (a == "--slots-file" && i + 1 < argc) slots_file = argv[++i];
      else if (a == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
      else if (a == "--repeat-suffix" && i + 1 < argc) repeat_suffix = argv[++i];
    }
    // --ram-only deliberately touches no GPU: it is the offline loader check, runnable while the
    // baseline server holds both cards.
    if (!ram_only && !Engine::instance().init()) { fprintf(stderr, "engine init failed\n"); return 2; }
    Model m;
    if (!m.load(dir, ram_only, true)) { fprintf(stderr, "model load failed\n"); return 3; }
    if (ram_only) {
      printf("LOAD OK: layers=%zu experts=%d tensors ok, no GPU touched\n",
             m.layers.size(), m.cfg.n_expert);
      return 0;
    }
    bool do_gen = (cmd == "gen");
    Tokenizer tk;
    if (!tk.load(dir)) { fprintf(stderr, "tokenizer load failed\n"); return 4; }
    Runner runner;
    if (!runner.init(m, cap, max_chunk)) return 5;
    runner.set_stops(tk.stop_ids());
    runner.set_mtp(use_mtp);
    if (cmd == "load") {
      printf("LOAD OK: layers=%zu experts=%d ctx=%d mtp=%d gpu0=%.2fGB gpu1=%.2fGB\n",
             m.layers.size(), m.cfg.n_expert, cap, (int)use_mtp, (double)m.gpu0_bytes/GiB,
             (double)m.gpu1_bytes/GiB);
      return 0;
    }
    if (cmd == "bench-m") {
      // Width sweep for the batched-verification decision. Reports ms per CALL and the amortised
      // ms per token, because the speculation arithmetic needs both: a width-n call that costs
      // n x ms_per_token_of_M1 has bought nothing, and the whole plan rests on where that crosses.
      printf("[bench] gate=%d max_chunk=%d ctx=%d\n", g_moe_decode_max_n, max_chunk, cap);
      double base = 0.0;
      for (int n : {1, 2, 3, 4, 5}) {
        double ms = runner.bench_chunk(n, 12, 3);
        if (n == 1) base = ms;
        printf("[bench] n=%d  %7.3f ms/call  %7.3f ms/token  %5.2fx vs M=1\n",
               n, ms, ms / n, base / (ms / n));
        fflush(stdout);
      }
      return 0;
    }
    if (cmd == "bench-gdn") {
      // Gates batched verification: a partial accept rewinds the GDN recurrence by replaying the
      // accepted prefix, and if that replay is not bit-exact every later token silently diverges.
      bool all = true;
      for (int w : {1, 2, 3, 4}) {
        unsigned a = 0, b = 0;
        bool ok = runner.gdn_replay_selftest(w, &a, &b);
        printf("[gdn] replay width=%d  %s  fp_before=%08x fp_after=%08x\n", w,
               ok ? "BIT-EXACT" : "MISMATCH", a, b);
        all = all && ok;
      }
      printf("[gdn] %s\n", all ? "REPLAY SELFTEST PASS" : "REPLAY SELFTEST FAIL");
      return all ? 0 : 1;
    }
    {
      size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
      cudaSetDevice(Engine::instance().gpu(0).phys_idx()); cudaMemGetInfo(&f0, &t0);
      cudaSetDevice(Engine::instance().gpu(1).phys_idx()); cudaMemGetInfo(&f1, &t1);
      cudaSetDevice(Engine::instance().gpu(0).phys_idx());
      printf("[mem] gpu0 used %.2f GB free %.2f (pool %.2f) | gpu1 used %.2f GB free %.2f (pool %.2f)\n",
             (t0 - f0) / 1073741824.0, f0 / 1073741824.0,
             Engine::instance().gpu(0).pool_used() / 1073741824.0,
             (t1 - f1) / 1073741824.0, f1 / 1073741824.0,
             Engine::instance().gpu(1).pool_used() / 1073741824.0);
    }
    if (do_gen) {
      GenParams p;
      p.max_tokens = max_tokens; p.temperature = temperature;
      // An explicit plan, when given, REPLACES the --repeat loop. Each line is
      // "<slot> <prompt-file> [tokens]", run in file order through this one process, so several
      // conversations interleave inside a single engine - the only arrangement in which one
      // conversation's leftover state is even visible to the next.
      if (!slots_file.empty()) {
        FILE* pf = fopen(slots_file.c_str(), "rb");
        if (!pf) { fprintf(stderr, "cannot open %s\n", slots_file.c_str()); return 2; }
        char line[8192];
        int n = 0;
        while (fgets(line, sizeof(line), pf)) {
          if (line[0] == '#' || line[0] == '\n') continue;
          int want = -1, want_tok = max_tokens;
          char path[4096] = {0};
          if (sscanf(line, "%d %4095s %d", &want, path, &want_tok) < 2) continue;
          FILE* rf = fopen(path, "rb");
          if (!rf) { fprintf(stderr, "cannot open prompt file %s\n", path); fclose(pf); return 2; }
          std::string raw; char buf[1 << 16]; size_t got;
          while ((got = fread(buf, 1, sizeof(buf), rf)) > 0) raw.append(buf, got);
          fclose(rf);
          // Both ends stripped. A trailing newline is a real token, and for a bare instruction it
          // changes the answer completely - the same text with and without it differed by 20 tokens
          // versus an immediate end-of-turn, which reads as a routing bug and is not one.
          size_t t = raw.find_first_not_of(" \t\r\n");
          if (t != std::string::npos) raw = raw.substr(t);
          size_t e = raw.find_last_not_of(" \t\r\n");
          if (e != std::string::npos) raw = raw.substr(0, e + 1);
          std::vector<int> ids = tk.encode(raw);
          ids.insert(ids.begin(), tk.eos_id());
          const int bound = runner.acquire_slot(want);
          if (bound < 0) {
            fprintf(stderr, "slot %d out of range (%d slots)\n", want, runner.sequences());
            fclose(pf);
            return 2;
          }
          printf("[gen] request %d slot=%d prompt_tokens=%zu max_tokens=%d\n", ++n, bound,
                 ids.size(), want_tok);
          fflush(stdout);
          GenParams pp = p;
          pp.max_tokens = want_tok;
          runner.generate(ids, pp, [&](int tok) {
            printf("%s", tk.decode({tok}).c_str());
            fflush(stdout);
            return true;
          });
          printf("\n");
          fflush(stdout);
        }
        fclose(pf);
        return 0;
      }
      for (int it = 0; it < repeat; it++) {
        if (repeat > 1) printf("--- request %d ---\n", it + 1);
        const std::string text =
            gen_text.empty() ? "Hello!" : gen_text + (it ? repeat_suffix : std::string());
        std::vector<ChatMsg> msgs = {{"user", text}};
        std::vector<int> ids;
        if (!gen_ids.empty()) {
          ids = gen_ids;
        } else if (raw_prompt) {
          // Plain completion: the literal text prefixed with the begin token, matching what the
          // baseline feeds the same /v1/completions request. This checkpoint declares no bos_token
          // (add_bos_token = false), so the begin token is the tokenizer's eos/<|im_end|> id - the
          // same id exllamav3's add_bos = True prepends.
          ids = tk.encode(text);
          ids.insert(ids.begin(), tk.eos_id());
        } else {
          std::string prompt = tk.apply_chat_template(msgs, true);
          ids = tk.encode(prompt);
        }
        const int bound = runner.acquire_slot(slot);
        if (bound < 0) {
          fprintf(stderr, "slot %d out of range (%d slots)\n", slot, runner.sequences());
          return 2;
        }
        // HELIOS_BATCH_PARITY=1: check that decoding two sequences PAIRED (one row each, one
        // forward) produces exactly what decoding them one after the other produces. Both runs are
        // driven by the SAME fixed tokens and compared on the argmax after every step, so a row that
        // read another sequence's recurrent state, position or KV base diverges at step 1.
        if (getenv("HELIOS_BATCH_PARITY") && repeat >= 2) {
          // HELIOS_BATCH_SAME=1 feeds BOTH slots the identical prompt, so the two rows of a paired
          // forward must come out bit-identical. Any difference is a per-row addressing bug with no
          // sequence difference to confound it.
          std::vector<int> ids2(getenv("HELIOS_BATCH_SAME")
                                    ? ids
                                    : std::vector<int>(ids.begin(), ids.begin() + ids.size() / 2 + 1));
          const int N = 10;
          // Teacher-force each model on ITS OWN continuation rather than on tokens scraped from the
          // prompt. Feeding arbitrary tokens drove both runs into a near-EOS state where the top two
          // logits are nearly tied, so any last-bit difference in a differently-tiled M=2 GEMM flips
          // the argmax and the test reports a mismatch that is not a correctness bug. Following the
          // model's own path is both a stronger test and a stable one.
          auto run_serial = [&](int slot, const std::vector<int>& q) {
            runner.bind_slot_pub(slot);
            runner.harness_reset();
            runner.harness_prefill(q);
            std::vector<int> traj;
            int t = runner.harness_argmax(0);
            traj.push_back(t);
            for (int k = 1; k < N; k++) {
              runner.harness_commit(t);
              if (getenv("HELIOS_BATCH_SAME") && k <= 2)
                runner.harness_dump_last((slot == 0 ? "serialA" : "serialB"));
              t = runner.harness_argmax(0); traj.push_back(t);
            }
            return traj;
          };
          std::vector<int> ta = run_serial(0, ids);
          std::vector<int> tb = run_serial(1, ids2);

          // The first prediction of each sequence has to be taken from its OWN prefill, because the
          // logits buffer is shared and slot 1's prefill overwrites slot 0's.
          runner.bind_slot_pub(0); runner.harness_reset(); runner.harness_prefill(ids);
          int na = runner.harness_argmax(0); runner.save_slot_pub(0);
          runner.bind_slot_pub(1); runner.harness_reset(); runner.harness_prefill(ids2);
          int nb = runner.harness_argmax(0); runner.save_slot_pub(1);
          runner.bind_slot_pub(0);

          // Do the two slots hold identical GDN state after identical prefills?
          for (int lay : {0, 1, 2, 10}) {
            runner.harness_fused2_ab(2);
            runner.harness_proj_ab(2);
            runner.harness_norm_rows_check(96, 128);
            printf("[gen] full GDN layer bsz=2 vs 2x bsz=1: worst abs diff %.6e\n",
                   runner.harness_gdn_layer_ab(2));
            runner.harness_state_ptr_gap_pub(0, 1, lay);
            double g = runner.harness_state_gap_pub(0, 1, lay);
            printf("[gen] state gap slots 0/1 layer %d: max|diff| = %.6e  %s\n", lay, g,
                   g == 0.0 ? "IDENTICAL" : "DIFFER");
          }

          std::vector<int> pa{na}, pb{nb};
          for (int k = 1; k < N; k++) {
            if (!runner.decode_pair(na, nb, 0, 1)) { printf("[gen] decode_pair declined at %d\n", k); break; }
            if (k <= 2) runner.harness_dump_last("paired");
            int r0 = runner.harness_argmax(0), r1 = runner.harness_argmax(1);
            // Row mapping settled EMPIRICALLY rather than by reading a comment: final_head computes
            // rows = min(n, head_rows) and projects sub_in + (n - rows), so with n == head_rows == 2
            // there is no reversal at all and row i is sub_in row i. HELIOS_BATCH_ROWSWAP flips it so
            // the two readings can be compared rather than argued about.
            if (getenv("HELIOS_BATCH_ROWSWAP")) { int t = r0; r0 = r1; r1 = t; }
            na = r0;
            nb = r1;
            pa.push_back(na); pb.push_back(nb);
          }
          bool oka = (ta == pa), okb = (tb == pb);
          printf("[gen] batch-parity A %s  B %s\n", oka ? "MATCH" : "MISMATCH",
                 okb ? "MATCH" : "MISMATCH");
          if (!oka || !okb) {
            printf("[gen]   serial A:"); for (size_t i = 0; i < ta.size() && i < 6; i++) printf(" %d", ta[i]);
            printf("\n[gen]   paired A:"); for (size_t i = 0; i < pa.size() && i < 6; i++) printf(" %d", pa[i]);
            printf("\n[gen]   serial B:"); for (size_t i = 0; i < tb.size() && i < 6; i++) printf(" %d", tb[i]);
            printf("\n[gen]   paired B:"); for (size_t i = 0; i < pb.size() && i < 6; i++) printf(" %d", pb[i]);
            printf("\n");
            return 3;
          }
          printf("[gen] batch-parity OK (%d steps each)\n", N);
          continue;
        }
        printf("[gen] request %d/%d slot=%d prompt_tokens=%zu\n", it + 1, repeat, bound, ids.size());
        if (g_pair && it == 0 && repeat >= 2) {
          // Two sequences, one forward per decode step. The second sequence's prompt is the same text
          // truncated, so both are real requests; the point is that both advance through ONE trunk
          // pass, which is what decode batching buys.
          std::vector<int> ids2(ids.begin(), ids.begin() + ids.size() / 2 + 1);
          std::vector<int> oa, ob;
          double t0 = (double)clock() / CLOCKS_PER_SEC;
          bool ok = runner.generate_pair(ids, ids2, p, &oa, &ob);
          double dt = (double)clock() / CLOCKS_PER_SEC - t0;
          if (!ok) { printf("[gen] generate_pair declined\n"); return 4; }
          printf("\n[gen-paired] %zu + %zu tokens in %.2fs -> %.1f tok/s aggregate (%.1f each)\n",
                 oa.size(), ob.size(), dt, dt > 0 ? (oa.size() + ob.size()) / dt : 0.0,
                 dt > 0 ? (oa.size() + ob.size()) / dt / 2 : 0.0);
          if (getenv("HELIOS_PAIR_DBG")) {
            printf("[gen] A ids:"); for (size_t i = 0; i < oa.size() && i < 8; i++) printf(" %d", oa[i]);
            printf("\n[gen] B ids:"); for (size_t i = 0; i < ob.size() && i < 8; i++) printf(" %d", ob[i]);
            printf("\n");
          }
          printf("[gen] seq A: %s\n", tk.decode(oa).substr(0, 120).c_str());
          printf("[gen] seq B: %s\n", tk.decode(ob).substr(0, 120).c_str());
          return 0;
        }
        double t0 = (double)clock() / CLOCKS_PER_SEC;
        auto out = runner.generate(ids, p, [&](int tok) {
          printf("%s", tk.decode({tok}).c_str());
          fflush(stdout);
          return true;
        });
        double dt = (double)clock() / CLOCKS_PER_SEC - t0;
        const Runner::Timings& tm = runner.timings();
        printf("\n[gen] %zu tokens in %.2fs (%.2f tok/s)  prefill %.1f tok/s decode %.2f tok/s",
               out.size(), dt, dt > 0 ? out.size() / dt : 0.0,
               tm.prefill_ms > 0 ? tm.prefill_tokens * 1000.0 / tm.prefill_ms : 0.0,
               tm.decode_ms > 0 ? tm.decode_tokens * 1000.0 / tm.decode_ms : 0.0);
        // Only meaningful with the cache on; resume is 0 on every path when it is off.
        printf("  [prefix] resume=%d of %d (%.1f ms prefill of %d tokens recomputed)\n",
               runner.prefix_resume(), (int)ids.size(), tm.prefill_ms, tm.prefill_tokens);
        if (tm.spec_steps > 0)
          printf("[gen] spec: %ld/%ld draft slots accepted (%.1f%%), %.2f tokens per step\n",
                 tm.spec_accepts, tm.spec_slots, 100.0 * tm.accept_rate(),
                 (double)(tm.spec_steps + tm.spec_accepts) / (double)tm.spec_steps);
        if (runner.prefix_snapshot_count() > 0)
          printf("[gen] prefix: %lld requests, %lld tokens reused, %d captures x %d tokens\n",
                 runner.prefix_requests(), runner.prefix_reuse_total(),
                 runner.prefix_snapshot_count(), runner.prefix_snapshot_interval());
        fflush(stdout);
      }
      return 0;
    }
    return run_server(runner, tk, host, port, 8, api_key, default_max_tokens, reasoning_effort, dir);
  }
  bool dump_mode = cmd == "dump";
  if (cmd != "inspect" && !dump_mode) { fprintf(stderr, "usage: helios inspect|dump <model_dir> [substr]\n"); return 1; }
  ShardSet ss;
  const char* filt = argc > 3 ? argv[3] : nullptr;

  ss.load_dir(dir);
  if (!dump_mode) printf("shards=%d tensors=%zu total=%.2f GB\n", ss.shard_count(), ss.tensors().size(), ss.total_bytes()/1e9);
  std::map<std::string, uint64_t> cat_bytes;
  std::map<std::string, std::pair<uint64_t,uint64_t>> trellis_bits; // category -> (bits_total, params)
  std::map<int, int> layer_shard;
  int n_trellis = 0;
  for (auto* t : ss.tensors()) {
    if (dump_mode) {
      if (!filt || t->name.find(filt) != std::string::npos) {
        printf("%-72s %-4s [", t->name.c_str(), dtype_name(t->dtype));
        for (auto d : t->shape) printf("%lld,", (long long)d);
        printf("] %zu B\n", t->bytes);
      }
      continue;
    }
    std::string cat = "other";
    std::smatch m;
    std::regex layer_re("model\\.language_model\\.layers\\.(\\d+)\\.(.+)");
    if (std::regex_match(t->name, m, layer_re)) {
      int layer = std::stoi(m[1]);
      std::string rest = m[2];
      layer_shard[layer] = t->shard;
      if (rest.find("experts.") != std::string::npos && rest.find("shared_experts") == std::string::npos) {
        cat = "routed_experts";
      } else if (rest.find("shared_experts.") == 0) cat = "shared_experts";
      else if (rest.find("self_attn.") == 0) cat = "attention";
      else if (rest.find("mlp.gate.") == 0) cat = "router";
      else if (rest.find("mlp.") == 0) cat = "dense_mlp";
      else if (rest.find("hc_") == 0 || rest.find("input_layernorm") == 0 || rest.find("post_attention") == 0) cat = "mhc_norms";
      else cat = "layer_other";
    } else if (t->name.find("embed_tokens") != std::string::npos) cat = "embed";
    else if (t->name.find("lm_head") != std::string::npos) cat = "lm_head";
    else if (t->name.find("vision") != std::string::npos) cat = "vision";
    else if (t->name.find("mtp") != std::string::npos || t->name.find("nextn") != std::string::npos) cat = "mtp";
    cat_bytes[cat] += t->bytes;

    if (t->name.size() > 8 && t->name.substr(t->name.size()-8) == ".trellis") {
      n_trellis++;
      // infer logical matrix dims from sibling svh/suh lengths
      std::string base = t->name.substr(0, t->name.size()-8);
      uint64_t out = 0, in = 0;
      if (auto* svh = ss.find(base + ".svh")) out = svh->elems;
      if (auto* suh = ss.find(base + ".suh")) in = suh->elems;
      uint64_t syms = t->elems;
      if (out && in) {
        std::string cat2 = "routed_experts";
        if (base.find("shared_experts") != std::string::npos) cat2 = "shared_experts";
        else if (base.find("self_attn") != std::string::npos) cat2 = "attention";
        else if (base.find("down_proj") != std::string::npos && base.find("experts") == std::string::npos) cat2 = "dense_mlp";
        auto& acc = trellis_bits[cat2];
        acc.first += syms * 16; acc.second += out * in;
      }
    }
  }
  printf("categories (GB):\n");
  for (auto& [k, v] : cat_bytes) printf("  %-16s %7.2f GB\n", k.c_str(), v/1e9);
  printf("trellis tensors=%d\nbits-per-param by category:\n", n_trellis);
  for (auto& [k, v] : trellis_bits) printf("  %-16s %.3f bpw (%zu params)\n", k.c_str(), (double)v.first/v.second, v.second);
  printf("layer->shard map:\n");
  int prev = -1;
  for (auto& [l, s] : layer_shard) if (s != prev) { printf("  L%-2d -> S%d\n", l, s); prev = s; }
  return 0;
}
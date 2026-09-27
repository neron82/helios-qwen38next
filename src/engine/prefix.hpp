#pragma once
// Cross-request prefix-cache policy, as pure logic.
//
// This is where a prefix cache goes wrong quietly. Resuming past the shared prefix serves tokens the
// caches never computed; resuming past a capture the prompt does not match restores recurrent state
// (GDN matrices, PLE conv, PLE n-gram window) that belongs to different tokens. Neither shows up as
// a crash - both just quietly change the answer - so the decision is unit-tested here (test_prefix)
// instead of being inferred from generated text.
#include <algorithm>
#include <vector>

namespace helios {

// Plan one request against the captures the ring holds. `snap_pos` is the position each capture was
// taken at (-1 = empty) and is UPDATED IN PLACE: a capture is only good for the token prefix it was
// taken on, so every capture past the resume point is dropped. That is not tidiness, it is the
// correctness rule - a capture at position Q survives a request only if the new prompt still agrees
// with the history on [0, Q), and the plan is by construction the last point where that holds.
//
//   common     - leading prompt tokens that match the resident history (Runner::prefix_match)
//   prompt_len - the new prompt's length
//
// Returns the slot to restore, or -1 to recompute the whole prompt. The result is never the
// prompt's own length: the last prompt token is always re-run, because it is the one the model has
// to see, and because it guarantees at least one row through the forward when the whole prompt is
// already resident.
inline int prefix_plan(int common, int prompt_len, std::vector<int>& snap_pos) {
  const int lim = prompt_len > 0 ? std::min(common, prompt_len - 1) : 0;
  int at = 0, best = -1;
  for (size_t i = 0; i < snap_pos.size(); i++) {
    if (snap_pos[i] < 0 || snap_pos[i] > lim) continue;
    if (snap_pos[i] > at) { at = snap_pos[i]; best = (int)i; }
  }
  // Everything above the resume point describes a prefix this request has just disowned.
  for (size_t i = 0; i < snap_pos.size(); i++)
    if (snap_pos[i] > at) snap_pos[i] = -1;
  return best;
}

}  // namespace helios

#pragma once
// HELIOS_SEQUENCES policy, as pure logic.
//
// The engine is single-sequence by design and by measurement: it has ONE activation set, one
// device-side recurrent state per layer and one logits buffer, so two forwards cannot be in
// flight at the same time without a redesign of every kernel's scratch plumbing. What IS
// achievable, and what this file governs, is keeping N conversations ALIVE at once: each slot
// owns its own slice of the (already-allocated) position-addressed KV cache and its own copy of
// the small persistent recurrent state, so a server can hold several open threads and interleave
// them a request at a time.
//
// The decisions that can be got wrong quietly live here rather than in the runner:
//   * how many slots actually FIT. Resident recurrent state is ~56 MB per card per slot at the
//     shipped 262144 context, and card 1 is down to ~0.6 GB free there - so an unbounded N is an
//     OOM at allocate time, several seconds into loading 31 GB of experts. It is clamped here
//     against measured free VRAM and the clamp is reported, not silently applied.
//   * which slot a request lands on. A pinned slot is a conversation; an unpinned one is
//     round-robin, and the two are not interchangeable (see seq_pick).
//
// This is the same split as prefix.hpp: policy is unit-testable without a GPU or a checkpoint,
// so it is tested there rather than inferred from generated text.
#include <cstddef>

namespace helios {

// What one ADDITIONAL slot costs, per card, in device bytes: that card's share of the GDN
// convolution + recurrence state, its PLE dilated conv if the PLE layer lives there, and the MTP
// carry tap. The KV cache is NOT in this number and must not be: it is already allocated in full
// at init and a slot only partitions it (each slot gets ctx/N rows), so a slot adds no KV bytes at
// all. Getting that wrong is what would make N look unaffordable when it is not.
struct SeqCost {
  size_t per_card[2] = {0, 0};
  void add(int card, size_t bytes) { if (card >= 0 && card < 2) per_card[card] += bytes; }
  size_t total() const { return per_card[0] + per_card[1]; }
};

// The most slots that fit, given measured free bytes per card. Slot 0 is the state init() has
// ALREADY allocated, so it is free; every further slot costs that card's per-slot bytes. `reserve`
// is held back from each card so a slot count that exactly fills VRAM does not leave the engine
// unable to allocate the next forward's temporaries.
//
// The answer is the MINIMUM over the cards that actually hold per-slot state: a slot is only real
// if every piece of its state has somewhere to live, so the emptiest card binds it. A card with no
// per-slot state is SKIPPED rather than treated as the limit - it cannot constrain anything, and
// folding its free bytes in as though they were usable is how a perfectly affordable N gets
// reported as impossible.
inline int seq_fit_slots(size_t free0, size_t per0, size_t free1, size_t per1, size_t reserve) {
  const size_t f[2] = {free0, free1}, p[2] = {per0, per1};
  int n = 0;
  bool any = false;
  for (int d = 0; d < 2; d++) {
    if (!p[d]) continue;                 // this card holds no per-slot state, so it cannot bind
    const size_t avail = f[d] > reserve ? f[d] - reserve : 0;
    const int fit = 1 + (int)(avail / p[d]);
    if (!any || fit < n) { n = fit; any = true; }
  }
  // No card holds per-slot state at all: nothing constrains the count, so it is left to the
  // context floor. Returning 0 here would be a "0 slots" engine, which is not a more conservative
  // answer, it is a broken one.
  return (any && n < 1) ? 1 : n;
}

// Reconcile the requested slot count with what fits and with a usable context per slot. Order
// matters: the VRAM clamp is applied first, then the context floor, so the reported number is
// always one that was actually built rather than one that was asked for and trimmed afterwards.
// A context floor exists because N slots share one fixed-size KV cache - halving the slots doubles
// each slot's context, and a slot with a 200-token context is not a conversation.
inline int seq_clamp_slots(int requested, int fit, int ctx_total, int min_slot_ctx) {
  int n = requested < 1 ? 1 : requested;
  if (n > fit) n = fit;
  while (n > 1 && ctx_total / n < min_slot_ctx) n--;
  return n < 1 ? 1 : n;
}

// Rows each slot may use. The KV cache is one flat buffer of ctx_total rows; a slot owns a
// contiguous stride of it, so the per-slot capacity is the floor division. Any remainder rows
// are simply unused - rounding DOWN is deliberate, because a slot's last row must exist inside
// the allocation, not one past its end.
inline int seq_slot_ctx(int ctx_total, int n_slots) {
  if (n_slots <= 1) return ctx_total;
  const int per = ctx_total / n_slots;
  return per > 0 ? per : 1;
}

// Which slot a request runs on.
//
//   want >= 0        - the client PINNED a slot. That is how a conversation stays itself: its
//                      KV rows and recurrent state live in that slot, so a follow-up turn must
//                      name the same one. A pin outside [0, n_slots) is rejected (-1) rather than
//                      silently folded, because folding it would run the turn against the wrong
//                      conversation's state and answer with another thread's context.
//   want < 0         - no pin: round-robin. Correct for UNRELATED single-shot requests, where
//                      each request begins its own sequence anyway. `rr` advances so consecutive
//                      unpinned requests do not all land on slot 0 and serialise behind a
//                      conversation that is being kept alive.
//
// One sequence (the default) is slot 0 and nothing else: the round-robin cursor does not advance
// and a pin of 0 is the same request, so the flag-off path is untouched.
inline int seq_pick(int want, int n_slots, int& rr) {
  // The pin is validated BEFORE the single-sequence shortcut. Testing `n_slots <= 1` first would
  // accept a pin of 5 on a one-slot engine and quietly answer it as slot 0 - the exact fold this
  // function exists to refuse, and an invisible one, because slot 0 is a legitimate answer to give.
  // On the shipped path this changes nothing: no pin (want < 0) and a pin of 0 both resolve to 0,
  // so the flag-off request is the same request it always was.
  if (want >= 0) return want < n_slots ? want : -1;
  if (n_slots <= 1) return 0;
  const int s = rr % n_slots;
  rr = (rr + 1) % n_slots;
  return s;
}

}  // namespace helios

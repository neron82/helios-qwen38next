// HELIOS_SEQUENCES scheduling policy: how many slots fit, how the KV cache is divided between
// them, and which slot a request lands on.
//
// These are the three decisions whose failure modes are all SILENT. Too many slots does not fail
// the feature loudly - it either OOMs thousands of seconds into loading 31 GB of experts, or (worse,
// on a bump allocator) steals headroom the next forward needs. A wrong per-slot context budget lets
// one conversation write KV rows that belong to another. And a misrouted slot answers a client's
// question from a different thread's state while looking entirely healthy. So the arithmetic is
// pinned here, with no GPU and no checkpoint, instead of being inferred from served text.
#include "engine/seq.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace helios;

static int failures = 0;

static void expect(int got, int want, const char* what) {
  if (got == want) return;
  printf("  FAIL %-58s got %d want %d\n", what, got, want);
  failures++;
}

static void expect_size(size_t got, size_t want, const char* what) {
  if (got == want) return;
  printf("  FAIL %-58s got %zu want %zu\n", what, got, want);
  failures++;
}

int main() {
  // ---- how many slots fit ----
  //
  // Slot 0 is already allocated by the time the clamp runs, so it is free; the answer is
  // 1 + whatever the remaining free bytes buy. 58 MB a slot is the measured per-card figure for
  // this model at the shipped configuration (18 GDN layers x 3.13 MB of recurrence, plus conv).
  {
    const size_t slot = 58ull * 1024 * 1024;
    const size_t reserve = 320ull * 1024 * 1024;
    // 1.4 GB free on card 0 with nothing on card 1: only the card that actually holds state can
    // bind the answer. A card with no per-slot state must not constrain it - counting an empty
    // card's free bytes as if they were usable is what would make N look unaffordable here.
    expect(seq_fit_slots(1400ull * 1024 * 1024, slot, 600ull * 1024 * 1024, 0, reserve),
           1 + (size_t)((1400ull * 1024 * 1024 - reserve) / slot), "one populated card binds the fit");
    // Both cards populated, and card 1 is the tighter one: the MINIMUM is what fits, because a slot
    // is only real if every piece of its state has somewhere to live.
    expect(seq_fit_slots(1400ull * 1024 * 1024, slot, 600ull * 1024 * 1024, slot, reserve),
           1 + (size_t)((600ull * 1024 * 1024 - reserve) / slot), "the tighter card binds the fit");
    // Less free than the reserve: no slot beyond the first is authorised, and the result is never
    // zero or negative (a "0 slots" engine is not a more conservative answer, it is a broken one).
    expect(seq_fit_slots(100ull * 1024 * 1024, slot, 100ull * 1024 * 1024, slot, reserve), 1,
           "no free room still leaves slot 0");
    expect(seq_fit_slots(0, slot, 0, slot, reserve), 1, "no VRAM at all still leaves slot 0");
    // Exactly the reserve: nothing fits beyond slot 0, and not one byte less.
    expect(seq_fit_slots(reserve + slot, slot, reserve + slot, slot, reserve), 2,
           "the reserve is headroom, not a hard wall");
  }

  // ---- reconcile the request with what fits and with a usable context ----
  {
    expect(seq_clamp_slots(4, 4, 262144, 8192), 4, "a request that fits is granted in full");
    expect(seq_clamp_slots(8, 3, 262144, 8192), 3, "a request that does not fit is clamped");
    // The context floor is a real limit and not decoration: at a small --cap, asking for more slots
    // than the cache can divide into usable conversations must reduce N rather than hand out slots
    // with a 200-token context each.
    expect(seq_clamp_slots(8, 8, 32768, 8192), 4, "the context floor caps the slot count");
    expect(seq_clamp_slots(2, 8, 8192, 8192), 1, "one slot at the context floor is still allowed");
    // Degenerate inputs resolve to 1 rather than to 0 or a negative count.
    expect(seq_clamp_slots(0, 4, 262144, 8192), 1, "a request for 0 slots means 1");
    expect(seq_clamp_slots(-3, 4, 262144, 8192), 1, "a negative request means 1");
    expect(seq_clamp_slots(4, 1, 262144, 8192), 1, "a fit of 1 cannot be exceeded");
  }

  // ---- how the cache is divided ----
  {
    // One slot owns everything - the shipped default, and the case that has to stay untouched.
    expect(seq_slot_ctx(262144, 1), 262144, "one slot owns the whole cache");
    expect(seq_slot_ctx(262144, 2), 131072, "two slots split the cache");
    expect(seq_slot_ctx(262144, 4), 65536, "four slots split the cache");
    // Rounding DOWN, and the reason is a bounds check rather than tidiness: a slot's last row must
    // be INSIDE the allocation. Rounding up would put the final slot's last row one past the end -
    // an out-of-bounds write in the KV append, not a slightly wrong answer.
    expect(seq_slot_ctx(1000, 3), 333, "an uneven split rounds down");
    expect(seq_slot_ctx(1000, 3) * 3 <= 1000 ? 1 : 0, 1, "every slot's rows fit in the cache");
    expect(seq_slot_ctx(1, 8), 1, "a degenerate cache still yields a usable slot");
    expect(seq_slot_ctx(0, 4), 1, "a zero cache still yields a usable slot");
  }

  // ---- which slot a request lands on ----
  {
    // The default: one sequence, slot 0, and the round-robin cursor does not move. This is what
    // makes the flag-off path the same code path.
    int rr = 0;
    expect(seq_pick(-1, 1, rr), 0, "one sequence is always slot 0");
    expect(seq_pick(0, 1, rr), 0, "pinning 0 on one sequence is the same request");
    expect(rr, 0, "one sequence does not advance the round-robin cursor");

    // Round-robin over N: consecutive unpinned requests must SPREAD, or every one of them lands on
    // slot 0 and a conversation being kept alive there is the only thing that ever runs.
    rr = 0;
    expect(seq_pick(-1, 3, rr), 0, "round-robin starts at slot 0");
    expect(seq_pick(-1, 3, rr), 1, "round-robin advances");
    expect(seq_pick(-1, 3, rr), 2, "round-robin wraps past the last slot");
    expect(seq_pick(-1, 3, rr), 0, "round-robin wraps around");
    // A cursor left pointing anywhere (here: mid-ring) still produces an in-range slot, and the
    // NEXT call is the one after it - the cursor is a function of where it is, not of history.
    rr = 2;
    expect(seq_pick(-1, 3, rr), 2, "a mid-ring cursor starts where it points");
    expect(seq_pick(-1, 3, rr), 0, "and advances from there");
    rr = 1;
    expect(seq_pick(-1, 4, rr), 1, "a cursor above the ring start is taken modulo the ring");
  }

  // A PINNED slot is a conversation: it must come back as the same slot every time, or a follow-up
  // turn is answered from whatever thread last ran.
  {
    int rr = 0;
    for (int i = 0; i < 5; i++) expect(seq_pick(2, 3, rr), 2, "a pinned slot is stable");
    expect(rr, 0, "a pinned request does not disturb the round-robin cursor");
  }

  // An out-of-range pin is REFUSED (-1), never folded into range. Folding is the one behaviour here
  // that would produce a confident answer to the wrong question: slot 2 occupied by a different
  // conversation would answer with its state and look completely healthy.
  {
    int rr = 0;
    expect(seq_pick(3, 3, rr), -1, "a pin at the ring size is refused");
    expect(seq_pick(99, 3, rr), -1, "a far out-of-range pin is refused");
    expect(seq_pick(-5, 3, rr), 0, "a negative pin means round-robin, not a refusal");
    // A refused pin leaves the cursor alone: it never ran, so it must not perturb the schedule of
    // the requests around it.
    expect(rr, 1, "a refused pin does not advance the cursor");
  }

  // ---- the cost model ----
  //
  // The one thing that must NOT be in the per-slot cost is the KV cache: N slots partition one
  // allocation rather than making N of them. A SeqCost that counted KV would report a per-slot
  // figure of several GB and make every N look unaffordable on a 24 GB card.
  {
    SeqCost c;
    c.add(0, 100);
    c.add(1, 200);
    c.add(0, 50);
    expect_size(c.per_card[0], 150, "cost accumulates per card");
    expect_size(c.per_card[1], 200, "cost keeps the cards separate");
    expect_size(c.total(), 350, "cost totals across cards");
    // An out-of-range card index is ignored rather than writing past the array. A negative one is
    // reachable from a config with no second card, and must not be an out-of-bounds write.
    SeqCost d;
    d.add(-1, 999);
    d.add(2, 999);
    expect_size(d.total(), 0, "an invalid card index contributes nothing");
  }

  // A pin is validated even on a ONE-slot engine. Returning 0 for `n_slots <= 1` before looking at
  // the pin would accept slot 5 on a single-sequence server and answer it as slot 0 - a fold that
  // is invisible precisely because slot 0 is a legitimate answer to give. Found by running a
  // 3-slot plan against a 1-slot engine, where every request silently became slot 0.
  {
    int rr = 0;
    expect(seq_pick(5, 1, rr), -1, "a pin is refused on a single-sequence engine");
    expect(seq_pick(1, 1, rr), -1, "any pin above 0 is refused on a single-sequence engine");
    expect(seq_pick(0, 1, rr), 0, "a pin of 0 is valid on a single-sequence engine");
    expect(rr, 0, "a refused pin leaves the cursor alone");
  }

  if (failures) { printf("test_seq: %d FAILURES\n", failures); return 1; }
  printf("test_seq: all cases PASS\n");
  return 0;
}

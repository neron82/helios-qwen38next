// Prefix-cache resume policy. The rules under test are the ones a wrong answer hides: resume at the
// NEWEST capture at or below the divergence point, never past it, never so far that the forward has
// no row left to run - and drop every capture the new request cannot vouch for.
#include "engine/prefix.hpp"
#include <cstdio>
#include <vector>

using namespace helios;

static int failures = 0;

static void expect(int got, int want, const char* what) {
  if (got == want) return;
  printf("  FAIL %-56s got %d want %d\n", what, got, want);
  failures++;
}

static void expect_at(const std::vector<int>& pos, int want, const char* what) {
  for (size_t i = 0; i < pos.size(); i++) {
    if (pos[i] == want) return;
  }
  printf("  FAIL %-56s no slot holds %d\n", what, want);
  failures++;
}

int main() {
  // A ring of captures at 1024-token boundaries, as prefill() takes them.
  const std::vector<int> ring = {1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192};

  // Nothing resident, or nothing captured: recompute the whole prompt - and keep nothing, because a
  // capture of a prefix this request shares nothing with is worse than no capture at all.
  {
    std::vector<int> p = ring;
    expect(prefix_plan(0, 5000, p), -1, "no common prefix restarts");
    for (size_t i = 0; i < p.size(); i++) expect(p[i], -1, "a restart invalidates the whole ring");
  }
  {
    std::vector<int> p;
    expect(prefix_plan(3000, 5000, p), -1, "empty ring restarts");
  }
  // A prompt shorter than one capture interval can never be resumed: there is no capture below it.
  {
    std::vector<int> p = ring;
    expect(prefix_plan(500, 500, p), -1, "sub-interval prompt restarts");
  }

  // The normal case: the newest capture at or below the divergence point, and it is the slot that
  // holds it that comes back.
  {
    std::vector<int> p = ring;
    expect(prefix_plan(4801, 4801, p), 3, "resume at the newest capture below the match");
    expect_at(p, 4096, "the chosen slot holds 4096");
    expect(p[0], 1024, "captures below the resume point are kept");
    expect(p[3], 4096, "the chosen capture is kept");
    expect(p[4], -1, "captures above the resume point are dropped");
    expect(p[7], -1, "the newest capture is dropped when the prompt diverged before it");
  }
  {
    std::vector<int> p = ring;
    expect(prefix_plan(4096, 9000, p), 3, "an exact capture boundary is usable");
    expect(p[2], 3072, "captures below an exactly-matching boundary survive");
    expect(p[4], -1, "captures past the divergence point go even at an exact boundary");
  }
  {
    std::vector<int> p = ring;
    expect(prefix_plan(4097, 9000, p), 3, "one past a boundary takes that boundary");
  }

  // Divergence INSIDE a capture interval: the rows between the capture and the match are recomputed.
  {
    std::vector<int> p = ring;
    expect(prefix_plan(1500, 9000, p), 0, "divergence inside an interval rewinds one");
    expect(p[1], -1, "a diverged interval's other captures are dropped");
  }

  // Empty slots are skipped, and a fully diverged request leaves the ring empty.
  {
    std::vector<int> p = {-1, 2048, -1, 1024, -1};
    expect(prefix_plan(3000, 9000, p), 1, "empty slots are skipped");
    expect(prefix_plan(1500, 9000, p), 3, "empty slots do not hide a lower capture");
    std::vector<int> q = {-1, 2048, -1, 1024, -1};
    expect(prefix_plan(500, 9000, q), -1, "a prompt below every capture restarts");
  }

  // The whole prompt already resident: the LAST token is still re-run, so the forward has a row.
  {
    std::vector<int> p = ring;
    expect(prefix_plan(4096, 4096, p), 2, "a fully resident prompt still runs its last chunk");
    std::vector<int> q = ring;
    expect(prefix_plan(1, 1, q), -1, "a one-token prompt always runs");
    std::vector<int> r = ring;
    expect(prefix_plan(0, 0, r), -1, "an empty prompt resumes nowhere");
  }

  // Two slots at the same position: both describe the same tokens, so either is correct, and the
  // plan must not depend on which one the ring happens to list first.
  {
    std::vector<int> p = {2048, 1024, 2048};
    expect(prefix_plan(3000, 3000, p), 0, "a duplicated position resolves to a slot holding it");
    expect_at(p, 1024, "the lower capture is dropped when a higher one is chosen");
  }

  if (failures) { printf("test_prefix: %d FAILURES\n", failures); return 1; }
  printf("test_prefix: all cases PASS\n");
  return 0;
}

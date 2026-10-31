#pragma once
// The step protocol (MOONSHOT-1.md section 2.2): the host and the step
// kernel share two pinned rings of two slots. The host writes a descriptor
// and publishes its state; the kernel reads the slot whose turn it is,
// runs the step and publishes the verdict with a system-scope release.
// Ownership follows the states: a descriptor slot is the host's until it
// publishes execute, the kernel's until the verdict is published, then the
// host's again. Tokens stay on the device: a step reads the feed the
// previous step wrote and writes the feed the next step reads, so a step
// whose descriptor brings no admissions, retirements or masks needs no
// host work before it. Host-published sequences let a phase wait for host
// work the step's own tokens started, the gathers of section 2.11.
//
// Steps are numbered by a 64-bit sequence that never wraps in practice and
// alternates the slots by parity; the epoch is the host's 64-bit generation
// the fold's chunk hash will carry (section 2.8).
#include <cstdint>

#if defined(__CUDACC__)
#define DGPP_STEP_HD __host__ __device__
#else
#define DGPP_STEP_HD
#endif

namespace dgpp::step {

enum StepState : uint32_t {
  kStepPending = 0,  // the host is writing it, or has not reached this step
  kStepExecute = 1,  // the kernel runs the described step
  kStepSkip = 2,     // every block exits now with the skip status
};

enum StepVerdictStatus : uint32_t {
  kVerdictOk = 0,
  kVerdictSkipped = 1,
  kVerdictDeadline = 2,  // the phase in bits 8 and up
  kVerdictPoisoned = 3,
  kVerdictBadUnit = 4,
};

struct alignas(128) StepDescriptor {
  uint32_t state = kStepPending;
  uint32_t rows = 1;           // the step's rows
  uint32_t top_k = 1;          // picks a row (the synthetic router until the consumers land)
  uint32_t seed = 1;
  uint64_t seq = 0;            // the step's number; slot = seq & 1
  uint64_t epoch = 0;          // the host's generation, echoed in the verdict
  uint64_t gather_seq = 0;     // a gather phase waits until the host's sequence reaches this (0: no wait)
  uint32_t tokens_from_host = 0;  // 1: the step's tokens are the descriptor's, not the device feed's
  uint32_t pad = 0;
  int32_t tokens[32];          // tokens_from_host: the rows' tokens
};

struct alignas(128) StepVerdict {
  uint32_t status = kVerdictOk;
  uint32_t rows = 0;
  uint64_t seq = 0;
  uint64_t epoch = 0;
  uint64_t tokens_fold = 0;    // a fold of the tokens the step wrote to the feed
  uint64_t begin_ns = 0;       // the kernel's globaltimer at the step's start and end
  uint64_t end_ns = 0;
  uint64_t published_ns = 0;   // when the kernel saw the descriptor's execute
};

constexpr int kStepRingSlots = 2;
constexpr int kStepMaxRows = 32;

// The next token of a row, until the consumers decide it: a hash of the
// token and the step, the same on the host and the device.
DGPP_STEP_HD inline uint64_t step_fold_mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h;
}
DGPP_STEP_HD inline int32_t step_next_token(int32_t token, uint64_t seq, int row) {
  uint64_t h = step_fold_mix(static_cast<uint64_t>(static_cast<uint32_t>(token)), seq);
  h = step_fold_mix(h, static_cast<uint64_t>(row));
  return static_cast<int32_t>(h & 0x7fffffffu);
}

}  // namespace dgpp::step

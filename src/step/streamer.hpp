#pragma once
// The streamer's host side (MOONSHOT-1.md section 2, phase 2): uploads a
// lowered plan, owns the control block and the protocol rings, launches the
// persistent kernel and drives it through the descriptor ring step by step,
// reading each verdict back. The streamer reads `engine.step_sms` as its
// worker count, `engine.l2_staging_mib` as the lead's cap and
// `engine.stream_governor` as the leaders bound; the step engine adapter
// passes them in from the template.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/step/stream.hpp"
#include "step/device_plan.hpp"
#include "step/protocol.hpp"

namespace dgpp::step {

struct StreamerConfig {
  int workers = 47;              // engine.step_sms minus the chain's SM
  int steps = 3;                 // descriptors the run publishes; the phase trace is the last step's
  int lookahead_layers = 2;      // how far ahead the lead may run
  uint64_t lead_cap_bytes = uint64_t{8} << 20;  // engine.l2_staging_mib
  unsigned max_leaders = 0;      // engine.stream_governor: 0 unbounded
  double deadline_s = 5.0;       // every in-step spin's deadline
  double pending_deadline_s = 30.0;  // the kernel's wait for the next descriptor
  int rows = 1;                  // the step's rows and the synthetic router's picks a row
  int top_k = 1;
  unsigned seed = 1;
  uint64_t first_seq = 0;        // the first step's sequence number; steps count up from it
  uint64_t epoch = 0;            // echoed by every verdict
  int skip_at_step = -1;         // publish skip instead of this step (-1: never)
  double gather_delay_us = 0;    // > 0: the host publishes the gather sequence this long after each descriptor
  std::vector<int32_t> tokens;   // the first step's tokens (empty: zeros); later steps read the device feed
  std::function<void(int step)> between_steps;  // host work done while the kernel waits for step `step`
};

struct PhaseTiming {
  int layer = 0;
  int phase = 0;        // index into DevicePlan::phases
  std::string name;
  bool stream = false;
  double begin_us = 0;  // from the step's first phase begin
  double end_us = 0;
  unsigned units = 0;
  unsigned pf_units = 0;
  double drained_us = 0;  // the first worker's idle wait from the phase's drain to its end (0: none seen)
};

struct StreamReport {
  unsigned status = 0;       // the control block's status word
  std::string status_text;
  float kernel_ms = 0;       // the whole launch
  unsigned steps_run = 0;    // steps the kernel completed
  std::vector<StepVerdict> verdicts;  // one per published step, in order
  bool folds_match = true;   // every verdict's token fold equals the host's replica of the feed
  std::vector<double> pending_us;  // step k: from verdict k-1's end to the kernel seeing execute k
  std::vector<double> bind_us;     // step k: from seeing execute to the first phase
  std::vector<double> step_us;     // step k: first phase begin to the verdict
  std::vector<PhaseTiming> phases;  // the last completed step
  std::vector<double> layer_us;     // the last completed step, each layer's first begin to last end
  std::vector<double> layer_bytes;  // what the layer streamed at its run list
  std::vector<int> run_counts;
  double lead_bytes = 0;            // the last step's lead reads
};

class Streamer {
 public:
  // `bases`: the device span base of every layer of `dp`, `plans`: the
  // plans the lowering came from (phase names for the report).
  Streamer(const DevicePlan& dp, const std::vector<const uint8_t*>& bases, const std::vector<LayerPlan>& plans);
  ~Streamer();
  Streamer(const Streamer&) = delete;
  Streamer& operator=(const Streamer&) = delete;

  // Launches the kernel for `cfg.steps` steps on the streamer's own stream,
  // publishes each descriptor as the previous verdict arrives, and reports.
  // Throws on a launch failure; a kernel that ended on a deadline, a poison
  // or a skip reports it.
  StreamReport run(const StreamerConfig& cfg);
  // Ends a running kernel from the host (the engine watchdog's poison).
  void poison();
  StepCtl* ctl() const { return ctl_; }
  // The consumers' operands when the plan has consumer pools (null otherwise):
  // x [kStepMaxRows][hidden] bf16, act [slots][inter] bf16, the fp32 partials
  // and y [kStepMaxRows][hidden] bf16; the shape they were sized for.
  const uint16_t* x() const { return x_; }
  const uint16_t* act() const { return act_; }
  const float* out() const { return out_; }
  const uint16_t* y() const { return y_; }
  int hidden() const { return hidden_; }
  int inter() const { return inter_; }
  bool consumers() const { return consumers_; }

 private:
  DevicePlan dp_;
  std::vector<std::string> phase_names_;
  const uint8_t** bases_ = nullptr;
  DevLayer* layers_ = nullptr;
  DevPhase* phases_ = nullptr;
  DevSegment* segments_ = nullptr;
  DevPool* pools_ = nullptr;
  unsigned* sink_ = nullptr;
  StepCtl* ctl_ = nullptr;
  StepDescriptor* desc_ = nullptr;      // pinned, host view
  StepDescriptor* desc_dev_ = nullptr;  // the device's view of the same memory
  StepVerdict* verdict_ = nullptr;
  StepVerdict* verdict_dev_ = nullptr;
  uint64_t* host_seq_ = nullptr;
  uint64_t* host_seq_dev_ = nullptr;
  int32_t* token_feed_ = nullptr;
  cudaStream_t stream_ = nullptr;
  bool consumers_ = false;
  int hidden_ = 0, inter_ = 0;
  uint16_t* x_ = nullptr;
  uint16_t* act_ = nullptr;
  float* out_ = nullptr;
  uint16_t* y_ = nullptr;
  unsigned consumer_smem_ = 0;
};

}  // namespace dgpp::step

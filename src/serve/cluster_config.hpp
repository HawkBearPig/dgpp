// Resolved runtime configuration shared by the launcher and server.
//
// The launcher combines a deployment JSON (model, world_size, engine options)
// with site settings from .env. This loader reads the resulting JSON, with
// explicit nodes, ports and paths. nodes[0] is rank 0, which serves HTTP and sends
// effective model/engine settings to peers through the admission journal.
// Peers read bootstrap addresses and local paths from their own files.
//
// Flags after --config override the file. Unknown keys and invalid types
// are errors. Engine defaults match the binary's defaults; deployment
// templates set serving values explicitly. Each rank hashes its effective
// shared configuration, and the warm record checks those hashes before
// serving. See docs/configuration.md for the schema and deploy/*.example.json for
// templates. Multi-node deployment templates must be resolved before this loader
// reads them. A single-node template may use world_size: 1 directly (localhost);
// nodes and world_size are mutually exclusive.
#pragma once
#include "serve/file_inputs.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "kernels/rope_scaling.hpp"
#include "serve/http_limits.hpp"
#include "sched/prefill_latency.hpp"

namespace dgpp::serve {

struct ClusterConfig {
  std::string model;
  std::vector<std::string> nodes;  // rank = index; nodes[0] is the head
  std::string ssh_user;            // empty: the launcher's own user
  std::string release;             // the installed release the launcher runs (launcher-only; empty: the development binary)
  size_t tokenizer_word_cache_entries = 0;
  int http_port = 18080;
  std::string http_bind = "0.0.0.0";
  int64_t http_max_body_bytes = kDefaultHttpMaxBodyBytes;
  int sse_ping_interval = kDefaultSsePingInterval;
  std::string stream_flush = "pass";
  // Local paths and device names may differ by rank; no credentials belong here.
  std::vector<std::map<std::string, std::string>> node_env;
  int fabric_port = 29970;
  int journal_port = 29971;
  // Each peer rank's metrics listener (rank_metrics.hpp), on its own node
  // address; 0 = off. Rank-local: not part of the config digest.
  int metrics_port = 0;
  struct Engine {
    int max_concurrency = 8;
    int64_t kv_capacity = 8192;
    // The cache format: GLM-5.3's latent cache bf16 | fp8 | fp4; MiMo's
    // K/V bf16 | fp8; Qwen3.5's K/V bf16 | fp8 (the e4m3 storage form at
    // the mixed NVFP4 release's shipped k_scale / v_scale, 2026-10-16; the
    // attention's arithmetic on it is prefill_attention below).
    std::string kv_dtype = "bf16";
    // The full GLM-5.3's embedding table: "replicated" on every rank, or
    // "vocab" — each rank holds its lm-head slice of the rows, gathered
    // per token and summed by one fold (bitwise the same numbers, one
    // small collective per lookup, −1.33 GiB per rank at world 4).
    std::string embed_sharding = "replicated";
    // The full GLM-5.3 draft block's head (2026-10-09): "checkpoint" reads
    // the lm head as the main step does (its 12-bit form under
    // bf16_weights); "fp8" reads a block-FP8 copy encoded at load (e4m3,
    // 128 x 128 fp32 scales — the FP8 releases' recipe; +0.23 GiB per rank
    // at world 4). The verifier keeps the main head, so the served
    // distribution is unchanged; only the draft's proposals (acceptance)
    // can move. Other families ignore it.
    std::string mtp_head = "checkpoint";
    // The DeepSeek-V4.1 indexer's budget (2026-10-18; the deepseek_v41
    // family alone): how many compressed entries a query attends
    // (index_topk; the release trained at 512, the select networks serve
    // up to 2048) out of a candidate pool of candidate_topk_blocks blocks
    // of candidate_block_size entries (2048 x 8 as shipped). 0 keeps the
    // checkpoint's value. Widening the budget changes the model's output
    // (more context visible to every query), not just its speed.
    int index_topk = 0;
    int candidate_topk_blocks = 0;
    int candidate_block_size = 0;
    // The full GLM-5.3 attention projections' resident form (2026-10-09):
    // "checkpoint" as shipped (int8 g64); "int4" re-encoded to int4 g64 at
    // load (RTN) — −2.1 GB per MTP pass per rank, NOT lossless. Other
    // families ignore it.
    std::string attention_weights = "checkpoint";
    // The Qwen n-gram table's residency: "resident" copies it
    // to the device (the default; 47.7 GiB at world 1), "mmap" leaves it
    // on the NVMe behind the page cache and gathers each step's rows on
    // the host — the single-Spark deployment.
    std::string ngram_table = "resident";
    // Where the Qwen n-gram table's shards live when the checkpoint does not
    // carry them (the AutoRound hybrid, docs/qwen38_autoround_int4_plan.md
    // D6): a Hugging Face repository id resolved through the model cache
    // like `model` — the FP8 release (`Qwen/Qwen3.8-Flash-Next-FP8`, whose
    // shards hold the table) or the extracted table repo. Only the table's
    // tensors are taken from it. Empty: the checkpoint's own shards.
    std::string ngram_table_model;
    // The DeepSeek-V4.1 Engram tables' source (2026-10-22, the EXL3 pack
    // ships none): a Hugging Face repository id resolved through the model
    // cache like `model` — the base release whose shards hold the tables
    // (`deepseek-ai/DeepSeek-V4.1-Flash-1017`). Only the table tensors are
    // taken from it; its Engram sidecar serves when the checkpoint has
    // none. Empty: the checkpoint's own shards. Refused for any other family.
    std::string engram_source;
    // The Qwen dense stack's form: "checkpoint" (the default:
    // the BF16 the checkpoint ships) or "fp8" (every dense projection
    // encoded to block FP8 at load — the same recipe as the FP8 releases;
    // docs/qwen38_single_spark.md).
    std::string dense_weights = "checkpoint";
    std::string fp8_head = "gemv";  // Qwen head: gemv | mma (opt-in)
    // The MTP draft head's read (2026-10-13; kernels/head_screen.hpp):
    // "exact" streams the whole head per draft token; "screen" finds the
    // candidate 128-row blocks through a 2-bit rotated copy and computes
    // only those exactly — output-exact, only which token is drafted can
    // differ. draft_head_blocks: the candidate cap, 1..64.
    std::string draft_head = "exact";
    int draft_head_blocks = 32;
    // The MTP draft layer's routed-expert layout: "fp8" (the default:
    // per-expert FP8 tensors as the NVIDIA release ships) or "bf16_fused"
    // (the RadixArk release: all 512 experts stacked as fused BF16
    // gate_up_proj and down_proj, encoded to FP8 at load). Applies to the
    // Qwen family only.
    std::string mtp_expert_format = "fp8";
    // The resident form of the bf16 weights the decode GEMV reads (the
    // attention / linear-attention projections, the lm head):
    // "checkpoint" (the default: the bf16 bytes alone), "bf12" — a lossless
    // 12-bit form of each matrix ALONE (kernels/bf12_gemv.hpp: the
    // sign+mantissa byte and a 4-bit exponent code, bitwise the bf16 GEMV,
    // 0.75 of the bytes a decode step streams and 0.75 of the memory; a
    // prefill GEMM expands the rows it reads into a small scratch) — or
    // "bf12+bf16": both forms resident (the same decode, prefill reads the
    // bf16 bytes in place, +0.75 of those matrices' memory). The memory
    // plan carries either (common/bf16_residency.hpp).
    std::string bf16_weights = "checkpoint";
    // The opt-in draft vocabulary slice (2026-09-29, the Qwen3.8 hybrid;
    // docs/qwen38_autoround_int4_plan.md §6.7): a .npy of token ids (int32
    // or int64, one dimension) — the MTP draft head scores only those rows
    // of the lm head (every other id is -inf for the draft), so a draft
    // step reads that slice instead of the whole head. The target verifies
    // every draft, so outputs are unchanged; only the draft acceptance can
    // move (a token outside the set is never proposed). Empty (the
    // default): the draft scores the whole vocabulary. Never counted toward
    // a headline number; tools/build_draft_vocab.py builds a set.
    std::string draft_vocab;
    // The opt-in prefill speed levers (2026-09-30, the Qwen3.8 hybrid;
    // docs/qwen38_autoround_int4_plan.md §6.15). Each changes the
    // prefill's arithmetic within the quantized model's tolerance — the
    // served transcripts can differ from the default chain's — so each is
    // off unless the deployment's config turns it on (the AutoRound
    // template turns on the first and the third, measured within the
    // default chain's eval band; the benchmarks list both rows).
    //   prefill_bf16_partials: the packed expert chain's down projection
    //     written in bf16 and its per-expert partials summed from bf16
    //     (half the bytes a chunk writes and reads back; the reference
    //     stack's form). Default: fp32 partials in the ordered chain.
    //   prefill_fold_scales: the wide packed expert GEMM with each
    //     group's scale folded into the bf16 weight values and one fp32
    //     accumulator across K (Marlin's form; no per-group fma).
    //   prefill_fp8_gemm: the block-FP8 dense stack's prefill GEMMs on
    //     the fp8 tensor cores (per-token 1 x 128 e4m3 activations, the
    //     checkpoint's 128 x 128 weight scales, fp32 promotion per group —
    //     the reference stack's cutlass blockwise GEMM) instead of the
    //     dequantized bf16 GEMM. Requires engine.dense_weights = fp8.
    //   prefill_fp8_per_tensor: the Qwen3.8-27B prefill recipe — every FP8
    //     projection's prefill GEMM (rows above the decode GEMV band, and
    //     every resumed chunk) on cuBLASLt's per-tensor-scale e4m3 kernels
    //     from boot-requantized per-tensor weights and per-call per-tensor
    //     activations: ~2x the dequantized bf16 GEMM's rate, +23 GiB
    //     resident at the 27B's shape, and not transcript-preserving at
    //     long context. Default: the dequantized bf16 GEMM (exact).
    //   prefill_activations: "bf16" (the default: activations stay bf16 and
    //     quantized weights are decoded to bf16 for the products) or
    //     "native" (the checkpoint's own activation recipe on every
    //     prefill row — 2026-10-15: the Qwen3.8-27B NVFP4 release's MLP
    //     inputs quantized to NVFP4 per row with its shipped input globals,
    //     the products on the block-scaled fp4 tensor cores, and its
    //     channel-FP8 projections and late MLPs from per-token e4m3 rows on
    //     the e4m3 tensor cores; the FP8 release's blockwise product; decode
    //     rows keep bf16 activations).
    //   prefill_attention: "bf16" (the default: the attention products in
    //     bf16 — on an fp8 K/V pool the codes widened exactly) or "fp8"
    //     (2026-10-16, the Qwen3.5 family on engine.kv_dtype fp8: the
    //     prefill attention on the fp8 tensor cores — Q and the
    //     probabilities quantized to e4m3 per row, S and O on
    //     mma.e4m3 at twice the bf16 rate; decode rows keep the exact
    //     widened walk). Also supported for Flash-Next QSA, with its
    //     per-token/head cache scales. Both require kv_dtype fp8.
    bool prefill_bf16_partials = false;
    bool prefill_fold_scales = false;
    bool prefill_fp8_gemm = false;
    bool prefill_fp8_per_tensor = false;
    std::string prefill_activations = "bf16";
    std::string prefill_attention = "bf16";
    // Flash-Next QSA decode arithmetic over its fp8 pool. Independent of
    // prefill_attention so either phase can retain bf16 compute.
    std::string decode_attention = "bf16";
    // The packed expert GEMM's form and companions (2026-09-30; every
    // serving switch is a config key — no environment variable selects a
    // kernel). All bitwise the default chain.
    //   expert_gemm: "wide" (the default: the 64 x 128 tensor-core tile,
    //     eight warps, the codes decoded once per step), "wide3" (register
    //     decode, three stages), "wide4" / "wide4r" (the four-warp forms),
    //     "narrow" (the 32 x 64 kernel).
    //   expert_gemm_prefetch: the L2 prefetch distance in k-steps (0 off).
    //   expert_tile_list: the compact tile list over the routed segments
    //     (off: the grid over the longest segment).
    //   expert_gemm_pair: gate and up as one launch (measured level).
    //   ngram_prestage (Qwen): the next chunk's n-gram rows gathered while
    //     this chunk runs (off: the one-channel staging).
    std::string expert_gemm = "wide";
    int expert_gemm_prefetch = 3;
    bool expert_tile_list = true;
    bool expert_gemm_pair = false;
    bool ngram_prestage = true;
    // The DeepSeek-V4.1 prefill mode (docs/deepseek_v41_flash_plan.md
    // §1.8): "bounded" (the default: the encoder over every prompt row,
    // the decoder over the last window rows — the model's own serving
    // recipe, half the prefill work) or "exact" (every layer over every
    // row, the parity mode).
    std::string prefill = "bounded";
    // The Dory1.0-4B loop (docs/dory_plan.md D1, D2). Applies to the Dory
    // family only; every other family refuses them (serve, by name).
    //   dory_loops: the recurrent block's loop count. Absent (0) is the
    //     exact path: the checkpoint's own `n_recurrent_loops` (2 in the
    //     release). Naming one changes what the engine computes.
    //   dory_kv_cache: "per_loop" (the default, exact: every loop of the
    //     recurrent block gets its own K/V planes) or "last_loop" (one
    //     plane per layer, only the last loop's rows kept — the pool at
    //     half the planes, the loops past the last one's history gone).
    int dory_loops = 0;
    std::string dory_kv_cache = "per_loop";
    std::string prefill_dense = "auto";  // family default | tile | streaming
    std::string prefill_window = "flash";  // the DeepSeek-V4.1 prefill's window attention: flash | scalar
    std::string decode_window = "flash";   // the DeepSeek-V4.1 decode's window attention: flash | scalar
    // The fabric's latency-class collective transport for the decode
    // step's folds and the pick (2026-10-19): "write" = rows RDMA-written
    // into the peers' write rings as self-describing chunks, folded by a
    // grid of warps as they prove; "send" = payload + doorbell SEND pairs
    // claimed through door cells by one block, hash-gated, acked, credited.
    std::string collective_transport = "write";
    bool prefill_fold_overlap = true;
    // The decode graph's kernel edges programmatic (2026-10-20): a kernel's
    // blocks scheduled as its predecessor drains, each waiting for the
    // predecessor's completion before its first read (engine/graph_pdl.hpp).
    bool decode_pdl = true;    // the DeepSeek-V4.1 prefill's boundary folds under the other row block's compute
    // The opt-in YaRN rope ramp (2026-09-18, the Qwen3.8-Flash-Next 512K
    // recipe): absent is the default and leaves every table and every
    // kernel argument exactly as they were; present builds the Qwen QSA
    // rope from the YaRN inverse frequencies, scales its cos/sin by the
    // vLLM attention factor (yarn_get_mscale(factor) * attn_factor) and
    // lifts the context cap from max_position_embeddings to
    // original_max_position_embeddings x factor. The checkpoint's own
    // config never carries it — the NVFP4 release declares no scaling and
    // the A/B target applies YaRN from the launcher's --hf-overrides.
    std::optional<dgpp::RopeScaling> rope_scaling;
    int default_max_tokens = 256;
    FileInputConfig file_inputs;
    int queue_limit = 64;
    // Rank 0 holds the first arrival at an idle engine this long for the
    // rest of its burst (requests sent together land a few ms apart), so
    // they are read in together. 0: tick at once.
    int admission_gather_ms = 3;
    int max_connections = 64;
    bool no_eos = false;
    bool compact_batches = true;
    bool decode_graph = false;
    // The decode engine (MOONSHOT-1.md §1.1): "graph", the engines that
    // serve today, or "step", the persistent-step engine, which selects
    // its own weight layout with it. A template that does not set it
    // cannot observe that the step engine exists.
    std::string decode_engine = "graph";
    // The step engine's streamer (docs/step_engine.md): the SMs its workers
    // take (0: every SM but the chain's), the lead's L2 cap in MiB, and the
    // governor as the workers that stream the lead at once (0: unbounded).
    int step_sms = 0;
    int l2_staging_mib = 8;
    int stream_governor = 0;
    bool mtp = false;
    int mtp_depth = 1;             // draft tokens per step (1..5); needs mtp
    bool mtp_depth_set = false;    // the file named it (else a family may default it: DSpark's block is 5)
    // The DFlash2 block drafter (Qwen3.5-family, eager path only): a
    // checkpoint directory or HF id whose config.json names the drafter.
    // Replaces mtp (mutually exclusive); the draft width is the
    // checkpoint's block_size - 1.
    std::string dflash_model = "";
    // The drafter's block matrices as served: "checkpoint" (bf16, packed
    // lossless 12-bit under bf16_weights bf12) or "fp8" (block-128 E4M3 —
    // lossy for the proposals only; the target's verify is exact).
    std::string dflash_weights = "checkpoint";
    // Its serving options: the multi-slot verify replayed as a captured
    // graph (the measured best), the redrafts stacked across slots, and a
    // verify-depth cap (0 = the whole block; transcripts are exact at any
    // value — unverified drafts re-draft next step).
    bool dflash_verify_graph = true;
    // Several cold prompts prefilled as the spans of one walk (the graph
    // worlds' admission). false: one prompt per walk — a prompt's prefill
    // then never depends on who arrived with it (the walk's row count
    // selects the GEMM lowering above 128 rows; a prompt shorter than that
    // started together with others can otherwise read differently from the
    // same prompt alone), at the cost of a burst's prefill throughput.
    bool prefill_group = true;
    // The L2 weight prefetcher (kernels/l2_prefetch.hpp; environment
    // variables until 2026-10-05): on/off, the form (load the bytes; one
    // L2 prefetch per 128-byte line; or touch one line per 64 KB for the
    // page walks alone), the window budget
    // in MiB, the boundary windows' budget (0: the window budget), the
    // rates (off | light | full) of the windows beside a collective and
    // inside the attention layers, and whether adjacent ranges merge.
    bool l2_prefetch = true;
    std::string l2_prefetch_form = "load";
    std::string l2_prefetch_full_form = "16x8";
    int l2_prefetch_window_mib = 12;
    int l2_prefetch_boundary_window_mib = 20;
    std::string l2_prefetch_boundary_rate = "light";
    std::string l2_prefetch_layer_rate = "light";
    bool l2_prefetch_merge = true;
    bool dflash_draft_batch = true;
    int dflash_depth = 0;
    // The confidence-scheduled verify depth (engine/verify_schedule.hpp,
    // 2026-09-14; needs mtp and a family with a confidence head — DSpark):
    // a step verifies only the leading drafts whose prefix survival beats
    // the value of a verify row. The cost curve and the value of time are
    // the world's (every rank takes rank 0's): `row_ms` one verify row,
    // `base_ms` the step's fixed cost (the draft included), `lambda` the
    // value of decode time in tokens/ms (0: the reservation rate
    // 1 / (base + row)); `min_depth` the fewest drafts a step verifies.
    bool mtp_schedule = false;
    double mtp_schedule_row_ms = 8.0;
    double mtp_schedule_base_ms = 28.0;
    double mtp_schedule_lambda = 0.0;
    int mtp_schedule_min_depth = 1;
    int mtp_schedule_options = 0;  // at most this many scheduled depths (0: as many as the graph-variant budget holds)
    bool mtp_schedule_adapt = true;  // lambda follows the modeled throughput, floored at mtp_schedule_lambda
    // A sampled request's acceptance per position as a fraction of the
    // confidence head's, when its drafts are the draft's argmax
    // (mtp_draft greedy); 0: sampled requests verify the whole block.
    double mtp_schedule_sampled_scale = 0.93;
    // A sampled request's drafts: "sampled" (draws from the draft's own
    // distribution, the ratio verify), "greedy" (the draft's argmax, accepted
    // with probability P(draft)) or "auto" (the family's measured better
    // rule: greedy for a DSpark block, sampled elsewhere). Exact either way.
    std::string mtp_draft = "auto";
    // The drawn drafts' temperature as a fraction of the request's (the
    // proposal is the draft's distribution at THIS temperature; any value is
    // exact — it only moves the overlap with the target, i.e. the acceptance
    // rate: a sharper draft keeps the argmax's rate on sharp distributions,
    // the request's keeps the overlap on flat ones). 1: the request's.
    double mtp_draft_temperature = 1.0;
    // Context-lookup drafting (engine/lookup_draft.hpp): fuse a strong
    // prompt-history match into the MTP drafts on greedy slots. Same
    // verify rows — throughput only, never the output. Off by default;
    // every rank must agree (journal + digest carry it).
    bool lookup_draft = false;
    int lookup_nmin = 6;     // minimum suffix-match length to propose
    int lookup_nstrong = 8;  // a match this long is taken on its own
    int lookup_agree = 2;    // shorter matches need this many leading MTP agreements (v1: unused)
    // Extra verify rows past the MTP block, filled from the prompt history
    // (a pinned serving mode: fewer batch slots for a longer block while
    // copying). 0: the MTP block alone.
    int lookup_tail = 0;
    // The sampled chain's verify rule: "block" (block verification — the
    // drafts decided jointly; exact, never fewer tokens in expectation
    // than the token rule) or "token" (the token-by-token test).
    std::string mtp_verify = "token";
    // The block drafter's batched verify rows budget: a batch family whose
    // slots times the block exceed it verifies the first drafts of every
    // slot's block (exact; the drafter proposes the whole block either
    // way). 0: every family verifies the whole block.
    int dflash_batch_rows = 0;
    int graph_batch_min_live = 0;  // 0 = min(2, max_concurrency) (the batch family, 2026-09-07)
    int sampling_candidates = 128;
    // Omitted fields inherit the checkpoint/neutral defaults. Process flags
    // override these defaults; each request can override them again.
    std::optional<float> repetition_penalty;
    std::optional<float> frequency_penalty;
    std::optional<float> presence_penalty;
    double prefix_cache_gib = 1.5;
    int prefix_min_tokens = 1024;       // no prefix snapshot below this position (0: every cut)
    bool prefix_head_snapshots = true;  // a cold prompt also keeps its first structural cut (the system prompt's end)
    // prompt_cache_key pins protect their entries while the pinned entries
    // hold at most this share of the arena's slots and the NVMe slab's
    // pages; past it a pin protects nothing (100: always).
    int prefix_pin_max_pct = 50;
    std::string admission = "full";
    int admission_window = 256;
    int prefill_budget_tokens = -1;  // automatic: 256 tokens, adjusted to engine geometry
    std::string model_alias = "";  // the served model name: listed, reported and required of requests (empty: the model id)
    int prefill_idle_budget_tokens = 0;  // auto busy: up to 4096 for grouped prefill; explicit busy: same budget
    int decode_gap_target_ms = sched::kDefaultDecodeGapTargetMs;
    int first_token_target_ms = sched::kDefaultFirstTokenTargetMs;
    sched::PrefillLatencyTuning prefill_latency;
    // Omitted: resolve from the constructed engine's mixed-execution support.
    // Settings retain auto/on/off; the warm journal carries the resolved bool.
    std::optional<bool> prefill_early_admission;
    double bulk_pace_gbps = -1.0;  // derived from the port rate
    int bulk_inflight = -1;
    int rendezvous_timeout_ms = 120000;
    double stats_interval_s = 10.0;
    bool reasoning_in_content = false;
  } engine;
  // The NVMe cold tier for the prefix cache (issue #26): a preallocated
  // slab per rank under `path`, sized to `capacity_gib`, that evicted
  // entries spill to and later requests restore from. Top-level, so a
  // deployment turns the tier on with one value; `enabled` requires a
  // positive capacity. Startup checks the path, the free space and the
  // minimum size before anything is allocated. `min_tokens` (0: one
  // prefill chunk) is the shortest entry worth keeping cold.
  struct NvmeCache {
    bool enabled = false;
    std::string path = "~/dgpp/nvme-cache";
    double capacity_gib = 0.0;
    int64_t min_tokens = 0;
    // One staging slice in MiB: two slices of device work move per tick, so
    // the chunk paces spills and restores against the decode step and
    // sizes the staging (4 x chunk + 8 MiB) in the memory plan.
    int64_t chunk_mib = 64;
    // Spills and restores started but not yet committed by the world: a
    // started entry is busy (never evicted) until its commit.
    int64_t max_ops = 4;
    // Free space the pre-flight leaves on the slab's filesystem, in GiB.
    double reserve_gib = 1.0;
    // The slab outlives the process: a start with the same checkpoint,
    // settings and slab geometry adopts its entries (when every rank holds
    // the same index). False recreates the slab at every start.
    bool persist = true;
    // Select an existing binding across build/settings changes. This is
    // the operator's compatibility assertion; structural checks still run.
    std::string reuse_binding;
    // A filesystem that refuses direct I/O (tmpfs) is refused unless this
    // allows buffered I/O: the page cache would grow into the model's memory.
    bool allow_buffered = false;
    // An op without progress for this long fails, so a request waiting on
    // its restore admits cold instead of never (0: never).
    int64_t op_timeout_s = 120;
  } nvme_cache;
  struct Paths {
    std::string log_dir = "~/dgpp/log";
    std::string stage_dir = "/tmp/bus4";     // where the launcher puts the peers' binary and config
    std::string release_dir = "~/dgpp/releases";
    std::string resident_cache;              // empty: the binary's default (~/.cache/dgpp/resident)
  } paths;

  int world() const { return static_cast<int>(nodes.size()); }
};

// Parses the JSON text; `what` names it in errors. Throws
// std::runtime_error naming the offending key.
ClusterConfig parse_cluster_config(const std::string& json, const std::string& what);
// Reads and parses the file.
ClusterConfig load_cluster_config(const std::string& path);

// "~" and "~/..." to $HOME; anything else unchanged.
std::string expand_home(const std::string& path);

// The 64-bit FNV-1a of a canonical configuration string, as 16 hex digits.
std::string config_digest(const std::string& canonical);

}  // namespace dgpp::serve

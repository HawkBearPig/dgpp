#pragma once
// Resident weight loader for the GLM text model (M4 deliverable 2,
// DESIGN §4). Loads one layer at a time from the mapped checkpoint into
// DEVICE memory through a pinned staging mirror (see GlmLayerBump for why
// not managed), transformed into the exact layouts the M2/M3 layer
// kernels consume: merged KDA in_proj ([f_a|g_a|q|k|v|b] rows) and conv
// (q|k|v channels), fused DSA qkv_a, F32 indexer APE. The E4M3+scale pairs
// of the MLP/MoE matrices stay resident in compressed form — "load" never
// means a persistent BF16 expansion of the model. The four DSA attention
// matrices (q_a, kv_a, q_b, o_proj) are dequantized per layer as a
// documented TRANSIENT bridge for the M3 IGemm interface (bf16 weights); the M4
// scale-aware GEMM replaces that bridge.
//
// Streaming model (M4 deliverable 4): exactly one layer is resident at a
// time — a bump allocation reset per load, sized to the largest layer.
// Globals (embed/lm_head/final norm) load once and persist. Peak device
// memory is therefore largest-layer + globals, which is what makes
// full-model correctness testable on a single node without the TP placement.
//
// RESIDENT model (the production residency contract, DESIGN §3, M6 d5 —
// implemented M5 ahead of the serving integration): GlmResidency::Resident
// materializes each layer ONCE into its own exact-formula bump and keeps
// it for the stream's lifetime — storage is never touched again after a
// layer's first load. load_layer on a materialized layer returns the
// cached view with NO sync (the loader wrote nothing; the streaming
// contract's sync exists to cover loader WRITES) and NO storage reads
// (source_bytes_read stops growing — glm_stream_check's second pass is
// the contract's proof). release_layer is a no-op so existing call sites
// (the model's preconstruct pass) stay correct unchanged. resident_bytes()
// is the exact formula total, so callers can refuse a world that cannot
// fit BEFORE the first allocation: only TP=4 fits 128 GB at real dims
// (~79 GiB weights/rank, ~49 GB headroom; TP=2 and TP=1 cannot — the
// memory rationale for the four-rank deployment target).
//
// Sharded build (M5 d4): world>1 produces the resident layer directly at
// this rank's TP geometry — the same layouts GlmTpViews::bind would carve
// from a full layer, pinned bitwise by glm_tp_test's shard-parity test.
// world=1 stays the degenerate rank 0 (the M4 build, byte-for-byte).
//
// Synchronization contract: load_layer/load_globals cudaDeviceSynchronize
// before returning, so the weights are readable by device work issued
// afterwards; the resident pointers are DEVICE addresses (cudaMemcpy them
// to inspect from the host) and must not be overwritten while GPU work runs.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "models/quant_matrix.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/glm/resident_image.hpp"
#include "models/dsa_layer.hpp"
#include "models/glm/binding.hpp"
#include "models/glm/config.hpp"
#include "models/kda_layer.hpp"

namespace dgpp {

// Which residency contract a stream serves (see the file comment).
enum class GlmResidency {
  Streaming,  // one layer resident at a time (the M4 diagnostic instrument)
  Resident,   // every layer materialized once, kept for the lifetime
};

// Which globals contract a stream serves for the lm head (M6 d3, the
// "vocabulary-sharded lm-head interface"): Full keeps the M4/M5 diagnostic
// default — every rank holds every vocab row, logits are the full
// [T, vocab], and every pinned parity gate stays byte-stable.
// VocabSharded gives rank r the contiguous row slice [V*r/W, V*(r+1)/W)
// (integer bounds: contiguous, gap-free, any V), so a rank computes only
// its logits slice [T, count] — the head GEMM's N. Sampling merges the
// slices per DESIGN §10 (glm_sampler); the per-column K-reduction is
// unchanged, and the head-shard parity gate pins slice == slice-of-full
// bitwise. embed/final_norm stay replicated in v1 (a row gather and a
// [hidden] vector: sharding buys nothing, and the embed lookup contract
// keeps the token broadcast trivial).
enum class GlmHeadSharding {
  Full,
  VocabSharded,
};

// The weight bump (loaders/weight_build.hpp): aligned grants, reset per
// layer; counting mode walks the same grant sequence without allocating,
// which is how the byte formula and the allocator share one code path.
using GlmLayerBump = LayerBump;

// One layer's routed experts in compressed form. At world>1 (M5 d4, resliced
// 2026-09-02) `experts` holds EVERY expert's slice of the intermediate dim —
// gate/up rows and down columns [rank*I/world, (rank+1)*I/world) — the same
// slicing the shared expert carries; world=1 holds the full matrices.
struct GlmMoeResident {
  const uint16_t* router_gate = nullptr;  // BF16 [n_routed_experts, hidden]
  const float* router_bias = nullptr;     // F32 [n_routed_experts]
  GlmQuantMatrix shared[3];               // gate, up, down (FP8 under both formats)
  // The routed experts in the layer's format (GlmTextConfig::expert_format):
  // FP8 triples in `experts`, or NVFP4 triples in `experts_fp4` with every
  // matrix's F32 global scale gathered into one [n_experts, 3] array
  // (`expert_global_scales`, device) that the views point into. Exactly one
  // of the two vectors is populated.
  std::vector<GlmQuantMatrix> experts;    // gate, up, down per expert (sliced)
  std::vector<GlmFp4Matrix> experts_fp4;  // gate, up, down per expert (sliced)
  const float* expert_global_scales = nullptr;
  bool nvfp4() const { return !experts_fp4.empty(); }
  const GlmQuantMatrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
  const GlmFp4Matrix& expert_fp4(int e, int i) const {
    return experts_fp4[static_cast<size_t>(e) * 3 + i];
  }
};

// Multi-head hyper-connection coefficients (all replicated across TP).
struct GlmMhcResident {
  const float* attn_base = nullptr;   // F32 [24]
  const uint16_t* attn_fn = nullptr;  // BF16 [24, hc_mult * hidden]
  const float* attn_scale = nullptr;  // F32 [3]
  const float* ffn_base = nullptr;
  const uint16_t* ffn_fn = nullptr;
  const float* ffn_scale = nullptr;
};

struct GlmLayerResident {
  int layer = -1;
  GlmLayerKind kind = GlmLayerKind::Kda;
  GlmMhcResident mhc;  // empty on the MTP draft layer (it has none)
  const uint16_t* ln1 = nullptr;  // BF16 [hidden]
  const uint16_t* ln2 = nullptr;
  // Exactly one attention view is populated (by layer kind).
  KdaLayerWeights kda;  // merged in_proj/conv layouts (M2 kernel contract)
  DsaLayerWeights dsa;  // bf16 layouts (M3 kernel contract; transient bridge)
  // Exactly one MLP view is populated (by layer MLP class).
  GlmQuantMatrix dense[3];  // gate, up, down
  GlmMoeResident moe;
  // MTP draft head (populated only for layer == mtp_layer()).
  const uint16_t* enorm = nullptr;         // BF16 [hidden]
  const uint16_t* hnorm = nullptr;         // BF16 [hidden]
  const uint16_t* eh_proj = nullptr;       // BF16 [hidden, 2 * hidden]
  const uint16_t* shared_head_norm = nullptr;  // BF16 [hidden]
  size_t bytes = 0;  // exact device bytes this layer occupies
};

struct GlmGlobalsResident {
  const uint16_t* embed = nullptr;      // BF16 [vocab, hidden]
  const uint16_t* lm_head = nullptr;    // BF16 [lm_vocab_count, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  // The lm_head's vocab slice (Full: [0, vocab) — count 0 reads as full
  // for pre-M6 call sites; VocabSharded: this rank's contiguous slice).
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// Boot-time digest over replicated weight source bytes (DESIGN §5.2: "boot
// checks hash all replicated tensors"). One order-independent sum-hash per
// layer (MTP last) plus the globals: ranks compare element-wise, so a
// mismatch pinpoints the layer. Each tensor folds as FNV-1a over its name
// then its raw checkpoint bytes, summed per layer — commutative, so load
// order cannot change the value. Not adversarial: this is a
// checksum-class agreement check between ranks loading the same files,
// not a content-authentication hash.
//
// Replicated in v1: mHC, both layer norms, routers, the DSA indexer and
// APE, the DSA latents (q_a/kv_a/their norms), KDA f_a/g_a/o_norm, the
// MTP draft head, and the globals (embed/lm_head/final_norm load full on
// every rank until the M6 vocabulary-sharded lm-head interface). The DSA q_b/
// o_proj dequant bridges are not here — they are read in full by every
// rank (the bf16 interface), but their resident outputs are sharded; they
// count as full-read in the byte reconcile instead.
struct GlmReplicatedDigest {
  std::vector<uint64_t> layer;  // per layer 0..max_layer-1
  uint64_t globals = 0;
  uint64_t bytes = 0;    // total folded source bytes
  uint64_t tensors = 0;  // total folded tensors
};

class GlmLayerStream {
 public:
  // Opens every safetensors shard in `checkpoint_dir` (sorted by name) and
  // validates the full text binding (glm_validate_text_binding); throws on
  // any mismatch. Allocates the layer bump at the max layer size.
  //
  // Sharded load (M5 d4): `world` > 1 builds each resident layer directly
  // at this rank's local geometry — head row ranges and 128-aligned
  // quantized row/column slices of every MLP matrix, routed experts
  // included (the §5.2 scale-grid contract) — so only rank-local
  // checkpoint bytes are ever read. world=1 is the degenerate rank 0: the M4 full-geometry
  // build, byte-for-byte (same grant sequence, same bytes — the layer
  // formula cannot drift). TP geometry is validated BEFORE any shard is
  // opened; misaligned inter quotients fail there, loudly.
  // `resident_mtp`: the model will materialize the MTP draft layer too
  // (counted in the resident footprint check; loaded like any layer).
  GlmLayerStream(const GlmTextConfig& cfg, const std::string& checkpoint_dir,
                 int rank = 0, int world = 1,
                 GlmResidency residency = GlmResidency::Streaming,
                 GlmHeadSharding head = GlmHeadSharding::Full,
                 bool resident_mtp = false);
  ~GlmLayerStream();
  GlmLayerStream(const GlmLayerStream&) = delete;
  GlmLayerStream& operator=(const GlmLayerStream&) = delete;

  // Loads layer `layer` (0..num_hidden_layers-1, or mtp_layer() when the
  // draft layer is present). STREAMING: frees the previously resident layer
  // first; the returned view stays valid until the next
  // load_layer/release_layer. RESIDENT: materializes into the layer's own
  // bump on first call and serves the cached view forever after — no
  // storage reads, no sync on cache hits, release_layer is a no-op.
  const GlmLayerResident& load_layer(int layer);

  // Loads embed/lm_head/final norm once; persists across load_layer calls.
  const GlmGlobalsResident& load_globals();

  void release_layer();
  void release_globals();

  // RESIDENT mode: drops the checkpoint mappings and evicts their page
  // cache. The caller decides when the load is complete (the model calls
  // it after its first full forward — every layer its stack will ever
  // read is on the device by then; a caller that also wants the MTP draft
  // loads it first). Why it matters: the mmapped checkpoint (~70 GB at
  // GLM) cannot be dropped by the kernel while mapped, so an 80 GB
  // resident model + the cache pinned every decode box at its memory
  // watermark — and the kernel answered by swapping the process's own
  // cold pages (tokenizer tables, heap), each faulting back in at 2-10 ms
  // in the decode loop. After this, load_layer of a
  // never-materialized layer and hash_replicated() throw: the bytes are
  // gone by design. Streaming mode ignores the call.
  void release_sources();
  bool sources_released() const { return sources_released_; }

  // The resident image cache (GlmResidentImage): process-wide location,
  // set once by the app before any stream is constructed; "" (the
  // default) disables it. Every RESIDENT stream then restores layers from
  // `dir/<key>.img` when present and captures the ones it had to build.
  // The key covers the checkpoint (every shard's header fold), config.json,
  // world, rank, head sharding, and the loader's format version.
  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  // Layers restored from / captured to the image by this stream.
  int image_layers_restored() const { return image_restored_; }
  int image_layers_captured() const { return image_captured_; }
  // RESIDENT: a materialized layer's whole device byte range (the bump) —
  // what the image stores; the round-trip test compares it bitwise.
  // {nullptr, 0} when the layer is not materialized.
  std::pair<const void*, size_t> resident_layer_span(int layer) const;
  // A resident layer's bytes in grant order — the staging mirror's and the
  // image's layout, whatever the device placement (side grants included;
  // all of them must still be mapped). `dst` holds layer_bytes. Synchronous.
  void copy_resident_layer(int layer, void* dst) const;

  // Exact device bytes load_layer will use for a layer — the same formula
  // that sizes the bump; load_layer throws if actual usage ever differs,
  // so the formula and the allocator cannot silently drift apart. At
  // world>1 this is the LOCAL geometry's formula (the sharded bump).
  static size_t layer_bytes(const GlmTextConfig& cfg, int layer,
                            int rank = 0, int world = 1);
  // `head` shrinks the lm-head share of the formula to the rank's slice.
  static size_t globals_bytes(const GlmTextConfig& cfg, int rank = 0,
                              int world = 1,
                              GlmHeadSharding head = GlmHeadSharding::Full);
  // The lm-head rows this rank holds (the whole vocabulary, or its slice
  // under VocabSharded) — what sizes the logits buffers.
  static int lm_vocab_count(const GlmTextConfig& cfg, int rank = 0,
                            int world = 1,
                            GlmHeadSharding head = GlmHeadSharding::Full);

  // Exact device bytes a RESIDENT stream at this rank's geometry holds
  // once every layer + the globals are materialized: the sum of every
  // layer's formula plus the globals formula. Callers use it to refuse a
  // world that cannot fit BEFORE the first allocation (at the real GLM
  // dims: world 1 ~306 GiB, world 2 ~155 GiB per rank, world 4 ~79 GiB —
  // only world 4 fits a 128 GB GB10).
  static size_t resident_bytes(const GlmTextConfig& cfg, int rank = 0,
                                int world = 1,
                                GlmHeadSharding head = GlmHeadSharding::Full);

  // bf12-only residency (common/bf16_residency.hpp): of the bytes above,
  // the packable bf16 matrices — the KDA projections, the draft's eh_proj,
  // the lm head — that a resident stream grants ASIDE (LayerBump) and the
  // model returns once their 12-bit companions exist (release_packed).
  static size_t layer_side_bytes(const GlmTextConfig& cfg, int layer, int rank = 0, int world = 1);
  static size_t globals_side_bytes(const GlmTextConfig& cfg, int rank = 0, int world = 1,
                                   GlmHeadSharding head = GlmHeadSharding::Full);
  static size_t side_bytes(const GlmTextConfig& cfg, int rank, int world, GlmHeadSharding head,
                           bool with_mtp);
  bool side_grants() const { return side_grants_; }
  // Returns a packed matrix's bf16 bytes to the device (layer < 0: the
  // globals'). 0 when `weight` was not granted aside. Nothing outstanding
  // may read it.
  size_t release_packed(int layer, const void* weight);

  // Registers the stream whose kernels READ resident layers (the model's
  // compute stream). When set, load boundaries synchronize only that
  // stream plus the loader's own dequant stream — the complete set of
  // bump readers — instead of the whole device. The device-wide wait is
  // correct for standalone callers (conservative default), but in a
  // one-PROCESS multi-rank world it deadlocks by construction: rank A's
  // spinning collective kernel never completes, so rank B's
  // cudaDeviceSynchronize inside a layer load never returns, so B never
  // posts the doorbell A spins on (measured: first-collective stall, CI
  // under post-build load — ~15ms of thread skew is enough). Peers'
  // kernels never touch this rank's bump; the precise sync is strictly
  // safer than the device-wide one everywhere, including the fabric.
  void set_reader_stream(cudaStream_t reader) { reader_ = reader; }

  const GlmTextConfig& config() const { return cfg_; }
  GlmResidency residency() const { return residency_; }
  // Capacity of the per-layer bump (the largest layer's exact size at
  // this rank's geometry). In resident mode the shared bump is never
  // allocated (each layer owns its exact-size bump instead) — this stays
  // the largest layer's formula, the number callers report.
  size_t layer_capacity() const;

  int rank() const { return rank_; }
  int world() const { return world_; }

  // Folds every replicated tensor's raw source bytes (all layers + the
  // globals) straight from the mmaps — no layer residency required, which
  // is what makes it a BOOT check. Rank-invariant by construction.
  GlmReplicatedDigest hash_replicated() const;

  // Checkpoint source bytes this rank has TOUCHED across all loads — the
  // "reconcile per-rank byte totals" input (§5.2): the sharded-class bytes
  // partition across ranks exactly once, the replicated+bridge bytes are
  // re-read by every rank. verbatim_source_bytes() is that rank-invariant
  // re-read subset, so sum_r(source) == world1_total + (world-1) *
  // verbatim reconciles the shard coverage arithmetically.
  uint64_t source_bytes_read() const { return source_bytes_; }
  uint64_t verbatim_source_bytes() const { return verbatim_bytes_; }

 private:
  GlmTextConfig cfg_;
  int rank_ = 0;
  int world_ = 1;
  GlmResidency residency_ = GlmResidency::Streaming;
  GlmHeadSharding head_ = GlmHeadSharding::Full;
  cudaStream_t reader_ = nullptr;  // bump readers' stream (see above)
  uint64_t source_bytes_ = 0;
  uint64_t verbatim_bytes_ = 0;
  bool sources_released_ = false;
  bool resident_mtp_ = false;
  bool side_grants_ = false;  // packable matrices load into releasable ranges
  std::string checkpoint_dir_;
  std::unique_ptr<GlmResidentImage> image_;  // resident mode, when configured
  int image_restored_ = 0;
  int image_captured_ = 0;
  void check_resident_footprint_fits() const;
  uint64_t resident_image_key() const;
  void open_resident_image();
  GlmReplicatedDigest compute_replicated_digest() const;
  // The image paths of load_layer (resident): lay out the views without a
  // source byte, then stream the blob in; or dump a freshly built bump.
  bool restore_layer_from_image(int layer, GlmLayerBump& bump, GlmLayerResident& out);
  void capture_layer_to_image(int layer, const GlmLayerBump& bump,
                              size_t bytes);
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::unordered_map<std::string, const TensorInfo*> tensors_;
  std::unique_ptr<GlmLayerBump> layer_bump_;  // streaming only
  std::unique_ptr<GlmLayerBump> globals_bump_;
  // Resident mode: one exact-size bump + one view per layer, alive for
  // the stream's lifetime. Index l serves load_layer(l) directly (the MTP
  // draft layer's index is mtp_layer(), inside the same range).
  std::vector<std::unique_ptr<GlmLayerBump>> resident_bumps_;
  std::vector<GlmLayerResident> resident_layers_;
  size_t capacity_ = 0;  // largest layer's formula (layer_capacity)
  GlmLayerResident resident_;  // streaming only (the active layer)
  GlmGlobalsResident globals_;
  cudaStream_t stream_ = nullptr;  // dedicated; synced before returning
  void* staging_ = nullptr;        // pinned host mirror the builds write
  size_t staging_bytes_ = 0;

  // The one layer build both residency modes share — grant sequence,
  // byte accounting, formula check and sync are identical, so resident
  // bytes are streaming bytes by construction (the parity driver pins
  // this bitwise at every world).
  void build_layer_into(int layer, GlmLayerBump& bump,
                        GlmLayerResident& out);
};

}  // namespace dgpp

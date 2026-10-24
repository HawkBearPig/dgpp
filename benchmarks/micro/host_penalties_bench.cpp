// Host preparation and full-sampler costs on deterministic synthetic inputs.
// This is not an inference or serving-throughput benchmark.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "sample/sampler.hpp"

namespace {
using Clock = std::chrono::steady_clock;
uint64_t sink = 0;

template <typename Fn>
void measure(int context, const char* regime, const char* operation, int rounds, Fn fn) {
  for (int i = 0; i < 3; ++i) fn();
  std::vector<double> times;
  for (int i = 0; i < rounds; ++i) {
    const auto start = Clock::now();
    fn();
    times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
  }
  std::sort(times.begin(), times.end());
  std::printf("%d,%s,%s,%d,%.3f\n", context, regime, operation, rounds, times[times.size() / 2]);
}
}  // namespace

int main() {
  using namespace dgpp::sample;
  constexpr int vocab = 248320;
  std::puts("# automatic neutral-penalty skip; reference always counts and applies penalties");
  std::puts("# synthetic logits/context; vocab=248320; temperature=0.8; top_k=40; top_p=0.95");
  std::puts("context,regime,operation,rounds,median_us");
  std::vector<float> logits(vocab);
  for (int i = 0; i < vocab; ++i)
    logits[i] = static_cast<float>(static_cast<int>(splitmix64_step(i) % 24001) - 12000) / 1000.0f;
  const auto layout = vocab_layout(vocab, 1);
  for (const int length : {512, 8192, 262144}) {
    std::vector<int32_t> context(length);
    for (int i = 0; i < length; ++i) context[i] = splitmix64_step(i + 17) % 4096;
    for (const bool active : {false, true}) {
      Params p;
      p.temperature = 0.8f;
      p.top_k = 40;
      p.top_p = 0.95f;
      if (active) {
        p.repetition_penalty = 1.1f;
        p.frequency_penalty = 0.2f;
        p.presence_penalty = 0.1f;
      }
      const char* regime = active ? "active" : "neutral";
      measure(length, regime, "copy_count_penalty_reference", 31, [&] {
        auto adjusted = logits;
        apply_penalties(adjusted.data(), vocab, 0, p, count_context(context));
        asm volatile("" : : "g"(adjusted.data()), "g"(adjusted.size()) : "memory");
      });
      measure(length, regime, "copy_count_penalty", 31, [&] {
        auto adjusted = logits;
        apply_context_penalties(adjusted.data(), vocab, 0, p, context);
        // Keep every adjusted byte observable even when the helper is a no-op.
        asm volatile("" : : "g"(adjusted.data()), "g"(adjusted.size()) : "memory");
      });
      Rng rng{42, 0};
      measure(length, regime, "sample_full_logits", 11, [&] {
        sink += static_cast<uint32_t>(
            sample_full_logits(logits.data(), vocab, layout, p, rng, context).token);
      });
    }
  }
  std::fprintf(stderr, "sink=%llu\n", static_cast<unsigned long long>(sink));
}

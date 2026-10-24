// Micro test harness (dependency-free). Add framework later if desired.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace dgpp::test {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// A missing optional prerequisite is a distinct outcome, never a passing case.
class Skip final : public std::exception {
 public:
  explicit Skip(const std::string& reason) : reason_(reason) {}
  const char* what() const noexcept override { return reason_.c_str(); }

 private:
  std::string reason_;
};

[[noreturn]] inline void skip(const std::string& reason) {
  throw Skip(reason);
}

// Runs every test whose name contains DGPP_TEST_FILTER (unset/empty = all).
// Sanitizer passes target the smem-heavy groups (e.g. DGPP_TEST_FILTER=select
// for racecheck on the bitonic kernels) so the gate stays rigorous without
// paying full-suite instrumentation time on every tool.
inline int run_all() {
  const char* filter = std::getenv("DGPP_TEST_FILTER");
  int failed = 0;
  size_t ran = 0;
  size_t skipped = 0;
  for (auto& tc : registry()) {
    if (filter && filter[0] != '\0' &&
        !std::strstr(tc.name, filter))
      continue;
    ++ran;
    try {
      tc.fn();
      std::printf("[ OK ] %s\n", tc.name);
    } catch (const Skip& e) {
      ++skipped;
      std::printf("[SKIP] %s: %s\n", tc.name, e.what());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[FAIL] %s: %s\n", tc.name, e.what());
    }
  }
  if (ran == 0) {
    std::printf("no tests match filter \"%s\"\n", filter ? filter : "");
    return 1;
  }
  std::printf("%zu tests, %d failed, %zu passed, %zu skipped\n", ran, failed,
              ran - static_cast<size_t>(failed) - skipped, skipped);
  return failed == 0 ? 0 : 1;
}

}  // namespace dgpp::test

#define DGPP_TEST(name)                                                        \
  static void dgpp_test_##name();                                             \
  static ::dgpp::test::Registrar dgpp_reg_##name{#name, dgpp_test_##name};     \
  static void dgpp_test_##name()

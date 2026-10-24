// HF hub-cache resolution tests against synthetic cache trees — no
// dependence on the machine's real cache or environment.
#include "loaders/hf_cache.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

#include "../common/temp_dir.hpp"
#include "common/test.hpp"

namespace fs = std::filesystem;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The hub's on-disk name for a repo id, built independently of the
// implementation so the tests are a cross-check, not a mirror (a
// disagreement fails the resolution loudly as "no cached model").
std::string hub_dir_name(const std::string& model_id) {
  std::string name = "models--";
  for (const char c : model_id) {
    if (c == '/') name += "--";
    else name += c;
  }
  return name;
}

// Builds a minimal hub cache: refs/main -> commit, snapshots/<commit>/ with
// config.json. `with_ref=false` omits refs (the fresh-download shape).
fs::path make_cache(const fs::path& root, const std::string& model_id,
                    const std::string& commit, bool with_ref) {
  const fs::path model = root / hub_dir_name(model_id);
  fs::create_directories(model / "snapshots" / commit);
  fs::create_directories(model / "refs");
  FILE* f =
      std::fopen((model / "snapshots" / commit / "config.json").c_str(), "wb");
  std::fputs("{}", f);
  std::fclose(f);
  if (with_ref) {
    f = std::fopen((model / "refs" / "main").c_str(), "wb");
    std::fputs((commit + "\n").c_str(), f);
    std::fclose(f);
  }
  return model;
}

std::string resolve(const fs::path& root, const std::string& model_id,
                    std::string* error) {
  return dgpp::hf::model_dir_in_root(root.string(), model_id, error);
}

}  // namespace

DGPP_TEST(hf_cache_resolves_ref_pinned_snapshot) {
  // GIVEN a cache with refs/main pinning commit abc123:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_ref");
  const fs::path& root = fixture.path();
  make_cache(root, "org/name", "abc123", /*with_ref=*/true);

  // WHEN resolving the model:
  std::string error;
  const std::string dir = resolve(root, "org/name", &error);

  // THEN the ref-pinned snapshot is returned, error-free:
  require(!dir.empty(), error);
  require(dir == (root / "models--org--name" / "snapshots" / "abc123").string(),
          "expected the ref-pinned snapshot, got " + dir);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_resolves_single_snapshot_without_ref) {
  // GIVEN a cache with no refs (fresh manual download shape):
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_noref");
  const fs::path& root = fixture.path();
  make_cache(root, "org/name", "abc123", /*with_ref=*/false);

  // WHEN resolving the model:
  std::string error;
  const std::string dir = resolve(root, "org/name", &error);

  // THEN the single snapshot is unambiguous and returned:
  require(!dir.empty(), error);
  require(dir == (root / "models--org--name" / "snapshots" / "abc123").string(),
          "expected the single snapshot, got " + dir);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_refuses_ambiguous_snapshots_without_ref) {
  // GIVEN a cache with two snapshots and no refs/main:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_ambig");
  const fs::path& root = fixture.path();
  make_cache(root, "org/name", "abc123", /*with_ref=*/false);
  make_cache(root, "org/name", "def456", /*with_ref=*/false);

  // WHEN resolving the model:
  std::string error;
  const std::string dir = resolve(root, "org/name", &error);

  // THEN resolution refuses to guess and says why:
  require(dir.empty(), "ambiguous cache must not resolve");
  require(error.find("ambiguous") != std::string::npos,
          "error should name the ambiguity, got: " + error);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_refuses_missing_model_loudly) {
  // GIVEN a cache without the requested model:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_missing");
  const fs::path& root = fixture.path();

  // WHEN resolving a model that is not cached:
  std::string error;
  const std::string dir = resolve(root, "org/absent", &error);

  // THEN the error names the model and the root searched:
  require(dir.empty(), "missing model must not resolve");
  require(error.find("org/absent") != std::string::npos &&
              error.find(root.string()) != std::string::npos,
          "error should name model and root, got: " + error);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_refuses_snapshot_without_config) {
  // GIVEN a ref-pinned snapshot whose config.json never landed:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_noconf");
  const fs::path& root = fixture.path();
  const fs::path model = make_cache(root, "org/name", "abc123", true);
  fs::remove(model / "snapshots" / "abc123" / "config.json");

  // WHEN resolving the model:
  std::string error;
  const std::string dir = resolve(root, "org/name", &error);

  // THEN the failure is at the cache interface, in the cache's vocabulary:
  require(dir.empty(), "config-less snapshot must not resolve");
  require(error.find("config.json") != std::string::npos,
          "error should mention config.json, got: " + error);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_refuses_ref_pointing_at_missing_snapshot) {
  // GIVEN refs/main pinning a commit whose snapshot directory is absent:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_dangling");
  const fs::path& root = fixture.path();
  const fs::path model = make_cache(root, "org/name", "abc123", false);
  (void)model;
  fs::create_directories(root / "models--org--name" / "refs");
  FILE* f = std::fopen((root / "models--org--name" / "refs" / "main").c_str(),
                       "wb");
  std::fputs("0000000000000000000000000000000000000000\n", f);
  std::fclose(f);

  // WHEN resolving the model:
  std::string error;
  const std::string dir = resolve(root, "org/name", &error);

  // THEN the dangling ref is reported, not guessed past:
  require(dir.empty(), "dangling ref must not resolve");
  require(error.find("interrupted") != std::string::npos,
          "error should suggest the interrupted-pull cause, got: " + error);
  fs::remove_all(root);
}

DGPP_TEST(hf_cache_honors_hf_hub_cache_precedence) {
  // GIVEN HF_HUB_CACHE pointing at a synthetic root (huggingface_hub's
  // highest-precedence location) with a cached model:
  const dgpp::test::TempDir fixture("dgpp_hf_cache_test_env");
  const fs::path& root = fixture.path();
  make_cache(root, "org/name", "abc123", true);
  const char* saved = std::getenv("HF_HUB_CACHE");
  require(::setenv("HF_HUB_CACHE", root.string().c_str(), 1) == 0,
          "setenv failed");

  // WHEN resolving by model id with no explicit root:
  std::string error;
  const std::string dir = dgpp::hf::model_dir("org/name", &error);

  // THEN the env-pinned root is honored (and the env is restored):
  if (saved)
    ::setenv("HF_HUB_CACHE", saved, 1);
  else
    ::unsetenv("HF_HUB_CACHE");
  require(!dir.empty(), error);
  require(dir.rfind(root.string(), 0) == 0,
          "expected the HF_HUB_CACHE root to win, got: " + dir);
  fs::remove_all(root);
}

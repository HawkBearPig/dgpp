#pragma once
#include <set>
#include <string>

extern char** environ;

namespace dgpp::adapters {
inline std::string numerical_environment() {
  // Runtime overrides must agree too. Host transport/cache/log locations are
  // intentionally rank-local; every other DGPP override joins the identity.
  const std::set<std::string> local = {"DGPP_ROCE_DEVICES",
                                       "DGPP_ROCE_GID_INDICES",
                                       "DGPP_RESIDENT_CACHE_DIR",
                                       "DGPP_RESIDENT_CACHE_VERIFY",
                                       "DGPP_LOG_LEVEL",
                                       "DGPP_MLOCK",
                                       "DGPP_NO_SWAP",
                                       "DGPP_NODES",
                                       "DGPP_SSH_USER",
                                       "DGPP_CLUSTER_CONFIG",
                                       "DGPP_HTTP_PORT",
                                       "DGPP_FABRIC_PORT",
                                       "DGPP_JOURNAL_PORT",
                                       "DGPP_LOG_DIR",
                                       "DGPP_STAGE_DIR",
                                       "DGPP_RELEASE_DIR",
                                       "DGPP_HTTP_BIND",
                                       "DGPP_NODE_OVERRIDES",
                                       "DGPP_BUILD_DIR",
                                       "DGPP_DATA_DIR",
                                       "DGPP_ENV_FILE",
                                       "DGPP_HERETIC_WORKER",
                                       "DGPP_TEST_FILTER"};
  std::set<std::string> values;
  for (char** entry = environ; entry && *entry; ++entry) {
    const std::string value(*entry);
    if (value.starts_with("DGPP_") && !local.count(value.substr(0, value.find('='))))
      values.insert(value);
  }
  std::string out;
  for (const auto& value : values) {
    out += value;
    out += '\0';
  }
  return out;
}

}  // namespace dgpp::adapters

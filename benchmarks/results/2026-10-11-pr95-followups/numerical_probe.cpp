#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/bf16_residency.hpp"
#include "models/qwen/model35.hpp"
int main(int argc, char** argv) {
  if (argc < 4) return 2;
  auto cfg = dgpp::Qwen35TextConfig::from_json_file(std::string(argv[1]) + "/config.json");
  dgpp::Qwen35LayerStream::set_resident_image_dir(std::string(getenv("HOME")) +
                                                  "/.cache/dgpp/resident");
  dgpp::Qwen35Model model(cfg, argv[1], 4096, 8192, dgpp::LoaderResidency::Resident, nullptr, 0, 1,
                          1, 8, false);
  for (int file = 3; file < argc; file++) {
    std::ifstream in(argv[file]);
    std::vector<int64_t> tokens;
    int64_t id;
    while (in >> id) tokens.push_back(id);
    if (tokens.size() < 2049 || tokens.size() > 4097) return 3;
    int64_t last = tokens.back();
    tokens.pop_back();
    auto result = model.forward(tokens, false);
    std::ofstream out(std::string(argv[2]) + "-" + std::to_string(file - 3) + ".tsv");
    out << std::setprecision(12);
    out << "position\ttarget\tlogp\ttop1\ttop1_logit\ttop2_logit\tlogits_hash\n";
    int V = cfg.vocab_size;
    for (size_t row = 0; row < tokens.size(); row++) {
      const float* logits = result.logits.data() + row * V;
      int best = 0, second = 1;
      if (logits[second] > logits[best]) std::swap(best, second);
      for (int i = 2; i < V; i++) {
        if (logits[i] > logits[best]) {
          second = best;
          best = i;
        } else if (logits[i] > logits[second])
          second = i;
      }
      double sum = 0;
      uint64_t hash = 14695981039346656037ull;
      for (int i = 0; i < V; i++) {
        if (!std::isfinite(logits[i])) return 4;
        sum += std::exp(double(logits[i]) - logits[best]);
        uint32_t bits;
        std::memcpy(&bits, logits + i, 4);
        hash ^= bits;
        hash *= 1099511628211ull;
      }
      int64_t target = row + 1 < tokens.size() ? tokens[row + 1] : last;
      double lp = double(logits[target]) - logits[best] - std::log(sum);
      out << row << '\t' << target << '\t' << lp << '\t' << best << '\t' << logits[best] << '\t'
          << logits[second] << '\t' << hash << '\n';
    }
    std::cout << "scored " << tokens.size() << " positions from " << argv[file] << std::endl;
  }
}

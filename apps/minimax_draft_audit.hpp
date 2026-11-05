#pragma once
// Eager real-checkpoint draft snapshots for independent prediction checks.
// The target still verifies and commits a greedy prefix on every round.
namespace {
struct DraftAuditCase {
  std::string name;
  std::vector<int64_t> ids;
};
std::vector<DraftAuditCase> draft_audit_cases(const std::string& file) {
  const auto contents = read(file);
  const auto json = dgpp::minijson::parse_document(contents);
  if (!json.root.is_array()) throw std::runtime_error("invalid draft audit manifest");
  std::vector<DraftAuditCase> cases;
  for (const auto& value : json.root.items()) {
    DraftAuditCase c;
    c.name = value.at("name").as_string();
    if (c.name.empty() || c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_") != std::string::npos)
      throw std::runtime_error("draft audit case needs a simple name");
    for (const auto& token : value.at("ids").items()) c.ids.push_back(token.as_int());
    if (c.ids.empty()) throw std::runtime_error("empty draft audit prompt");
    cases.push_back(std::move(c));
  }
  if (cases.empty()) throw std::runtime_error("empty draft audit manifest");
  return cases;
}
template <class T>
std::vector<T> draft_audit_copy(const T* p, size_t n) {
  std::vector<T> out(n);
  DGPP_CUDA_OK(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return out;
}
void audit_draft(dgpp::MinimaxModel& model, const std::vector<DraftAuditCase>& cases,
                 int rounds, const std::string& dump, dgpp::net::CollectiveBus* bus,
                 int rank, int world, bool native, dgpp::LatentFormat format) {
  if (dump.empty() || rounds < 1) throw std::runtime_error("draft audit needs --dump-dir and positive --draft-rounds");
  uint16_t* scratch = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(&scratch, dgpp::kPickScratchElems(world) * sizeof(uint16_t), cudaHostAllocDefault));
  const auto pick = [&](const auto& out) {
    std::vector<dgpp::sample::Candidate> local;
    for (size_t at = 0; at < out.logits.size(); at += out.lm_vocab_count)
      local.push_back(dgpp::sample::local_max(out.logits.data() + at, out.lm_vocab_count, out.lm_vocab_begin));
    if (bus) return dgpp::bus_greedy_pick_rows(*bus, rank, world, local, scratch, 120000);
    std::vector<int32_t> ids;
    for (const auto& c : local) ids.push_back(c.id);
    return ids;
  };
  const auto& df = model.dspark_drafter();
  const auto& cfg = df.cfg();
  const int h = cfg.hidden_size, q = cfg.query_rows(), d = cfg.drafts(), kv = cfg.local_kv_row(world);
  for (const auto& c : cases) {
    const auto root = std::filesystem::path(dump) / c.name / ("rank" + std::to_string(rank));
    std::filesystem::create_directories(root);
    // Capture the actual target layer outputs once, separately from its live
    // session, so the reference can check FC fusion and context K/V too.
    const auto full = model.forward(c.ids, true);
    for (int tap : cfg.target_layer_ids)
      write(root / ("target-layer-" + std::to_string(tap) + ".bf16"), full.layer_states.at(tap));
    auto out = model.session_prefill(0, c.ids);
    write(root / "features.f32", draft_audit_copy(df.acc(), c.ids.size() * h));
    write(root / "prompt.i64", c.ids);
    int32_t anchor = pick(out).back();
    std::vector<int32_t> transcript{anchor};
    for (int round = 0; round < rounds; ++round) {
      const auto path = root / ("block-" + std::to_string(round));
      std::filesystem::create_directories(path);
      const int64_t position = model.session_position(0);
      std::vector<int32_t> proposals;
      if (!model.dflash2_draft(0, anchor, &proposals)) throw std::runtime_error("draft audit could not draft");
      DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
      write(path / "hidden.bf16", draft_audit_copy(df.block_hidden(), size_t(q) * h));
      write(path / "logits.f32", draft_audit_copy(df.block_logits(), size_t(q) * model.lm_vocab_count()));
      write(path / "proposals.i32", proposals);
      const auto* table = model.pool().request_table_row(0);
      for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        const auto plane = model.pool().draft_view(layer);
        for (const auto& [data, name] : {std::pair{plane.k, "k"}, {plane.v, "v"}}) {
          std::vector<uint16_t> logical(size_t(position + q) * kv);
          for (int64_t at = 0; at < position + q; ++at) {
            const size_t physical = size_t(table[at / 128]) * 128 + at % 128;
            DGPP_CUDA_OK(cudaMemcpy(logical.data() + at * kv, data + physical * kv, kv * sizeof(uint16_t), cudaMemcpyDeviceToHost));
          }
          write(path / (std::string(name) + "-" + std::to_string(layer) + ".bf16"), logical);
        }
      }
      std::vector<int64_t> feed{anchor};
      feed.insert(feed.end(), proposals.begin(), proposals.end());
      out = model.session_verify(0, feed);
      write(path / "verify-features.f32", draft_audit_copy(df.acc(), size_t(d + 1) * h));
      const auto targets = pick(out);
      write(path / "targets.i32", targets);
      int accepted = 0;
      while (accepted < d && proposals[accepted] == targets[accepted]) ++accepted;
      std::ofstream meta(path / "metadata.json");
      meta << "{\"position\":" << position << ",\"anchor\":" << anchor
           << ",\"accepted\":" << accepted << ",\"world\":" << world << ",\"rank\":" << rank
           << ",\"native\":" << (native ? "true" : "false") << ",\"kv_dtype\":\""
           << dgpp::latent_format_name(format) << "\",\"vocab_begin\":" << out.lm_vocab_begin
           << ",\"vocab_count\":" << out.lm_vocab_count << "}\n";
      model.session_rollback(0, accepted + 1, d + 1);
      transcript.insert(transcript.end(), proposals.begin(), proposals.begin() + accepted);
      anchor = targets[accepted];
      transcript.push_back(anchor);
      std::printf("[draft_audit] case %s round %d position %lld accepted %d\n", c.name.c_str(), round, static_cast<long long>(position), accepted);
      std::fflush(stdout);
    }
    write(root / "generated.i32", transcript);
    model.session_close(0);
  }
  cudaFreeHost(scratch);
}
}  // namespace

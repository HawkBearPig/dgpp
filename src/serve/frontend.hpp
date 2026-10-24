#pragma once
// The real ModelFrontend: the Stage 3/3b exact tokenizer and chat
// template behind the service's interface (M6 Stage 4). The service never
// links CUDA; this adapter is likewise host-only.
//
// Chat rendering: the service's template globals (minijson DOM: the
// normalized OpenAI messages array plus tools / reasoning_effort /
// clear_thinking when the request carried them) convert to the
// template's Value model via text::Value::from_minijson — member order
// preserved, exactly what the template interpreter consumes — with
// add_generation_prompt=true (this is a generation request).
//
// Markers (M6 6f): the template's reasoning and tool-call tokens looked
// up by text among the tokenizer's added tokens at construction — the
// service's parser keys on their ids, the forced tool_choice prefix on
// their text.
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/dsv41_prompt.hpp"
#include "text/dsv4_prompt.hpp"
#include "text/tokenizer.hpp"
#include "text/tool_parser.hpp"
#include "serve/generation_service.hpp"

namespace dgpp::serve {

class TextFrontend : public ModelFrontend {
 public:
  // model_opens_thinking: the family's declaration that its template's
  // generation prompt opens no reasoning block and the model opens its
  // own <think> (ChatMarkers::model_opens_thinking; ServeFamily states it).
  TextFrontend(const dgpp::text::Tokenizer* tok,
              const dgpp::text::ChatTemplate* tpl, bool model_opens_thinking = false)
      : tok_(tok), tpl_(tpl) {
    if (tok_ == nullptr || tpl_ == nullptr)
      throw std::invalid_argument(
          "TextFrontend: tokenizer and chat template must both be loaded");
    markers_ = dgpp::text::ChatMarkers::from_tokenizer(*tok_);
    markers_.model_opens_thinking = model_opens_thinking;
  }

  std::vector<int64_t> encode_text(std::string_view text) const override {
    return tok_->encode(text);
  }

  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    // skip_special_tokens=true — the SSE content contract (EOS and the
    // other special tokens never appear in generated text).
    return tok_->decode(ids, /*skip_special_tokens=*/true);
  }

  bool template_reads(std::string_view name) const override { return tpl_->reads(name); }

  ReasoningSettings reasoning_settings(std::string_view effort) const override {
    auto out = ModelFrontend::reasoning_settings(effort);
    if (!out.effort) return out;
    // The checkpoint contracts differ: Qwen exposes low/medium/xhigh;
    // GLM's effort-aware template exposes low/high/max. Never pass an
    // unrecognised value through GLM's silent fallback to max.
    if (markers_.tool_format() == dgpp::text::ToolFormat::kQwenXml) {
      out.effort = effort == "minimal" ? "low" :
                   (effort == "high" || effort == "max") ? "xhigh" : std::string(effort);
    } else if (markers_.tool_format() == dgpp::text::ToolFormat::kGlmMarkers) {
      out.effort = (effort == "minimal" || effort == "low") ? "low" :
                   (effort == "medium" || effort == "high") ? "high" : "max";
    }
    return out;
  }

  std::string render_chat(const minijson::Value& globals) const override {
    dgpp::text::Value::Members members;
    for (const minijson::Member& m : globals.members()) {
      members.emplace_back(m.key, dgpp::text::Value::from_minijson(m.value));
    }
    members.emplace_back("add_generation_prompt",
                         dgpp::text::Value::boolean(true));
    return tpl_->render(dgpp::text::Value::map_value(std::move(members)));
  }

  dgpp::text::ChatMarkers markers() const override { return markers_; }
  std::vector<int64_t> boundary_token_ids() const override {
    std::vector<int64_t> ids;
    for (const dgpp::text::ChatMarker& m : markers_.role_markers) ids.push_back(m.id);
    return ids;
  }

 private:
  const dgpp::text::Tokenizer* tok_;
  const dgpp::text::ChatTemplate* tpl_;
  dgpp::text::ChatMarkers markers_;
};

// The DeepSeek-V4.1 frontend (docs/deepseek_v41_flash_plan.md G6): the
// checkpoint ships no chat template — text/dsv41_prompt renders its
// encoder's format; the markers come off its tokenizer (<think>,
// </think>, the ｜DSML｜ tag token, the <｜User｜> / <｜Assistant｜> /
// <｜System｜> turn markers as the prefix cache's boundaries).
class Dsv41Frontend : public ModelFrontend {
 public:
  explicit Dsv41Frontend(const dgpp::text::Tokenizer* tok) : tok_(tok) {
    if (tok_ == nullptr) throw std::invalid_argument("Dsv41Frontend: the tokenizer must be loaded");
    markers_ = dgpp::text::ChatMarkers::from_tokenizer(*tok_);
  }
  std::vector<int64_t> encode_text(std::string_view text) const override { return tok_->encode(text); }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    return tok_->decode(ids, /*skip_special_tokens=*/true);
  }
  bool template_reads(std::string_view name) const override { return dgpp::text::Dsv41Prompt::reads(name); }
  std::string render_chat(const minijson::Value& globals) const override {
    return dgpp::text::Dsv41Prompt::render(globals);
  }
  dgpp::text::ChatMarkers markers() const override { return markers_; }
  std::vector<int64_t> boundary_token_ids() const override {
    std::vector<int64_t> ids;
    for (const dgpp::text::ChatMarker& m : markers_.role_markers) ids.push_back(m.id);
    return ids;
  }

 private:
  const dgpp::text::Tokenizer* tok_;
  dgpp::text::ChatMarkers markers_;
};

// The DeepSeek-V4 frontend (DeepSeek-V4-Flash-0731, model_type
// deepseek_v4): text/dsv4_prompt renders its encoder's format; the markers
// come off its tokenizer (<think>, </think>, the ｜DSML｜ tag token, the
// <｜User｜> / <｜Assistant｜> turn markers as the prefix cache's boundaries —
// this tokenizer has no <｜System｜>). The tokenizer shares the tag token
// with DeepSeek-V4.1's, so the DSML spelling is stated here:
// DsmlDialect::kV4 ("<｜DSML｜tool_calls>", no space after the tag, no
// namespaces). The grammar vocabulary the engine builds must state the
// same (GrammarVocab::from_tokenizer's dsml_dialect argument).
class Dsv4Frontend : public ModelFrontend {
 public:
  static constexpr dgpp::text::DsmlDialect kDsmlDialect = dgpp::text::DsmlDialect::kV4;

  explicit Dsv4Frontend(const dgpp::text::Tokenizer* tok) : tok_(tok) {
    if (tok_ == nullptr) throw std::invalid_argument("Dsv4Frontend: the tokenizer must be loaded");
    markers_ = dgpp::text::ChatMarkers::from_tokenizer(*tok_);
    markers_.dsml_dialect = kDsmlDialect;
  }
  std::vector<int64_t> encode_text(std::string_view text) const override { return tok_->encode(text); }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    return tok_->decode(ids, /*skip_special_tokens=*/true);
  }
  bool template_reads(std::string_view name) const override { return dgpp::text::Dsv4Prompt::reads(name); }
  std::string render_chat(const minijson::Value& globals) const override {
    return dgpp::text::Dsv4Prompt::render(globals);
  }
  dgpp::text::ChatMarkers markers() const override { return markers_; }
  std::vector<int64_t> boundary_token_ids() const override {
    std::vector<int64_t> ids;
    for (const dgpp::text::ChatMarker& m : markers_.role_markers) ids.push_back(m.id);
    return ids;
  }

 private:
  const dgpp::text::Tokenizer* tok_;
  dgpp::text::ChatMarkers markers_;
};

}  // namespace dgpp::serve

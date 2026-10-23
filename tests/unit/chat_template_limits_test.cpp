#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "text/chat_template.hpp"

namespace {
using dgpp::text::ChatTemplate;
using dgpp::text::Value;
constexpr size_t kMiB = 1024 * 1024;
std::string render(const std::string& source, Value::Members members = {}) {
  return ChatTemplate::compile(source).render(Value::map_value(std::move(members)));
}
void equal(const std::string& a, const std::string& b) {
  if (a != b) throw std::runtime_error("unexpected render result");
}
void rejects(const std::string& source, Value::Members members, const char* reason) {
  bool rejected = false;
  try {
    (void)render(source, std::move(members));
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    if (message.find("chat-template: line ") == std::string::npos ||
        message.find(reason) == std::string::npos)
      throw;
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("expected structural refusal");
  equal(render("{{ 40 + 2 }}"), "42");
}
Value text(size_t n, char c = 'x') {
  return Value::string_value(std::string(n, c));
}
std::string nested(size_t n, const std::string& open, const std::string& close) {
  std::string s;
  for (size_t i = 0; i < n; ++i) s += open;
  s += "1";
  for (size_t i = 0; i < n; ++i) s += close;
  return s;
}
}  // namespace

DGPP_TEST(chat_template_limits_compile) {
  equal(render(std::string(kMiB, 'x')), std::string(kMiB, 'x'));
  rejects(std::string(kMiB + 1, 'x'), {}, "source bytes");
  equal(render("{{ " + nested(31, "(", ")") + " }}"), "1");
  rejects("{{ " + nested(32, "(", ")") + " }}", {}, "parse depth");
  equal(render(nested(63, "{% if true %}", "{% endif %}")), "1");
  rejects(nested(64, "{% if true %}", "{% endif %}"), {}, "parse depth");
  rejects("{{ " + nested(10000, "(", ")") + " }}", {}, "expression tokens");
  std::string chain = "1";
  for (int i = 0; i < 254; ++i) chain += " + 1";
  equal(render("{{ " + chain + " }}"), "255");
  chain += " + 1";
  // Iterative parsing still creates a recursive left-associated AST. The token
  // cap bounds its traversal/destruction on success and on syntax-error unwind.
  (void)ChatTemplate::compile("{{ " + chain + " }}");
  rejects("{{ " + chain + " }}", {}, "render depth");
  rejects("{{ " + chain + " + }}", {}, "expected an expression");
  rejects("{{ " + chain + " + 1 }}", {}, "expression tokens");
  std::string wide = "[-1";
  for (int i = 0; i < 254; ++i) wide += ",1";
  (void)ChatTemplate::compile("{{ " + wide + "] }}");  // exactly 512 tokens
  rejects("{{ " + wide + ",1] }}", {}, "expression tokens");
  std::string unary;
  for (int i = 0; i < 100; ++i) unary += "not ";
  rejects("{{ " + unary + "true }}", {}, "parse depth");
}

DGPP_TEST(chat_template_limits_render_depth) {
  const std::string macro =
      "{% macro f(n) %}{% if n > 0 %}{{ f(n - 1) }}{% else %}x{% endif %}{% endmacro %}{{ f(n) }}";
  equal(render(macro, {{"n", Value::integer(10)}}), "x");
  rejects(macro, {{"n", Value::integer(1000)}}, "render depth");
  rejects("{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}", {}, "render depth");
  rejects("{% macro a() %}{{ b() }}{% endmacro %}{% macro b() %}{{ a() }}{% endmacro %}{{ a() }}",
          {}, "render depth");
}

DGPP_TEST(chat_template_limits_range) {
  equal(render("{{ range(262144)|length }}"), "262144");
  equal(render("{{ range(262144,0,-1)|length }}"), "262144");
  rejects("{{ range(262145)|length }}", {}, "range elements");
  rejects("{{ range(262145,0,-1)|length }}", {}, "range elements");
  rejects("{{ range(9223372036854775807)|length }}", {}, "range elements");
  const auto lo = Value::integer(std::numeric_limits<int64_t>::min());
  const auto hi = Value::integer(std::numeric_limits<int64_t>::max());
  rejects("{{ range(lo,hi)|length }}", {{"lo", lo}, {"hi", hi}}, "range elements");
  equal(render("{{ range(hi,lo,lo)|length }}", {{"lo", lo}, {"hi", hi}}), "2");
  equal(render("{{ range(3,0)|length }}"), "0");
}

DGPP_TEST(chat_template_limits_strings_and_output) {
  equal(render("{{ s }}", {{"s", text(16 * kMiB)}}).size() == 16 * kMiB ? "ok" : "bad", "ok");
  rejects("{{ s }}", {{"s", text(16 * kMiB + 1)}}, "string bytes");
  equal(render("{{ s + s }}", {{"s", text(8 * kMiB)}}).size() == 16 * kMiB ? "ok" : "bad", "ok");
  rejects("{{ s + s }}", {{"s", text(8 * kMiB + 1)}}, "string bytes");
  rejects("{{ s ~ s }}", {{"s", text(8 * kMiB + 1)}}, "string bytes");
  rejects("{{ s|replace('', r) }}", {{"s", text(8192)}, {"r", text(2048)}}, "string bytes");
  equal(render("{{ s|replace('x', r)|length }}", {{"s", text(8192)}, {"r", text(2048)}}),
        "16777216");
  rejects("{{ s|replace('x', r) }}", {{"s", text(8193)}, {"r", text(2048)}}, "string bytes");
  rejects(
      "{% set n = namespace(s='x') %}{% for i in range(25) %}{% set n.s = n.s + n.s %}{% endfor %}",
      {}, "string bytes");
  // Four discarded 16 MiB captures exactly fill the cumulative emission budget.
  const std::string captures = "{% for i in range(n) %}{% set x %}{{ s }}{% endset %}{% endfor %}";
  equal(render(captures, {{"s", text(16 * kMiB)}, {"n", Value::integer(4)}}), "");
  rejects(captures, {{"s", text(16 * kMiB)}, {"n", Value::integer(5)}}, "emitted bytes");
  // Small independent macro buffers share a single emission budget.
  const std::string macro =
      "{% macro m() %}{{ s }}{% endmacro %}{% for i in range(n) %}{% set x = m() %}{% endfor %}";
  auto compiled = ChatTemplate::compile(macro);
  auto globals = Value::map_value({{"s", text(kMiB)}, {"n", Value::integer(64)}});
  equal(compiled.render(globals), "");
  equal(compiled.render(globals), "");  // repeated calls get fresh counters
  rejects(macro, {{"s", text(kMiB)}, {"n", Value::integer(65)}}, "emitted bytes");
  globals.set_attr("n", Value::integer(65));
  bool failed = false;
  try {
    (void)compiled.render(globals);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  if (!failed) throw std::runtime_error("expected emission refusal");
  globals.set_attr("n", Value::integer(1));
  equal(compiled.render(globals), "");  // the same compiled object recovers
}

DGPP_TEST(chat_template_limits_steps) {
  // Each shallow macro does bounded arithmetic and emits nothing. Resetting
  // counters with its local frame would let this exceed the dispatch budget.
  std::string expression = "1";
  for (int i = 0; i < 126; ++i) expression += " + 1";
  const std::string source = "{% macro m() %}{% set x = " + expression +
                             " %}{% endmacro %}{% for i in range(n) %}{{ m() }}{% endfor %}";
  equal(render(source, {{"n", Value::integer(100)}}), "");
  rejects(source, {{"n", Value::integer(40000)}}, "render steps");
  // Empty loop bodies still spend a step for every iteration.
  rejects("{% for i in range(10000) %}{% for j in range(1000) %}{% endfor %}{% endfor %}", {},
          "render steps");
}

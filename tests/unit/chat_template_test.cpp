#include "text/chat_template.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

namespace {
using dgpp::text::ChatTemplate;
using dgpp::text::Value;
constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

void equal(const std::string& actual, const std::string& expected) {
  if (actual != expected)
    throw std::runtime_error("render mismatch: expected " + expected + ", got " + actual);
}

std::string render(const std::string& source, Value::Members globals = {}) {
  return ChatTemplate::compile(source).render(Value::map_value(std::move(globals)));
}

Value::Members numbers(int64_t a, int64_t b, int64_t step = 1) {
  return {{"a", Value::integer(a)}, {"b", Value::integer(b)}, {"step", Value::integer(step)}};
}

void rejects(const std::string& source, Value::Members globals, const char* message) {
  try {
    (void)render(source, std::move(globals));
  } catch (const std::runtime_error& error) {
    const std::string text = error.what();
    if (text.find("chat-template: line 2:") != std::string::npos &&
        text.find(message) != std::string::npos)
      return;
    throw;
  }
  throw std::runtime_error("expected template error");
}
}  // namespace

DGPP_TEST(chat_template_replace_empty_pattern) {
  struct Case {
    std::string input, replacement, expected;
  };
  const std::vector<Case> cases = {
      {"", "", ""},
      {"", "-", "-"},
      {"abc", "", "abc"},
      {"abc", "-", "-a-b-c-"},
      {"é中😀", "|", "|é|中|😀|"},
      {"e\xCC\x81", "_",
       "_e_\xCC\x81"
       "_"},
      {std::string("a\0b", 3), "|", std::string("|a|\0|b|", 7)},
  };
  for (const auto& c : cases)
    equal(render("{{ text|replace('', replacement) }}",
                 {{"text", Value::string_value(c.input)},
                  {"replacement", Value::string_value(c.replacement)}}),
          c.expected);
}

DGPP_TEST(chat_template_boolean_test_distinguishes_numbers_and_strings) {
  equal(render("{% for x in [true, false, 0, 1, 'true', none, [], obj] %}"
               "{{ x is boolean }}:{{ x is not boolean }};{% endfor %}",
               {{"obj", Value::map_value({})}}),
        "True:False;True:False;False:True;False:True;False:True;False:True;False:True;False:True;");
}

DGPP_TEST(chat_template_replace_nonempty_pattern) {
  equal(render("{{ 'aaaa'|replace('aa', 'b') }}"), "bb");
  equal(render("{{ 'abc'|replace('b', '') }}"), "ac");
  equal(render("{{ 'abc'|replace('z', 'x') }}"), "abc");
}

DGPP_TEST(chat_template_integer_index_boundaries) {
  for (int64_t i : {kMin, kMax, int64_t(-4), int64_t(3)}) {
    equal(render("{{ [10,20,30][a] is undefined }}", numbers(i, 0)), "True");
    equal(render("{{ [][a] is undefined }}", numbers(i, 0)), "True");
    equal(render("{{ 'é中😀'[a] is undefined }}", numbers(i, 0)), "True");
  }
  equal(render("{{ [10,20,30][-3] }}"), "10");
  equal(render("{{ [10,20,30][-1] }}"), "30");
  equal(render("{{ [10,20,30][0] }}"), "10");
  equal(render("{{ 'é中😀'[-1] }}"), "😀");
}

DGPP_TEST(chat_template_range_boundary_steps) {
  struct Case {
    int64_t start, stop, step;
    const char* expected;
  };
  const Case cases[] = {
      {kMax - 1, kMax, 2, "[9223372036854775806]"},
      {kMin + 1, kMin, -2, "[-9223372036854775807]"},
      {kMin, kMax, kMax, "[-9223372036854775808, -1, 9223372036854775806]"},
      {kMax, kMin, kMin, "[9223372036854775807, -1]"},
      {kMin, kMin + 2, 1, "[-9223372036854775808, -9223372036854775807]"},
      {kMax, kMax - 2, -1, "[9223372036854775807, 9223372036854775806]"},
      {kMin, kMax, -1, "[]"},
      {kMax, kMin, 1, "[]"},
      {0, 0, kMin, "[]"},
      {1, 6, 2, "[1, 3, 5]"},
      {5, 0, -2, "[5, 3, 1]"},
  };
  for (const auto& c : cases)
    equal(render("{{ range(a,b,step)|tojson }}", numbers(c.start, c.stop, c.step)), c.expected);
  equal(render("{{ range(3)|tojson }}"), "[0, 1, 2]");
  equal(render("{{ range(2,4)|tojson }}"), "[2, 3]");
  rejects("\n{{ range(a,b,step) }}", numbers(0, 1, 0), "step must not be zero");
}

DGPP_TEST(chat_template_slice_boundary_steps) {
  equal(render("{{ [0,1,2][1:3:step]|tojson }}", numbers(0, 0, kMax)), "[1]");
  equal(render("{{ [0,1,2][::step]|tojson }}", numbers(0, 0, kMin)), "[2]");
  equal(render("{{ [0,1,2][a:b]|tojson }}", numbers(kMin, kMax)), "[0, 1, 2]");
  equal(render("{{ [0,1,2][a:b:-1]|tojson }}", numbers(kMax, kMin)), "[2, 1, 0]");
  equal(render("{{ [][::step]|tojson }}", numbers(0, 0, kMin)), "[]");
  equal(render("{{ 'é中😀'[1:3:step] }}", numbers(0, 0, kMax)), "中");
  equal(render("{{ 'é中😀'[::step] }}", numbers(0, 0, kMin)), "😀");
  equal(render("{{ 'abc'[::-1] }}"), "cba");
  rejects("\n{{ [0,1,2][::step] }}", numbers(0, 0, 0), "step cannot be zero");
}

DGPP_TEST(chat_template_integer_arithmetic_boundaries) {
  struct Case {
    int64_t a, b, sum;
  };
  const Case sums[] = {{kMin, 0, kMin}, {kMax, 0, kMax},     {kMin, kMax, -1},
                       {0, kMin, kMin}, {kMin, 1, kMin + 1}, {kMax, -1, kMax - 1}};
  for (const auto& c : sums) equal(render("{{ a + b }}", numbers(c.a, c.b)), std::to_string(c.sum));
  const Case differences[] = {{kMin, kMin, 0}, {kMax, kMax, 0},  {kMin, 0, kMin},
                              {kMax, 0, kMax}, {-1, kMin, kMax}, {kMin, -1, kMin + 1}};
  for (const auto& c : differences)
    equal(render("{{ a - b }}", numbers(c.a, c.b)), std::to_string(c.sum));
  equal(render("{{ a + b }}", {{"a", Value::number(1.5)}, {"b", Value::integer(2)}}), "3.5");
  equal(render("{{ a - b }}", {{"a", Value::number(1.5)}, {"b", Value::integer(2)}}), "-0.5");
  equal(render("{{ 'hello' + ' world' }}"), "hello world");
}

DGPP_TEST(chat_template_integer_arithmetic_overflow_refuses) {
  rejects("\n{{ a + b }}", numbers(kMax, 1), "integer addition overflow");
  rejects("\n{{ a + b }}", numbers(kMin, -1), "integer addition overflow");
  rejects("\n{{ a + b }}", numbers(kMin, kMin), "integer addition overflow");
  rejects("\n{{ a - b }}", numbers(kMin, 1), "integer subtraction overflow");
  rejects("\n{{ a - b }}", numbers(kMax, -1), "integer subtraction overflow");
  rejects("\n{{ a - b }}", numbers(0, kMin), "integer subtraction overflow");
}

// transformers' {% generation %} block marks the span an assistant turn
// generates. A render takes its body as written. The expected strings and
// refusals are the reference environment's (a call block): the body reads
// the enclosing scope, its own assignments end with it, a namespace
// attribute stays set, and a break may not leave it.
DGPP_TEST(chat_template_generation_block_renders_its_body_in_its_own_scope) {
  equal(render("{% set x = 1 %}{% generation %}{% set x = 2 %}{{ x }}{% endgeneration %}{{ x }}"), "21");
  equal(render("{% generation %}{% set y = 5 %}{% endgeneration %}{{ y is defined }}"), "False");
  equal(render("{% set ns = namespace(v=1) %}{% generation %}{% set ns.v = 2 %}{% endgeneration %}{{ ns.v }}"), "2");
  const Value::Members turns = {
      {"ms", Value::list_value({Value::string_value("a"), Value::string_value("b")})}};
  equal(render("{% for m in ms %}[{% generation %}{{ m }}{{ loop.index0 }}{% endgeneration %}]{% endfor %}", turns),
        "[a0][b1]");
  equal(render("a\n  {%- generation %}\n  b\n  {%- endgeneration %}\nc"), "a  bc");
  equal(render("{% generation %}a{% generation %}b{% endgeneration %}c{% endgeneration %}"), "abc");
  // A break that ends a loop of the block's own is the loop's.
  equal(render("{% generation %}{% for m in ms %}{% if m == 'b' %}{% break %}{% endif %}{{ m }}{% endfor %}"
               "{% endgeneration %}!",
               turns),
        "a!");
  rejects("\n{% for m in ms %}{% generation %}{% if m %}{% break %}{% endif %}{% endgeneration %}{% endfor %}", turns,
          "generation: 'break' outside loop");
  rejects("\n{% generation now %}x{% endgeneration %}", {}, "generation: the tag takes no arguments");
  rejects("\n{% generation %}x", {}, "missing 'endgeneration'");
  rejects("\nx{% endgeneration %}", {}, "unsupported statement 'endgeneration'");
}

// The knob gate asks which globals a template can read. A name the block
// sets is the block's alone, so a read of it after the block is the global.
DGPP_TEST(chat_template_generation_block_scopes_the_global_reads) {
  const auto inside = ChatTemplate::compile("{% generation %}{{ inner }}{% endgeneration %}");
  if (!inside.reads("inner")) throw std::runtime_error("a read inside the block is a read of the global");
  const auto after = ChatTemplate::compile("{% generation %}{% set flag = 1 %}{{ flag }}{% endgeneration %}{{ flag }}");
  if (!after.reads("flag")) throw std::runtime_error("the block's own name must not hide the global after it");
  const auto local = ChatTemplate::compile("{% generation %}{% set flag = 1 %}{{ flag }}{% endgeneration %}");
  if (local.reads("flag")) throw std::runtime_error("a name the block set and read is not the global");
}

namespace {
// The globals of the scope cases below: `label` is the name a render
// supplies and a template assigns.
Value::Members scope_globals() {
  return {{"label", Value::string_value("global")},
          {"xs", Value::list_value({Value::string_value("p")})},
          {"ms", Value::list_value({Value::string_value("a"), Value::string_value("b")})}};
}
std::string scoped(const std::string& source) { return render(source, scope_globals()); }
}  // namespace

// The reference decides from the syntax tree which scope owns a name: one
// that assigns it owns it for its whole extent, and the scopes nested in it
// read that copy. The expected strings are the reference environment's.
DGPP_TEST(chat_template_scope_owns_the_names_it_assigns_from_its_first_statement) {
  // The owner's copy is undefined until the assignment runs, also for a
  // nested scope that reads it earlier and with a global of that name
  // supplied: nested generation blocks, one block at the top level, loops.
  equal(scoped("{% generation %}{% generation %}[{{ label }}]{% endgeneration %}{% set label = 'local' %}"
               "{% endgeneration %}"),
        "[]");
  equal(scoped("{% generation %}[{{ label }}]{% endgeneration %}{% set label = 'local' %}"), "[]");
  equal(scoped("{% for a in xs %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{% endfor %}"), "[]");
  equal(scoped("{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}"), "[]");
  equal(scoped("{% for b in xs %}[{{ label is defined }}]{% endfor %}{% set label = 'local' %}[{{ label is defined }}]"),
        "[False][True]");
  // Each iteration of the owner starts over.
  equal(scoped("{% for m in ms %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = m %}{% endfor %}"), "[][]");
  // Once assigned, the nested scope reads the value, and it ends with its scope.
  equal(scoped("{% generation %}{% set label = 'local' %}{% generation %}[{{ label }}]{% endgeneration %}"
               "{% endgeneration %}[{{ label }}]"),
        "[local][global]");
  // The copy starts as the global instead when the scope reads the name
  // before it assigns it (an earlier statement, or the assignment's own
  // value), when the assignment is under an if, and when an enclosing scope
  // mentions the name anywhere, further down included.
  equal(scoped("{% generation %}{{ label }}{% generation %}[{{ label }}]{% endgeneration %}{% set label = 'local' %}"
               "{% endgeneration %}"),
        "global[global]");
  equal(scoped("{% generation %}{% generation %}[{{ label }}]{% endgeneration %}{% set label = label ~ '!' %}{{ label }}"
               "{% endgeneration %}"),
        "[global]global!");
  equal(scoped("{% generation %}{% generation %}[{{ label }}]{% endgeneration %}{% if true %}{% set label = 'local' %}"
               "{% endif %}{% endgeneration %}"),
        "[global]");
  equal(scoped("{% for a in xs %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{% endfor %}"
               "{{ label }}"),
        "[global]global");
}

// The body of a block set and a macro body are scopes like a loop's.
DGPP_TEST(chat_template_block_set_and_macro_bodies_are_scopes) {
  equal(scoped("{% set a %}{% set b = 1 %}{{ b }}{% endset %}{{ a }}{{ b is defined }}"), "1False");
  equal(scoped("{% set cap %}[{{ label }}]{% endset %}{% set label = 'local' %}{{ cap }}"), "[]");
  equal(scoped("{% set cap %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{{ label }}{% endset %}"
               "{{ cap }}{{ label }}"),
        "[global]localglobal");
  // A macro reads the top level's copy, and owns what its body assigns.
  equal(scoped("{% macro m() %}[{{ label }}]{% endmacro %}{{ m() }}{% set label = 'local' %}{{ m() }}"), "[][local]");
  equal(scoped("{% macro m() %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{{ label }}"
               "{% endmacro %}{{ m() }}"),
        "[]local");
  equal(scoped("{% macro m() %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{{ label }}"
               "{% endmacro %}{{ m() }}{{ label }}"),
        "[global]localglobal");
  // A parameter's default is evaluated in the macro's scope: it reads the
  // top level and the parameters before it, never the caller's names, and
  // a parameter after it is still undefined.
  equal(scoped("{% macro m(p=label) %}[{{ p }}]{% endmacro %}{% for label in ms %}{{ m() }}{% endfor %}"),
        "[global][global]");
  equal(scoped("{% macro m(a='1', b=a) %}[{{ a }}{{ b }}]{% endmacro %}{{ m() }}{{ m('2') }}"), "[11][22]");
  equal(scoped("{% macro m(a=label, label='x') %}[{{ a }}|{{ label }}]{% endmacro %}{{ m() }}"), "[|x]");
  equal(scoped("{% macro m(a=label) %}{% set label = 'local' %}[{{ a }}|{{ label }}]{% endmacro %}{{ m() }}"),
        "[global|local]");
  // A macro's name is assigned where it is defined.
  equal(scoped("{% for b in xs %}{{ m is defined }}{% endfor %}{% macro m() %}x{% endmacro %}{{ m() }}"), "Falsex");
  // A break in a block set ends the enclosing loop before the name is assigned.
  equal(scoped("{% for m in ms %}{% set cap %}{{ m }}{% break %}{% endset %}[{{ cap }}]{% endfor %}!{{ cap is defined }}"),
        "!False");
  equal(scoped("{% for m in ms %}[{{ m }}]{% set cap %}{% if m == 'a' %}{% break %}{% endif %}{% endset %}{% endfor %}!"),
        "[a]!");
  bool refused = false;
  try {
    (void)render("{% set cap %}{% break %}{% endset %}");
  } catch (const std::runtime_error& error) {
    refused = std::string(error.what()).find("'break' outside of a loop") != std::string::npos;
  }
  if (!refused) throw std::runtime_error("a break in a top-level block set leaves no loop");
}

// The knob gate follows the same ownership: inside the scope that owns a
// name, no read of it is a read of the global.
DGPP_TEST(chat_template_scope_ownership_decides_the_global_reads) {
  const auto reads = [](const std::string& source) { return ChatTemplate::compile(source).reads("label"); };
  if (reads("{% generation %}{% generation %}[{{ label }}]{% endgeneration %}{% set label = 'local' %}"
            "{% endgeneration %}"))
    throw std::runtime_error("the outer block owns the name the inner block reads");
  if (reads("{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}"))
    throw std::runtime_error("the top level owns the name the loop reads");
  if (reads("{% set cap %}[{{ label }}]{% endset %}{% set label = 'local' %}{{ cap }}"))
    throw std::runtime_error("the top level owns the name the block set reads");
  if (reads("{% macro m() %}[{{ label }}]{% endmacro %}{{ m() }}{% set label = 'local' %}"))
    throw std::runtime_error("the top level owns the name the macro reads");
  if (!reads("{% generation %}{% generation %}[{{ label }}]{% endgeneration %}{% if true %}{% set label = 'local' %}"
             "{% endif %}{% endgeneration %}"))
    throw std::runtime_error("an assignment under an if leaves the global readable");
  if (!reads("{% for a in xs %}{% for b in xs %}[{{ label }}]{% endfor %}{% set label = 'local' %}{% endfor %}"
             "{{ label }}"))
    throw std::runtime_error("a read at the top level is a read of the global");
  if (!reads("{% generation %}{{ label }}{% set label = 'local' %}{% endgeneration %}"))
    throw std::runtime_error("a read before the scope's own assignment is a read of the global");
  if (!reads("{% macro m(p=label) %}[{{ p }}]{% endmacro %}{{ m() }}"))
    throw std::runtime_error("a default that reads the global is a read of it");
  if (reads("{% macro m(a=label, label='x') %}[{{ a }}]{% endmacro %}{{ m() }}"))
    throw std::runtime_error("a default that reads a later parameter does not read the global");
}

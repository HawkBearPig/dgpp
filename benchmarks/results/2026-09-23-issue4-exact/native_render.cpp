#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: native_render CHECKPOINT GLOBALS_JSON OUTPUT_PREFIX");
    std::ifstream in(argv[2]);
    if (!in) throw std::runtime_error("cannot open globals");
    const std::string input((std::istreambuf_iterator<char>(in)), {});
    const auto parsed = dgpp::minijson::parse(input);
    auto chat = dgpp::text::ChatTemplate::load(std::string(argv[1]) + "/chat_template.jinja");
    auto tokenizer = dgpp::text::Tokenizer::load(std::string(argv[1]) + "/tokenizer.json");
    const auto rendered = chat.render(dgpp::text::Value::from_minijson(parsed.root));
    const auto ids = tokenizer.encode(rendered);
    std::ofstream text(std::string(argv[3]) + ".rendered.txt");
    text << rendered;
    std::ofstream tokens(std::string(argv[3]) + ".ids.json");
    tokens << '[';
    for (size_t i = 0; i < ids.size(); ++i) tokens << (i ? "," : "") << ids[i];
    tokens << "]\n";
    if (!text || !tokens) throw std::runtime_error("cannot write output");
    std::cout << ids.size() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

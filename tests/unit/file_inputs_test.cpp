#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "common/test.hpp"
#include "serve/file_inputs.hpp"
#include "text/json_grammar.hpp"

namespace {
using namespace dgpp::serve;
using V = dgpp::minijson::Value;
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
std::string encode64(std::string_view data) {
  constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < data.size(); i += 3) {
    unsigned x = static_cast<unsigned char>(data[i]) << 16;
    if (i+1 < data.size()) x |= static_cast<unsigned char>(data[i+1]) << 8;
    if (i+2 < data.size()) x |= static_cast<unsigned char>(data[i+2]);
    out += alphabet[x >> 18]; out += alphabet[(x >> 12) & 63];
    out += i+1 < data.size() ? alphabet[(x >> 6) & 63] : '=';
    out += i+2 < data.size() ? alphabet[x & 63] : '=';
  }
  return out;
}
HttpRequest chat_file(const std::string& name, const std::string& data) {
  HttpRequest req;
  req.body = dgpp::text::json_text_of(V::make_object({{"messages", V::make_array({V::make_object({
    {"role", V::make_string("user")}, {"content", V::make_array({V::make_object({
      {"type", V::make_string("file")}, {"file", V::make_object({{"filename", V::make_owned_string(name)},
        {"file_data", V::make_owned_string(encode64(data))}})}})})}})})}}));
  return req;
}
std::string pdf(bool text) {
  std::string out = "%PDF-1.4\n";
  std::string stream = text ? "BT /F1 12 Tf 72 720 Td (PDF input works) Tj ET\n" : "";
  const std::vector<std::string> objects = {
    "<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    "<< /Length " + std::to_string(stream.size()) + " >>\nstream\n" + stream + "endstream"};
  std::vector<size_t> offsets{0};
  for (size_t i = 0; i < objects.size(); ++i) {
    offsets.push_back(out.size());
    out += std::to_string(i+1) + " 0 obj\n" + objects[i] + "\nendobj\n";
  }
  size_t xref = out.size();
  out += "xref\n0 6\n0000000000 65535 f \n";
  for (size_t i = 1; i < offsets.size(); ++i) {
    auto n = std::to_string(offsets[i]);
    out += std::string(10-n.size(), '0') + n + " 00000 n \n";
  }
  return out + "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
}
}

DGPP_TEST(file_inputs_pdfAndTextExtractionLimitsAndErrors) {
  FileInputs files({});
  auto text = files.prepare(chat_file("notes.md", "line one\n日本語"));
  check(text.body.find("日本語") != std::string::npos, "UTF-8 document text preserved");
  if (::access("/usr/bin/pdftotext", X_OK) == 0) {
    auto doc = files.prepare(chat_file("report.pdf", pdf(true)));
    check(doc.body.find("PDF input works") != std::string::npos, "real PDF text extraction");
    bool empty = false;
    try { files.prepare(chat_file("scan.pdf", pdf(false))); }
    catch (const FileInputError& e) { empty = e.code == "pdf_text_unavailable"; }
    check(empty, "PDF with no text needs OCR/vision");
  }
  for (const auto& [name, data] : std::vector<std::pair<std::string,std::string>>{
      {"bad.txt", std::string("a\0b",3)}, {"bad.txt", std::string("\xff",1)}, {"image.png", "PNG"}}) {
    bool failed = false;
    try { files.prepare(chat_file(name, data)); } catch (const FileInputError&) { failed = true; }
    check(failed, "reject invalid text or unsupported binary input");
  }
  FileInputConfig config;
  config.max_file_bytes = 4;
  FileInputs limited(config);
  bool large = false;
  try { limited.prepare(chat_file("x.txt", "12345")); } catch (const FileInputError& e) { large = e.status == 413; }
  check(large, "decoded file limit enforced");
  config.max_file_bytes = 50 * 1024 * 1024;
  config.pdf_command = "/no/such/dgpp-pdf-extractor";
  FileInputs missing(config);
  bool unavailable = false;
  try { missing.prepare(chat_file("x.pdf", pdf(true))); }
  catch (const FileInputError& e) { unavailable = e.code == "pdf_extractor_unavailable"; }
  check(unavailable, "missing PDF executable reports actionable error");
}

DGPP_TEST(file_inputs_persistentStoreSurvivesRestart) {
  std::string directory = "/tmp/dgpp-file-test-XXXXXX";
  check(mkdtemp(directory.data()) != nullptr, "temporary test directory");
  struct Cleanup { std::string path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
  FileInputConfig config; config.directory = directory;
  std::string id;
  {
    FileInputs files(config);
    HttpRequest upload;
    upload.method = "POST"; upload.path = "/v1/files";
    upload.headers = {{"content-type", "multipart/form-data; boundary=test"}};
    upload.body = "--test\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n--test\r\ncontent-disposition: form-data; filename=\"a;\\\"report.txt\"; name=\"file\"\r\n\r\nhello\r\n--test--\r\n";
    const auto response = files.route(upload);
    id = dgpp::minijson::parse(response.body).root.at("id").as_string();
    check(dgpp::minijson::parse(response.body).root.at("filename").as_string() == "a;\"report.txt",
          "multipart parameter boundaries, order and quoted escapes");
  }
  FileInputs restarted(config);
  HttpRequest get; get.method = "GET"; get.path = "/v1/files/" + id + "/content";
  check(restarted.route(get).body == "hello", "persisted file content available after restart");
  get.path = "/v1/files/" + id; get.method = "DELETE";
  check(restarted.route(get).body.find("true") != std::string::npos, "persisted file deletion");
}

namespace {
HttpRequest upload_file(const std::string& data, const std::string& name = "report.txt") {
  HttpRequest req;
  req.method = "POST"; req.path = "/v1/files";
  req.headers = {{"content-type", "multipart/form-data; boundary=BOUND"}};
  req.body = "--BOUND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n--BOUND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"" + name + "\"\r\n\r\n" + data + "\r\n--BOUND--\r\n";
  return req;
}
template<class F> void file_error(F operation, int status, std::string_view param = {}) {
  try { operation(); }
  catch (const FileInputError& e) {
    check(e.status == status && (param.empty() || e.param == param),
          "wrong file error: " + std::to_string(e.status) + " " + e.param + " " + e.what());
    return;
  }
  throw std::runtime_error("expected file error " + std::to_string(status));
}
}

DGPP_TEST(file_inputs_edge_base64ValidationAndUnicodeDetection) {
  FileInputs files({});
  for (const auto& encoded : {"A", "AAA", "====", "=AAA", "A=AA", "AA=A", "AAAA====", "YWJj\n", "Zh==", "Zm9=", "!AAA", "data:text/plain,hello", "data:;base64"}) {
    auto req = chat_file("x.txt", "x");
    const auto at = req.body.find("eA==");
    req.body.replace(at, 4, dgpp::text::json_text_of(V::make_string(encoded)).substr(1,
        dgpp::text::json_text_of(V::make_string(encoded)).size()-2));
    file_error([&] { files.prepare(req); }, 400, "messages[0].content[0].file.file_data");
  }
  for (const auto& text : {"f", "fo", "foo", "日本語", "😀"}) {
    auto req = chat_file("x.txt", text);
    const auto encoded = encode64(text);
    req.body.replace(req.body.find(encoded), encoded.size(), "data:text/plain;base64," + encoded);
    check(files.prepare(req).body.find(text) != std::string::npos, "base64 padding lengths and data URI preserve UTF-8");
  }
  auto escaped = chat_file("x.txt", "x");
  for (size_t at = 0; (at = escaped.body.find("file", at)) != std::string::npos; at += 9)
    escaped.body.replace(at, 4, "\\u0066ile");
  check(FileInputs::needed(escaped), "escaped JSON field names and file type still trigger preprocessing");
  check(files.prepare(escaped).body.find("[Document:") != std::string::npos, "escaped file fields normalize");
}

DGPP_TEST(file_inputs_require_complete_body_before_normalization) {
  FileInputs files({});
  for (const std::string& suffix :
       {std::string("junk"), std::string("{}"), std::string(" \n[]"), std::string(1, '\0')}) {
    auto req = chat_file("input.txt", "hello 1e+");
    req.body += suffix;
    check(!FileInputs::needed(req), "malformed bodies reach the API body validator");
    file_error([&] { files.prepare(req); }, 400, "body");
  }
  auto padded = chat_file("input.txt", "hello 1e+");
  padded.body += " \t\r\n";
  check(FileInputs::needed(padded), "JSON whitespace still permits file detection");
  check(files.prepare(padded).body.find("hello 1e+") != std::string::npos,
        "document bytes remain literal text");
}

DGPP_TEST(file_inputs_edge_multipartPreservesBoundaryPrefixesAndRejectsMalformedHeaders) {
  FileInputs files({});
  const std::string content = "line one\r\n--BOUND-not-a-delimiter\r\n--BOUND--also-not-a-delimiter\r\nline two";
  const auto uploaded = files.route(upload_file(content));
  const std::string id(dgpp::minijson::parse(uploaded.body).root.at("id").as_string());
  HttpRequest get; get.method = "GET"; get.path = "/v1/files/" + id + "/content";
  check(files.route(get).body == content, "boundary-looking bytes inside a file are preserved");
  auto mixed_case = upload_file("case-sensitive content");
  mixed_case.headers[0].second = "Multipart/Form-Data; BOUNDARY=\"BOUND\"";
  constexpr std::string_view file_disposition = "form-data; name=\"file\"; filename=";
  mixed_case.body.replace(mixed_case.body.find(file_disposition), file_disposition.size(),
                          "Form-Data; NAME=\"file\"; FILENAME=");
  check(files.route(mixed_case).status == 200, "MIME parameter names are case-insensitive");
  std::vector<HttpRequest> malformed;
  auto bad = upload_file("x"); bad.headers[0].second = "multipart/form-data-invalid; boundary=BOUND"; malformed.push_back(bad);
  bad = upload_file("x"); bad.body.resize(bad.body.size()-5); malformed.push_back(bad);
  bad = upload_file("x"); bad.body.replace(bad.body.find("form-data; name=\"file\""), 9, "attachment"); malformed.push_back(bad);
  bad = upload_file("x"); bad.body.replace(bad.body.find("name=\"file\""), 11, "name=\"file\"; name=\"purpose\""); malformed.push_back(bad);
  for (const auto& req : malformed) file_error([&] { files.route(req); }, 400);
}

DGPP_TEST(file_inputs_edge_limitsAreInclusiveCombinedAndReclaimed) {
  FileInputConfig config; config.max_file_bytes = 5; config.max_request_bytes = 9;
  config.max_storage_bytes = 10; config.max_text_bytes = 8;
  FileInputs files(config);
  const auto first = files.route(upload_file("12345"));
  files.route(upload_file("67890"));
  file_error([&] { files.route(upload_file("x")); }, 413);
  file_error([&] { files.route(upload_file("123456")); }, 413);
  HttpRequest del; del.method = "DELETE";
  del.path = "/v1/files/" + std::string(dgpp::minijson::parse(first.body).root.at("id").as_string());
  files.route(del);
  check(files.route(upload_file("abcde")).status == 200, "deletion reclaims storage exactly");
  file_error([&] { files.route(del); }, 404);
  auto combined = chat_file("x.txt", "12345");
  const auto parsed = dgpp::minijson::parse(combined.body);
  const auto part = parsed.root.at("messages").items()[0].at("content").items()[0];
  const auto join = [&](const V& second) {
    HttpRequest req;
    req.body = dgpp::text::json_text_of(V::make_object({{"messages", V::make_array({V::make_object({
      {"role", V::make_string("user")}, {"content", V::make_array({part, second})}})})}}));
    return req;
  };
  file_error([&] { files.prepare(join(part)); }, 413);
  auto smaller = chat_file("y.txt", "6789");
  const auto small = dgpp::minijson::parse(smaller.body);
  file_error([&] { files.prepare(join(small.root.at("messages").items()[0].at("content").items()[0])); }, 413);
  config.max_text_bytes = 9;
  FileInputs exact(config);
  check(exact.prepare(join(small.root.at("messages").items()[0].at("content").items()[0])).body.find("6789") != std::string::npos,
        "combined limits accept equality");
}

DGPP_TEST(file_inputs_edge_invalidShapesAndCorruptMetadata) {
  FileInputs files({});
  for (const auto* file : {"null", "[]", "{}", R"({"file_data":3})", R"({"file_id":"../etc/passwd"})",
                          R"({"filename":null,"file_data":"eA=="})", R"({"file_id":"file-x","file_data":"eA=="})"}) {
    HttpRequest req; req.body = std::string(R"({"messages":[{"role":"user","content":[{"type":"file","file":)") + file + "}]}]}";
    file_error([&] { files.prepare(req); }, std::string_view(file).find("../etc/passwd") != std::string_view::npos ? 404 : 400);
  }
  std::string directory = "/tmp/dgpp-file-corrupt-XXXXXX";
  check(mkdtemp(directory.data()) != nullptr, "corrupt metadata test directory");
  struct Cleanup { std::string path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
  const std::string id = "file-0123456789abcdef0123456789abcdef";
  { std::ofstream out(directory + "/" + id); out << "x"; }
  for (const auto* metadata : {R"({"bytes":1})", R"({"bytes":1e100})", R"({"bytes":-1})", "[]"}) {
    { std::ofstream out(directory + "/" + id + ".json"); out << metadata; }
    FileInputConfig config; config.directory = directory;
    bool rejected = false;
    try { FileInputs broken(config); } catch (const std::exception&) { rejected = true; }
    check(rejected, "corrupt persistent metadata must fail before serving requests");
  }
}

DGPP_TEST(file_inputs_edge_concurrentUploadsRespectQuota) {
  FileInputConfig config; config.max_storage_bytes = 25;
  FileInputs files(config);
  std::vector<std::future<bool>> jobs;
  for (int i = 0; i < 12; ++i) jobs.push_back(std::async(std::launch::async, [&] {
    try { files.route(upload_file("12345")); return true; }
    catch (const FileInputError& e) { check(e.status == 413, "quota refusal"); return false; }
  }));
  int successes = 0;
  for (auto& job : jobs) successes += job.get();
  check(successes == 5, "concurrent requests cannot exceed shared file storage quota");
}

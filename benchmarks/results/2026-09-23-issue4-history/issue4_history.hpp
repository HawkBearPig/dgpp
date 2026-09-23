#pragma once
// Temporary exact-fixture instrumentation. All device accesses are read-only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"

namespace dgpp::issue4h {
inline bool active = false;
inline int rank = 0, layer = 0, rows_count = 0;
inline int64_t position = 0;
inline std::filesystem::path directory;
inline std::map<int, int64_t> next_position;
inline std::map<std::string, std::vector<unsigned char>> states;
inline std::map<std::string, std::vector<uint16_t>> caches;
inline std::string key(const std::string& name) { return std::to_string(layer) + "/" + name; }
inline void require(bool good, const std::string& what) {
  if (!good) throw std::runtime_error("issue4 history: layer " + std::to_string(layer) +
      " pos " + std::to_string(position) + " " + what);
}
inline void report(const std::string& name, size_t count) {
  std::ofstream f(directory / "checks.jsonl", std::ios::app);
  f << "{\"position\":" << position << ",\"rows\":" << rows_count <<
      ",\"check\":\"" << name << "\",\"elements\":" << count << "}\n";
  require(bool(f), "write checks");
}
inline void context(int r, int l, int64_t p, int n, bool decode, bool capture) {
  active = !decode && !capture && n > 128;
  if (!active) return;
  rank = r; layer = l; position = p; rows_count = n;
  require(p == next_position[l], "noncontiguous chunk");
  require((p < 260096 && n == 2048) || (p == 260096 && n == 1194), "unexpected fixture shape");
  next_position[l] = p + n;
  directory = std::filesystem::path("/tmp/dgpp-issue4-history-capture") /
      ("rank" + std::to_string(r)) / ("layer" + std::to_string(l));
  if (p == 0) require(!std::filesystem::exists(directory), "capture already exists");
  std::filesystem::create_directories(directory);
  report("chunk", n);
}
template<class T> std::vector<T> read(const T* ptr, size_t n, cudaStream_t stream) {
  std::vector<T> out(n);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  return out;
}
template<class T> void append(const std::string& name, const std::vector<T>& v) {
  std::ofstream f(directory / (name + ".bin"), std::ios::binary | std::ios::app);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  require(bool(f), "append " + name);
}
template<class T> void rows(const std::string& name, const std::vector<T>& v, size_t width) {
  const auto p = directory / (name + ".bin");
  require(v.size() == static_cast<size_t>(rows_count) * width, "row geometry " + name);
  require((std::filesystem::exists(p) ? std::filesystem::file_size(p) : 0) ==
      static_cast<size_t>(position) * width * sizeof(T), "stream offset " + name);
  append(name, v);
}
template<class T> void dump_rows(const std::string& name, const T* ptr, size_t width, cudaStream_t s) {
  rows(name, read(ptr, static_cast<size_t>(rows_count) * width, s), width);
}
template<class T> void once(const std::string& name, const T* ptr, size_t count, cudaStream_t s) {
  if (position == 0) append(name, read(ptr, count, s));
}
template<class T> void before(const std::string& name, const T* ptr, size_t count, cudaStream_t s) {
  auto v = read(ptr, count, s);
  auto& prior = states[key(name)];
  if (position == 0) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(v.data());
    require(std::all_of(bytes, bytes + v.size() * sizeof(T), [](auto x){ return x == 0; }), name + " not initialized to zero");
  } else {
    require(prior.size() == v.size() * sizeof(T), name + " state shape");
    require(std::memcmp(prior.data(), v.data(), prior.size()) == 0, name + " state handoff");
  }
  report(name + "_before_exact", count);
}
template<class T> std::vector<T> after(const std::string& name, const T* ptr, size_t count, cudaStream_t s) {
  auto v = read(ptr, count, s);
  auto& saved = states[key(name)];
  saved.resize(v.size() * sizeof(T));
  std::memcpy(saved.data(), v.data(), saved.size());
  return v;
}
inline std::vector<uint16_t> select(const std::vector<uint16_t>& v, int stride,
    const std::vector<std::pair<int,int>>& spans) {
  int width = 0; for (auto [begin,n] : spans) width += n;
  std::vector<uint16_t> out(static_cast<size_t>(rows_count) * width);
  for (int t = 0; t < rows_count; ++t) {
    int dst = 0;
    for (auto [begin,n] : spans) {
      std::copy_n(v.data() + static_cast<size_t>(t) * stride + begin, n,
          out.data() + static_cast<size_t>(t) * width + dst);
      dst += n;
    }
  }
  return out;
}
inline void gdn(const uint16_t* raw, const uint16_t* convolved, const uint16_t* a,
    const uint16_t* b, const uint16_t* core, const uint16_t* cw, const float* al,
    const float* dt, float* state, uint16_t* convstate, int lk, int lv, int k, int v,
    int width, float scale, cudaStream_t stream) {
  const int h = layer % lv, hk = h / (lv / lk), C = 2 * lk * k + lv * v;
  const std::vector<std::pair<int,int>> spans{{hk*k,k},{lk*k+hk*k,k},{2*lk*k+h*v,v}};
  auto raw_host = read(raw, static_cast<size_t>(rows_count) * C, stream);
  rows("sample_raw", select(raw_host, C, spans), 2*k+v);
  rows("sample_qkv", select(read(convolved, static_cast<size_t>(rows_count)*C,stream),C,spans),2*k+v);
  rows("sample_a",select(read(a,static_cast<size_t>(rows_count)*lv,stream),lv,{{h,1}}),1);
  rows("sample_b",select(read(b,static_cast<size_t>(rows_count)*lv,stream),lv,{{h,1}}),1);
  rows("sample_core",select(read(core,static_cast<size_t>(rows_count)*lv*v,stream),lv*v,{{h*v,v}}),v);
  auto s = after("recurrent", state, static_cast<size_t>(lv)*v*k, stream);
  append("sample_states", std::vector<float>(s.begin()+h*v*k,s.begin()+(h+1)*v*k));
  auto cs = after("conv",convstate,static_cast<size_t>(C)*(width-1),stream);
  for (int c=0;c<C;++c) for (int j=0;j<width-1;++j)
    require(cs[c*(width-1)+j] == raw_host[static_cast<size_t>(rows_count-width+1+j)*C+c],"conv tail placement");
  report("conv_tail_exact",cs.size());
  if (position == 0) {
    std::ofstream f(directory / "sample.meta");
    f << h << " " << k << " " << v << " " << width << "\n";
    append("sample_scale",std::vector<float>{scale});
    once("sample_alog",al+h,1,stream); once("sample_dt",dt+h,1,stream);
    auto weights=read(cw,static_cast<size_t>(C)*width,stream);
    std::vector<uint16_t> chosen;
    for(auto [start,n]:spans) chosen.insert(chosen.end(),weights.begin()+start*width,weights.begin()+(start+n)*width);
    append("sample_conv",chosen);
  }
}
inline void cache_check(const std::string& name, const uint16_t* ptr, size_t slots,
    const uint16_t* incoming, int width, int block_tokens, const std::vector<int32_t>& table,
    cudaStream_t stream) {
  auto& expected = caches[key(name)];
  require(expected.size()==static_cast<size_t>(position)*width,"cache expected length");
  auto tail=read(incoming,static_cast<size_t>(rows_count)*width,stream);
  expected.insert(expected.end(),tail.begin(),tail.end());
  auto actual=read(ptr,slots*width,stream);
  for(int64_t pos=0;pos<position+rows_count;pos+=block_tokens) {
    const int n=std::min<int64_t>(block_tokens,position+rows_count-pos);
    const int64_t physical=static_cast<int64_t>(table[pos/block_tokens])*block_tokens;
    require(physical>=0 && physical+n<=static_cast<int64_t>(slots),"cache block bounds");
    require(std::memcmp(actual.data()+physical*width,expected.data()+pos*width,
        static_cast<size_t>(n)*width*2)==0,name+" cache contents");
  }
  report(name+"_all_visible_exact",expected.size());
}
inline void index_check(const uint16_t* ptr, size_t slots, const uint16_t* raw,
    int raw_width, int dim, int raw_offset, int pools_per_block, int ratio,
    const std::vector<int32_t>& table, cudaStream_t stream) {
  const int64_t old_pools=position/ratio, pools=(position+rows_count)/ratio;
  auto& expected=caches[key("index")];
  require(expected.size()==static_cast<size_t>(old_pools)*dim,"index expected length");
  auto actual=read(ptr,slots*dim,stream);
  for(int64_t i=0;i<old_pools;++i) {
    const int64_t phys=static_cast<int64_t>(table[i/pools_per_block])*pools_per_block+i%pools_per_block;
    require(std::memcmp(actual.data()+phys*dim,expected.data()+i*dim,dim*2)==0,"old index cache modified");
  }
  for(int64_t i=old_pools;i<pools;++i) {
    const int64_t phys=static_cast<int64_t>(table[i/pools_per_block])*pools_per_block+i%pools_per_block;
    expected.insert(expected.end(),actual.begin()+phys*dim,actual.begin()+(phys+1)*dim);
  }
  auto source=read(raw,static_cast<size_t>(rows_count)*raw_width,stream);
  rows("index_raw",select(source,raw_width,{{raw_offset,dim}}),dim);
  // Only newly completed pools are appended; the final two raw rows stay a tail.
  append("index_created",std::vector<uint16_t>(expected.begin()+old_pools*dim,expected.end()));
  report("old_index_exact",old_pools*dim);
}
}

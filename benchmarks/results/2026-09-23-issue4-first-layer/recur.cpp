// Independent all-head first-layer recurrence from token-indexed projections.
// No DGPP kernels, loader, or model reference functions are called.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs=std::filesystem;
float bf(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits)<<16); }
uint16_t rb(double value) {
  uint32_t bits=std::bit_cast<uint32_t>(static_cast<float>(value));
  return static_cast<uint16_t>((bits+0x7fff+((bits>>16)&1))>>16);
}
template<class T> std::vector<T> read(const fs::path& path) {
  const size_t bytes=fs::file_size(path);
  if(bytes%sizeof(T))throw std::runtime_error("Bad file size");
  std::vector<T> values(bytes/sizeof(T));
  std::ifstream stream(path,std::ios::binary);
  stream.read(reinterpret_cast<char*>(values.data()),bytes);
  if(!stream)throw std::runtime_error("Read failed: "+path.string());
  return values;
}
template<class T> void write(const fs::path& path,const std::vector<T>& values) {
  std::ofstream stream(path,std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
  if(!stream)throw std::runtime_error("Write failed");
}

int main(int argc,char** argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: recur INPUT OUTPUT");
  const fs::path input(argv[1]),output(argv[2]);
  if(fs::exists(output))throw std::runtime_error("Refusing existing output");
  fs::create_directories(output);
  const auto tokens=read<int32_t>(input/"token_rows.bin");
  const int T=static_cast<int>(tokens.size()),start=std::max(T-1194,0),tail=T-start;
  if(T<1)throw std::runtime_error("Empty input");
  const auto scalars=read<float>(input/"scalars.bin");
  if(scalars.size()!=2)throw std::runtime_error("Wrong scalars");
  constexpr int H=24,K=128,V=128,C=5120;
  for(int rank=0;rank<2;++rank) {
   const std::string suffix=std::to_string(rank)+".bin";
   const auto lookup=read<uint16_t>(input/("qkv"+suffix));
   const auto a=read<uint16_t>(input/("a"+suffix)),b=read<uint16_t>(input/("b"+suffix));
   const auto weights=read<uint16_t>(input/("conv_weight"+suffix));
   const auto alog=read<float>(input/("a_log"+suffix)),dt=read<float>(input/("dt_bias"+suffix));
   const size_t unique=lookup.size()/C;
   if(lookup.size()!=unique*C || a.size()!=unique*H || b.size()!=unique*H || weights.size()!=C*4 || alog.size()!=H || dt.size()!=H)
    throw std::runtime_error("Wrong operand geometry");
   for(int index:tokens)if(index<0 || size_t(index)>=unique)throw std::runtime_error("Invalid lookup index");
   std::vector<double> before(H*V*K),after(H*V*K);
   std::vector<uint16_t> conv_before(C*3),conv_after(C*3),convolved(size_t(tail)*C),core(size_t(tail)*H*V);
   #pragma omp parallel for schedule(static)
   for(int head=0;head<H;++head) {
    const int kh=head/3;
    std::vector<double> state(V*K,0.0);
    std::array<double,3*K*3> history{};
    std::array<double,3*K> qkv{};
    std::array<int,3*K> channels{};
    for(int d=0;d<K;++d) {channels[d]=kh*K+d;channels[K+d]=8*K+kh*K+d;channels[2*K+d]=16*K+head*V+d;}
    const double decay_scale=std::exp(double(alog[head]));
    for(int t=0;t<T;++t) {
     if(t==start) {
      std::copy(state.begin(),state.end(),before.begin()+head*V*K);
      for(int c=0;c<3*K;++c)if(c>=2*K || head%3==0)
       for(int j=0;j<3;++j)conv_before[channels[c]*3+j]=rb(history[c*3+j]);
     }
     const size_t index=static_cast<size_t>(tokens[t]);
     for(int c=0;c<3*K;++c) {
      const int channel=channels[c];
      const double raw=bf(lookup[index*C+channel]);
      double value=raw*bf(weights[channel*4+3]);
      for(int j=0;j<3;++j)value+=history[c*3+j]*bf(weights[channel*4+j]);
      qkv[c]=bf(rb(value/(1+std::exp(-value))));
      history[c*3]=history[c*3+1];history[c*3+1]=history[c*3+2];history[c*3+2]=raw;
      if(t>=start && (c>=2*K || head%3==0))convolved[size_t(t-start)*C+channel]=rb(qkv[c]);
     }
     double qs=1e-6,ks=1e-6;
     for(int d=0;d<K;++d){qs+=qkv[d]*qkv[d];ks+=qkv[K+d]*qkv[K+d];}
     const double qscale=double(scalars[1])/std::sqrt(qs),kscale=1/std::sqrt(ks);
     for(int d=0;d<K;++d){qkv[d]*=qscale;qkv[K+d]*=kscale;}
     const double av=double(bf(a[index*H+head]))+double(dt[head]);
     const double decay=std::exp(-decay_scale*(av>20?av:std::log1p(std::exp(av))));
     const double beta=1/(1+std::exp(-double(bf(b[index*H+head]))));
     for(int v=0;v<V;++v) {
      double* row=state.data()+v*K;
      double dot=0;
      #pragma omp simd reduction(+:dot)
      for(int d=0;d<K;++d){row[d]*=decay;dot+=row[d]*qkv[K+d];}
      const double delta=(qkv[2*K+v]-dot)*beta;
      double value=0;
      #pragma omp simd reduction(+:value)
      for(int d=0;d<K;++d){row[d]+=delta*qkv[K+d];value+=row[d]*qkv[d];}
      if(t>=start)core[(size_t(t-start)*H+head)*V+v]=rb(value);
     }
    }
    std::copy(state.begin(),state.end(),after.begin()+head*V*K);
    for(int c=0;c<3*K;++c)if(c>=2*K || head%3==0)
     for(int j=0;j<3;++j)conv_after[channels[c]*3+j]=rb(history[c*3+j]);
   }
   write(output/("state_before"+suffix),before);write(output/("state_after"+suffix),after);
   write(output/("conv_before"+suffix),conv_before);write(output/("conv_after"+suffix),conv_after);
   write(output/("qkvc"+suffix),convolved);write(output/("core"+suffix),core);
   std::cout<<"Replayed rank "<<rank<<", all 24 heads, "<<T<<" tokens from zero state\n"<<std::flush;
  }
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

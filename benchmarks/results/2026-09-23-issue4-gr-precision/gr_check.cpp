// Independent FP64 oracle for the fused vLLM HC rounding policy.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qwen_gr.hpp"
using namespace dgpp;
struct Device {
  void* p;
  Device(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p,n)); }
  ~Device(){cudaFree(p);}
  void upload(const void* x,size_t n){DGPP_CUDA_OK(cudaMemcpy(p,x,n,cudaMemcpyHostToDevice));}
  void download(void* x,size_t n){DGPP_CUDA_OK(cudaMemcpy(x,p,n,cudaMemcpyDeviceToHost));}
};
struct Error {
  long double ss=0, rr=0; double maximum=0; size_t n=0;
  void add(double got,double ref){double d=got-ref; ss+=d*d;rr+=ref*ref;maximum=std::max(maximum,std::abs(d));++n;}
  double l2()const{return std::sqrt(double(ss/std::max(rr,1e-30L)));}
};
double f(uint16_t x){return bf16_bits_to_float(x);}
uint16_t b(double x){return float_to_bf16_bits(static_cast<float>(x));}
double sig(double x){return 1/(1+std::exp(-x));}
int main(){
  std::mt19937 rng(4);std::normal_distribution<float> normal;
  constexpr int H=2560,HC=4,W=HC*H;
  Error mix,combine,gates_error, fused_gates_error;
  for(int T:{1,4,8,1194}) {
    size_t nw=size_t(T)*W, nh=size_t(T)*H;
    std::vector<uint16_t> rn(nw),logit(nw),r(nw),y(nh),x(nh),wi(HC*W,0);
    for(auto&v:rn)v=b(normal(rng));for(auto&v:logit)v=b(normal(rng)*3);
    for(auto&v:r)v=b(normal(rng)*2);for(auto&v:y)v=b(normal(rng));
    for(int j=0;j<HC;++j)wi[j*W+j]=b(1.25*(j+1));
    Device drn(nw*2),dl(nw*2),dr(nw*2),dy(nh*2),dx(nh*2),dw(wi.size()*2),dg(T*HC*4);
    drn.upload(rn.data(),nw*2);dl.upload(logit.data(),nw*2);dr.upload(r.data(),nw*2);dy.upload(y.data(),nh*2);dw.upload(wi.data(),wi.size()*2);
    qwen_gr_mix_finish_bf16(dl.p,drn.p,dx.p,T,HC,H,nullptr);dx.download(x.data(),nh*2);
    for(int t=0;t<T;++t)for(int j=0;j<H;++j){double z=0;for(int i=0;i<HC;++i){size_t k=size_t(t)*W+i*H+j;z+=sig(f(logit[k]))*f(rn[k]);}mix.add(f(x[size_t(t)*H+j]),f(b(z/HC)));}
    qwen_gr_combine_bf16(dr.p,drn.p,dw.p,dy.p,(float*)dg.p,T,HC,H,nullptr);
    std::vector<uint16_t> out(nw);std::vector<float> gates(T*HC);dr.download(out.data(),nw*2);dg.download(gates.data(),gates.size()*4);
    for(int t=0;t<T;++t)for(int i=0;i<HC;++i){
      double dot=f(b(f(rn[size_t(t)*W+i])*f(wi[i*W+i])));double gate=2*sig(dot/HC);
      gates_error.add(gates[t*HC+i],gate);
      for(int j=0;j<H;++j){size_t k=size_t(t)*W+i*H+j;combine.add(f(out[k]),f(b(f(r[k])+f(y[size_t(t)*H+j])*gate)));}
    }
    if(T<=8){
      constexpr int L=320;std::vector<uint16_t> zero(size_t(L)*W,0),norm(W,0),normalized(nw);
      Device dd(zero.size()*2),dt(size_t(T)*L*2),dn(W*2);dd.upload(zero.data(),zero.size()*2);dn.upload(norm.data(),W*2);
      qwen_gr_down_inject_bf16(drn.p,dd.p,dt.p,L,dw.p,(float*)dg.p,HC,H,T,nullptr);
      dg.download(gates.data(),gates.size()*4);
      for(int t=0;t<T;++t)for(int i=0;i<HC;++i)fused_gates_error.add(gates[t*HC+i],2*sig(f(b(f(rn[size_t(t)*W+i])*f(wi[i*W+i])))/HC));
      dr.upload(r.data(),nw*2);
      qwen_gr_norm_down_bf16(dr.p,W,dn.p,HC,H,1e-6f,drn.p,dd.p,dt.p,L,T,nullptr,dw.p,(float*)dg.p);
      dg.download(gates.data(),gates.size()*4);drn.download(normalized.data(),nw*2);
      for(int t=0;t<T;++t)for(int i=0;i<HC;++i)fused_gates_error.add(gates[t*HC+i],2*sig(f(b(f(normalized[size_t(t)*W+i])*f(wi[i*W+i])))/HC));
    }
  }
  bool pass=mix.l2()<1e-4 && combine.l2()<1e-4 && gates_error.l2()<1e-6 && fused_gates_error.l2()<1e-6;
  std::cout<<"{\"mix_l2\":"<<mix.l2()<<",\"combine_l2\":"<<combine.l2()<<",\"gates_l2\":"<<gates_error.l2()<<",\"fused_gates_l2\":"<<fused_gates_error.l2()<<",\"scalars\":"<<mix.n+combine.n<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
  return pass?0:1;
}

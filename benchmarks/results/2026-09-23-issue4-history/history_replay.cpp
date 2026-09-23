// Independent CPU replay from zero state; no model/kernel reference functions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "common/dtypes.hpp"
namespace fs=std::filesystem;
float bf(uint16_t x){return dgpp::bf16_bits_to_float(x);}
float rb(float x){return bf(dgpp::float_to_bf16_bits(x));}
template<class T> std::vector<T> load(const fs::path& p){
 const size_t bytes=fs::file_size(p);if(bytes%sizeof(T))throw std::runtime_error("shape "+p.string());
 std::vector<T> v(bytes/sizeof(T));std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),bytes);if(!f)throw std::runtime_error("read "+p.string());return v;
}
template<class T> std::vector<T> take(std::ifstream& f,size_t n){std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw std::runtime_error("short stream");return v;}
struct Stats {
 uint64_t n=0,bad=0,nonfinite=0;double e=0,r=0,g=0,m=0;
 void add(double ref,double got){double d=std::abs(ref-got);++n;nonfinite+=!std::isfinite(ref)||!std::isfinite(got);bad+=d>1e-7&&d>(2./128)*std::max(std::abs(ref),std::abs(got));e+=d*d;r+=ref*ref;g+=got*got;m=std::max(m,d);}
 void merge(const Stats& a){n+=a.n;bad+=a.bad;nonfinite+=a.nonfinite;e+=a.e;r+=a.r;g+=a.g;m=std::max(m,a.m);}
 double l2()const{return std::sqrt(e/std::max({r,g,1e-300}));}
 void print()const{std::cout<<"{\"n\":"<<n<<",\"relative_l2\":"<<l2()<<",\"max_abs\":"<<m<<",\"relative_two_bf16_mismatches\":"<<bad<<",\"nonfinite\":"<<nonfinite<<"}";}
};
void gdn(const fs::path& p){
 int head,K,V,CW;std::ifstream(p/"sample.meta")>>head>>K>>V>>CW;int C=2*K+V;
 auto weights=load<uint16_t>(p/"sample_conv.bin");auto al=load<float>(p/"sample_alog.bin"),dt=load<float>(p/"sample_dt.bin"),scale=load<float>(p/"sample_scale.bin");
 size_t T=fs::file_size(p/"sample_a.bin")/2;if(T!=261290)throw std::runtime_error("incomplete GDN");
 std::ifstream raw(p/"sample_raw.bin",std::ios::binary),conv(p/"sample_qkv.bin",std::ios::binary),af(p/"sample_a.bin",std::ios::binary),bf_(p/"sample_b.bin",std::ios::binary),core(p/"sample_core.bin",std::ios::binary),states(p/"sample_states.bin",std::ios::binary);
 std::vector<double> state(K*V,0),history(C*(CW-1),0),q(K),k(K);
 Stats cs,os,ss;double worst_state=0,worst_output=0;size_t worst_state_at=0,worst_output_at=0;
 for(size_t start=0;start<T;start+=2048){int N=std::min<size_t>(2048,T-start);auto x=take<uint16_t>(raw,N*C),qc=take<uint16_t>(conv,N*C),a=take<uint16_t>(af,N),b=take<uint16_t>(bf_,N),out=take<uint16_t>(core,N*V);Stats chunk_output;
  for(int t=0;t<N;++t){
   for(int c=0;c<C;++c){double value=double(bf(x[t*C+c]))*bf(weights[c*CW+CW-1]);for(int j=0;j<CW-1;++j)value+=history[c*(CW-1)+j]*bf(weights[c*CW+j]);cs.add(rb(value/(1+std::exp(-value))),bf(qc[t*C+c]));for(int j=0;j<CW-2;++j)history[c*(CW-1)+j]=history[c*(CW-1)+j+1];history[c*(CW-1)+CW-2]=bf(x[t*C+c]);}
   double qs=1e-6,ks=1e-6;for(int d=0;d<K;++d){q[d]=bf(qc[t*C+d]);k[d]=bf(qc[t*C+K+d]);qs+=q[d]*q[d];ks+=k[d]*k[d];}double qi=scale[0]/std::sqrt(qs),ki=1/std::sqrt(ks);for(int d=0;d<K;++d){q[d]*=qi;k[d]*=ki;}
   double gate=bf(a[t])+double(dt[0]);double decay=std::exp(-std::exp(double(al[0]))*(gate>20?gate:std::log1p(std::exp(gate))));double beta=1/(1+std::exp(-double(bf(b[t]))));
   for(int v=0;v<V;++v){double* s=state.data()+v*K;double dot=0;for(int d=0;d<K;++d){s[d]*=decay;dot+=s[d]*k[d];}double delta=(bf(qc[t*C+2*K+v])-dot)*beta;double output=0;for(int d=0;d<K;++d){s[d]+=delta*k[d];output+=s[d]*q[d];}chunk_output.add(rb(output),bf(out[t*V+v]));}
  }
  os.merge(chunk_output);if(chunk_output.l2()>worst_output){worst_output=chunk_output.l2();worst_output_at=start;}
  auto actual=take<float>(states,K*V);Stats one;for(int i=0;i<K*V;++i)one.add(state[i],actual[i]);ss.merge(one);if(one.l2()>worst_state){worst_state=one.l2();worst_state_at=start+N;}
 }
 std::cout<<"\"kind\":\"gdn\",\"head\":"<<head<<",\"tokens\":"<<T<<",\"convolution\":";cs.print();std::cout<<",\"recurrence\":";os.print();std::cout<<",\"checkpoint_states\":";ss.print();std::cout<<",\"worst_state_chunk_l2\":"<<worst_state<<",\"worst_state_position\":"<<worst_state_at<<",\"worst_output_chunk_l2\":"<<worst_output<<",\"worst_output_start\":"<<worst_output_at<<"}"<<std::endl;
}
void index(const fs::path& p){
 int D,K,R;std::ifstream(p/"index.meta")>>D>>K>>R;auto raw=load<uint16_t>(p/"index_raw.bin"),out=load<uint16_t>(p/"index_created.bin"),norm=load<uint16_t>(p/"index_norm.bin");auto inv=load<float>(p/"inv_freq.bin"),sc=load<float>(p/"index_scalars.bin");
 if(raw.size()!=size_t(261290)*D || out.size()!=size_t(261290/K)*D)throw std::runtime_error("incomplete index");
 Stats stats;std::vector<float> mean(D),normalized(D);int half=R/2;
 for(size_t i=0;i<out.size()/D;++i){double ss=0;for(int d=0;d<D;++d){float v=0;for(int j=0;j<K;++j)v+=bf(raw[(i*K+j)*D+d]);mean[d]=rb(v/K);ss+=double(mean[d])*mean[d];}float rs=float(1/std::sqrt(ss/D+double(sc[0])));for(int d=0;d<D;++d)normalized[d]=rb(mean[d]*rs*(1+bf(norm[d])));
  for(int d=0;d<D;++d){float y=normalized[d];if(d<R){float angle=float(i*K)*inv[d%half],co=rb(std::cos(angle)*sc[1]),si=rb(std::sin(angle)*sc[1]),rot=d<half?-normalized[d+half]:normalized[d-half];y=rb(rb(normalized[d]*co)+rb(rot*si));}stats.add(y,bf(out[i*D+d]));}
 }
 std::cout<<"\"kind\":\"index\",\"pools\":"<<out.size()/D<<",\"compression_norm_rope\":";stats.print();std::cout<<"}"<<std::endl;
}
void ple(const fs::path& p){
 int heads,begin,local,dim,eos,CW,dilation,hc,H;std::ifstream(p/"ple.meta")>>heads>>begin>>local>>dim>>eos>>CW>>dilation>>hc>>H;int W=hc*H,S=(CW-1)*dilation;
 auto weights=load<uint16_t>(p/"ple_conv_weight.bin");size_t T=fs::file_size(p/"ple_un.bin")/(2*W);if(T!=261290)throw std::runtime_error("incomplete PLE");
 std::ifstream uf(p/"ple_un.bin",std::ios::binary),gf(p/"ple_gv.bin",std::ios::binary),rf(p/"ple_residual_before.bin",std::ios::binary),of(p/"ple_residual_after.bin",std::ios::binary);
 std::vector<uint16_t> history(S*W,0);Stats stats;double worst=0;size_t worst_at=0;
 for(size_t start=0;start<T;start+=2048){int N=std::min<size_t>(2048,T-start);auto un=take<uint16_t>(uf,size_t(N)*W),gv=take<uint16_t>(gf,size_t(N)*W),r=take<uint16_t>(rf,size_t(N)*W),out=take<uint16_t>(of,size_t(N)*W);Stats chunk;
  #pragma omp parallel
  {
   Stats thread;
   #pragma omp for
   for(int t=0;t<N;++t)for(int c=0;c<W;++c){double cv=double(bf(un[t*W+c]))*bf(weights[c*CW+CW-1]);for(int j=0;j<CW-1;++j){int previous=t-S+j*dilation;double input=previous<0?bf(history[(previous+S)*W+c]):bf(un[previous*W+c]);cv+=input*bf(weights[c*CW+j]);}double v=rb(cv),act=rb(v/(1+std::exp(-v))),ple=rb(bf(gv[t*W+c])+act);thread.add(rb(bf(r[t*W+c])+ple),bf(out[t*W+c]));}
   #pragma omp critical
   chunk.merge(thread);
  }
  stats.merge(chunk);if(chunk.l2()>worst){worst=chunk.l2();worst_at=start;}std::copy(un.end()-S*W,un.end(),history.begin());
 }
 std::cout<<"\"kind\":\"ple\",\"tokens\":"<<T<<",\"convolution_residual\":";stats.print();std::cout<<",\"worst_chunk_l2\":"<<worst<<",\"worst_chunk_start\":"<<worst_at<<"}"<<std::endl;
}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("usage: history_replay {gdn,index,ple} LAYER_DIR");fs::path p=argv[2];std::cout<<std::setprecision(12)<<"{\"path\":\""<<p.string()<<"\",";std::string mode=argv[1];if(mode=="gdn")gdn(p);else if(mode=="index")index(p);else if(mode=="ple")ple(p);else throw std::runtime_error("unknown mode");}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

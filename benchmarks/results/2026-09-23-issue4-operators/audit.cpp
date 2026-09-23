// Independent CPU arithmetic on captured production inputs. No DGPP kernels
// or numerical reference functions are called; only IEEE BF16 conversion is shared.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "common/dtypes.hpp"
namespace fs=std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
float bf(uint16_t x){return bf16_bits_to_float(x);}
float rb(float x){return bf(float_to_bf16_bits(x));}
template<class T> std::vector<T> load(fs::path p){
 auto bytes=fs::file_size(p);if(bytes%sizeof(T))throw std::runtime_error("bad size");
 std::vector<T> x(bytes/sizeof(T));std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(x.data()),bytes);if(!f)throw std::runtime_error("read failed");return x;
}
int ordered(uint16_t x){return x&0x8000 ? int(~x&0xffff):int(x|0x8000);}
void stats(const std::string& name,const std::vector<double>& ref,const std::vector<double>& got){
 if(ref.size()!=got.size())throw std::runtime_error("comparison shape");
 double den=0,gotden=0,err=0,maxabs=0;size_t ulps2=0,relative2=0,nonfinite=0;
 for(size_t i=0;i<ref.size();++i){double d=got[i]-ref[i];den+=ref[i]*ref[i];gotden+=got[i]*got[i];relative2+=std::abs(d)>1e-7 && std::abs(d)>(2.0/128)*std::max(std::abs(ref[i]),std::abs(got[i]));err+=d*d;maxabs=std::max(maxabs,std::abs(d));nonfinite+=!std::isfinite(ref[i])||!std::isfinite(got[i]);ulps2+=std::abs(ordered(float_to_bf16_bits(got[i]))-ordered(float_to_bf16_bits(ref[i])))>2;}
 std::cout<<"\""<<name<<"\":{\"n\":"<<ref.size()<<",\"relative_l2\":"<<std::sqrt(err/std::max(den,1e-300))<<",\"symmetric_relative_l2\":"<<std::sqrt(err/std::max(std::max(den,gotden),1e-300))<<",\"relative_two_ulp_mismatches\":"<<relative2<<",\"max_abs\":"<<maxabs<<",\"over_two_bf16_ulps\":"<<ulps2<<",\"nonfinite\":"<<nonfinite<<"},";
}
std::vector<double> cv(const std::vector<uint16_t>& x){std::vector<double> a(x.size());for(size_t i=0;i<x.size();++i)a[i]=bf(x[i]);return a;}
std::vector<double> cv(const std::vector<float>& x){return {x.begin(),x.end()};}
void gdn(fs::path p){
 int T,LK,LV,K,V,CW;double eps,scale;std::ifstream(p/"gdn.meta")>>T>>LK>>LV>>K>>V>>CW>>eps>>scale;
 auto scalar=load<float>(p/"gdn_scalars.bin");eps=scalar[0];scale=scalar[1];int C=2*LK*K+LV*V;
 auto qkv=load<uint16_t>(p/"qkv.bin"),qkvc=load<uint16_t>(p/"qkvc.bin"),cw=load<uint16_t>(p/"conv_weight.bin"),cs=load<uint16_t>(p/"conv_before.bin"),ce=load<uint16_t>(p/"conv_after.bin");
 std::vector<double> conv(qkv.size());
 #pragma omp parallel for
 for(int c=0;c<C;++c){std::vector<double> hist(CW-1);for(int j=0;j<CW-1;++j)hist[j]=bf(cs[c*(CW-1)+j]);
  for(int t=0;t<T;++t){double x=bf(qkv[t*C+c]),v=bf(cw[c*CW+CW-1])*x;for(int j=0;j<CW-1;++j)v+=bf(cw[c*CW+j])*hist[j];conv[t*C+c]=rb(v/(1+std::exp(-v)));for(int j=0;j<CW-2;++j)hist[j]=hist[j+1];hist.back()=x;}
 }
 size_t conv_state_bad=0;for(int c=0;c<C;++c)for(int j=0;j<CW-1;++j)conv_state_bad+=ce[c*(CW-1)+j]!=qkv[(T-CW+1+j)*C+c];
 stats("conv_fp64",conv,cv(qkvc));std::cout<<"\"conv_state_mismatches\":"<<conv_state_bad<<",";
 auto a=load<uint16_t>(p/"a.bin"),b=load<uint16_t>(p/"b.bin"),out=load<uint16_t>(p/"core.bin");auto al=load<float>(p/"a_log.bin"),db=load<float>(p/"dt_bias.bin"),s0=load<float>(p/"state_before.bin"),se=load<float>(p/"state_after.bin");
 std::vector<double> state(s0.begin(),s0.end()),output(out.size());
 #pragma omp parallel for
 for(int h=0;h<LV;++h){int hk=h/(LV/LK);double* S=state.data()+size_t(h)*V*K;std::vector<double> q(K),k(K);double A=std::exp(double(al[h]));
  for(int t=0;t<T;++t){const uint16_t* row=qkvc.data()+size_t(t)*C;double qs=1e-6,ks=1e-6;
   for(int d=0;d<K;++d){q[d]=bf(row[hk*K+d]);k[d]=bf(row[LK*K+hk*K+d]);qs+=q[d]*q[d];ks+=k[d]*k[d];}
   double qi=scale/std::sqrt(qs),ki=1/std::sqrt(ks);for(int d=0;d<K;++d){q[d]*=qi;k[d]*=ki;}
   double gate=bf(a[t*LV+h])+double(db[h]);double decay=std::exp(-A*(gate>20?gate:std::log1p(std::exp(gate))));double beta=1/(1+std::exp(-double(bf(b[t*LV+h]))));
   for(int v=0;v<V;++v){double* s=S+v*K;double dot=0;for(int d=0;d<K;++d){s[d]*=decay;dot+=s[d]*k[d];}double delta=(bf(row[2*LK*K+h*V+v])-dot)*beta;double o=0;for(int d=0;d<K;++d){s[d]+=delta*k[d];o+=s[d]*q[d];}output[(size_t(t)*LV+h)*V+v]=rb(o);}
  }
 }
 stats("recurrence_fp64",output,cv(out));stats("state_fp64",state,cv(se));
 auto z=load<uint16_t>(p/"z.bin"),w=load<uint16_t>(p/"norm_weight.bin"),norm=load<uint16_t>(p/"normed.bin");std::vector<double> normalized(norm.size());
 #pragma omp parallel for
 for(int r=0;r<T*LV;++r){double ss=0;for(int d=0;d<V;++d){double x=bf(out[r*V+d]);ss+=x*x;}double inv=1/std::sqrt(ss/V+eps);for(int d=0;d<V;++d){float u=rb(bf(out[r*V+d])*inv);float weighted=rb(u*bf(w[d]));normalized[r*V+d]=rb(weighted/(1+std::exp(-double(bf(z[r*V+d])))));}}
 stats("gated_norm_fp64",normalized,cv(norm));std::cout<<"\"kind\":\"gdn\"}"<<std::endl;
}
std::vector<uint16_t> norm_rope(const uint16_t* x,const uint16_t* w,int dim,int rotary,int64_t pos,const std::vector<float>& inv,float eps,float mscale){
 double ss=0;for(int d=0;d<dim;++d)ss+=double(bf(x[d]))*bf(x[d]);float rs=float(1/std::sqrt(ss/dim+double(eps)));std::vector<float> a(dim);for(int d=0;d<dim;++d)a[d]=rb(bf(x[d])*rs*(1+bf(w[d])));std::vector<uint16_t> out(dim);int half=rotary/2;
 for(int d=0;d<dim;++d){float y=a[d];if(d<rotary){float angle=float(pos)*inv[d%half];float co=rb(std::cos(angle)*mscale),si=rb(std::sin(angle)*mscale);float rotated=d<half?-a[d+half]:a[d-half];y=rb(rb(a[d]*co)+rb(rotated*si));}out[d]=float_to_bf16_bits(y);}return out;
}
void qsa(fs::path p){
 int T,H,KV,D,IH,ID,KP,NS,R;int64_t pos;double scale,eps;std::ifstream(p/"qsa.meta")>>T>>H>>KV>>D>>IH>>ID>>KP>>NS>>scale>>eps>>R>>pos;
 auto scalars=load<float>(p/"qsa_scalars.bin");scale=scalars[1];
 auto qr=load<uint16_t>(p/"q.bin"),qw=load<uint16_t>(p/"q_norm.bin"),idx=load<uint16_t>(p/"idx.bin"),iqw=load<uint16_t>(p/"index_q_norm.bin"),ikw=load<uint16_t>(p/"index_k_norm.bin");auto inv=load<float>(p/"inv_freq.bin");std::vector<uint16_t> qref,iqref,cref;
 for(int h=0;h<H;++h){auto one=norm_rope(qr.data()+h*2*D,qw.data(),D,R,pos,inv,scalars[0],scalars[2]);qref.insert(qref.end(),one.begin(),one.end());}
 for(int h=0;h<IH;++h){auto one=norm_rope(idx.data()+(size_t(T-1)*(IH+1)+h)*ID,iqw.data(),ID,R,pos,inv,scalars[0],scalars[2]);iqref.insert(iqref.end(),one.begin(),one.end());}
 auto cache_for_check=load<uint16_t>(p/"index_cache.bin");int64_t firstpos=pos+1-T;std::vector<uint16_t> actual_compressed;
 for(int i=0;i<T/KP;++i){std::vector<uint16_t> mean(ID);for(int d=0;d<ID;++d){float sum=0;for(int j=0;j<KP;++j)sum+=bf(idx[(size_t(i*KP+j)*(IH+1)+IH)*ID+d]);mean[d]=float_to_bf16_bits(sum/KP);}auto one=norm_rope(mean.data(),ikw.data(),ID,R,firstpos+i*KP,inv,scalars[0],scalars[2]);cref.insert(cref.end(),one.begin(),one.end());auto start=cache_for_check.begin()+(firstpos/KP+i)*ID;actual_compressed.insert(actual_compressed.end(),start,start+ID);}
 stats("q_norm_rope",cv(qref),cv(load<uint16_t>(p/"qn.bin")));stats("index_q_norm_rope",cv(iqref),cv(load<uint16_t>(p/"qi.bin")));stats("compression_norm_rope",cv(cref),cv(actual_compressed));

 auto qi=load<uint16_t>(p/"qi.bin"),ic=load<uint16_t>(p/"index_cache.bin");auto keys=load<uint64_t>(p/"scores.bin");auto selected=load<int32_t>(p/"selected.bin");int N=keys.size();
 std::vector<double> scores(N);std::vector<uint64_t> exact_keys(N);size_t key_bad=0;
 #pragma omp parallel for reduction(+:key_bad)
 for(int n=0;n<N;++n){double s=0;float lanes[32]={};for(int h=0;h<IH;++h){double dot=0;for(int d=0;d<ID;++d)dot+=double(bf(qi[h*ID+d]))*bf(ic[n*ID+d]);s+=std::max(dot,0.0);for(int lane=0;lane<8;++lane){float v=0;for(int j=0;j<16;++j)v=std::fma(bf(qi[h*ID+lane*16+j]),bf(ic[n*ID+lane*16+j]),v);lanes[h*8+lane]=v;}}scores[n]=s/std::sqrt(double(ID));
  for(int mask:{1,2,4,8,16}){if(mask==8)for(float& x:lanes)x=std::max(x,0.f);float next[32];for(int lane=0;lane<32;++lane)next[lane]=lanes[lane]+lanes[lane^mask];std::copy(next,next+32,lanes);}
  float sf=lanes[0]/std::sqrt(float(ID));uint32_t u=std::bit_cast<uint32_t>(sf);uint32_t sortable=u^((int32_t(u)<0)?0xffffffffu:0x80000000u);exact_keys[n]=(uint64_t(~sortable)<<21)|uint64_t(n);key_bad+=exact_keys[n]!=keys[n];
 }
 std::vector<int> order(N);std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});order.resize(std::min(512,N));std::sort(order.begin(),order.end());std::vector<int32_t> expanded;for(int x:order)for(int k=0;k<KP;++k)expanded.push_back(x*KP+k);for(int64_t i=(pos+1)/KP*KP;i<=pos;++i)expanded.push_back(i);
 std::sort(exact_keys.begin(),exact_keys.end());std::vector<int32_t> selected_from_keys;for(int i=0;i<std::min(512,N);++i)selected_from_keys.push_back(exact_keys[i]&((1<<21)-1));std::sort(selected_from_keys.begin(),selected_from_keys.end());std::vector<int32_t> expected;for(int x:selected_from_keys)for(int k=0;k<KP;++k)expected.push_back(x*KP+k);for(int64_t i=(pos+1)/KP*KP;i<=pos;++i)expected.push_back(i);
 std::cout<<"\"score_key_mismatches\":"<<key_bad<<",\"selection_matches_source_order\":"<<(expected==selected?"true":"false")<<",\"selection_matches_fp64\":"<<(expanded==selected?"true":"false")<<",";
 auto kt=load<uint16_t>(p/"k_cache_tail.bin"),kn=load<uint16_t>(p/"kn_tail.bin"),vt=load<uint16_t>(p/"v_cache_tail.bin"),vv=load<uint16_t>(p/"v_tail.bin");std::cout<<"\"cache_append_exact\":"<<(kt==kn&&vt==vv?"true":"false")<<",";
 auto q=load<uint16_t>(p/"qn.bin"),k=load<uint16_t>(p/"k_cache.bin"),v=load<uint16_t>(p/"v_cache.bin"),qgate=load<uint16_t>(p/"q.bin");auto actual=load<float>(p/"attention_float.bin");auto gated=load<uint16_t>(p/"attention_gated.bin");int S=selected.size();std::vector<double> high(H*D),rounded(H*D),gref(H*D);
 #pragma omp parallel for
 for(int h=0;h<H;++h){int kh=h/(H/KV);std::vector<double> logits(S),weights(S);double maxlog=-INFINITY;for(int t=0;t<S;++t){double sum=0;for(int d=0;d<D;++d)sum+=double(bf(q[h*D+d]))*bf(k[(t*KV+kh)*D+d]);logits[t]=sum*scale;maxlog=std::max(maxlog,logits[t]);}double den=0;for(int t=0;t<S;++t){weights[t]=std::exp(logits[t]-maxlog);den+=weights[t];}for(int d=0;d<D;++d){double full=0,quant=0;for(int t=0;t<S;++t){double value=bf(v[(t*KV+kh)*D+d]);full+=weights[t]*value;quant+=rb(weights[t])*value;}high[h*D+d]=full/den;rounded[h*D+d]=quant/den;float gate=rb(1/(1+std::exp(-double(bf(qgate[h*2*D+D+d])))));gref[h*D+d]=rb(rb(rounded[h*D+d])*gate);}}
 stats("attention_normalized_fp64",high,cv(actual));stats("attention_unnormalized_bf16",rounded,cv(actual));stats("gated_attention_unnormalized_bf16",gref,cv(gated));
 double rms=0;for(double x:gref)rms+=x*x;rms=std::sqrt(rms/gref.size());size_t contract_bad=0;for(size_t i=0;i<gref.size();++i)contract_bad+=std::abs(double(bf(gated[i]))-gref[i])>std::max(0.02*rms,(8.0/128)*std::max(std::abs(gref[i]),std::abs(double(bf(gated[i])))));
 std::cout<<"\"multisplit_contract_mismatches\":"<<contract_bad<<",\"kind\":\"qsa\"}"<<std::endl;
}
int main(int argc,char** argv){try{if(argc!=2)throw std::runtime_error("usage: audit CAPTURE_LAYER");fs::path p=argv[1];std::cout<<std::setprecision(12)<<"{\"path\":\""<<p.string()<<"\",";if(fs::exists(p/"gdn.meta"))gdn(p);else if(fs::exists(p/"qsa.meta"))qsa(p);else throw std::runtime_error("no metadata");}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

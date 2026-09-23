#include "kernels/gdn_chunk.hpp"
#include <cstdio>
#include <exception>
extern "C" size_t issue4_gdn_bytes(int t, int k, int v) {
 return dgpp::gdn_chunk_workspace_bytes(t,k,v);
}
extern "C" int issue4_gdn(const uint16_t* x, const uint16_t* a, const uint16_t* b,
 const float* al, const float* dt, float* st, uint16_t* o, int t, int hk, int hv,
 float scale, void* ws, size_t n, cudaStream_t stream) {
 try { dgpp::gdn_chunk_fwd(x,a,hv,b,hv,al,dt,st,o,t,hk,hv,scale,ws,n,stream); return 0; }
 catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
#include "kernels/kda.hpp"
extern "C" int issue4_gdn_recurrent(const uint16_t* x, const uint16_t* a, const uint16_t* b,
 const float* al, const float* dt, float* st, uint16_t* o, int t, int hk, int hv,
 float scale, void*, size_t, cudaStream_t stream) {
 try { dgpp::gdn_recurrent_fwd(x,a,hv,b,hv,al,dt,st,o,t,hv,hv/hk,128,128,scale,stream); return 0; }
 catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "common/cuda_check.hpp"
#include "kernels/qwen_moe.hpp"
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  const std::string root=argv[1];
  constexpr size_t n=128*2560;
  std::vector<uint16_t> in(n), expected(n), got(n), inplace(n);
  std::vector<float> scales(96);
  std::ifstream(root+"/input.bin",std::ios::binary).read((char*)in.data(),n*2);
  std::ifstream(root+"/global.bin",std::ios::binary).read((char*)scales.data(),scales.size()*4);
  std::ifstream ref(root+"/expected.bin",std::ios::binary);
  uint16_t *di,*dout;
  DGPP_CUDA_OK(cudaMalloc(&di,n*2)); DGPP_CUDA_OK(cudaMalloc(&dout,n*2));
  size_t bad=0, inplace_bad=0, signed_zero=0;
  for(float global:scales) {
    ref.read((char*)expected.data(),n*2);if(!ref)throw std::runtime_error("short fixture");
    DGPP_CUDA_OK(cudaMemcpy(di,in.data(),n*2,cudaMemcpyHostToDevice));
    dgpp::issue4_nvfp4_activations(di,dout,128,2560,global,nullptr);
    DGPP_CUDA_OK(cudaMemcpy(got.data(),dout,n*2,cudaMemcpyDeviceToHost));
    dgpp::issue4_nvfp4_activations(di,di,1024,320,global,nullptr);
    DGPP_CUDA_OK(cudaMemcpy(inplace.data(),di,n*2,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i) {
      if(got[i]!=expected[i]) {
        if((got[i]&0x7fff)==0 && (expected[i]&0x7fff)==0) ++signed_zero;
        else { if(bad<10)std::cerr<<"mismatch "<<i<<" "<<got[i]<<" "<<expected[i]<<"\n"; ++bad; }
      }
      inplace_bad += got[i]!=inplace[i];
    }
  }
  cudaFree(di);cudaFree(dout);
  std::cout<<"{\"cases\":96,\"scalars\":"<<n*96<<",\"mismatches\":"<<bad<<",\"inplace_mismatches\":"<<inplace_bad<<",\"signed_zero_differences\":"<<signed_zero<<"}\n";
  return bad || inplace_bad ? 1 : 0;
}

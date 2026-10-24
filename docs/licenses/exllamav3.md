# exllamav3

The EXL3 trellis weight format served for the DeepSeek-V4.1-Flash-1017
3.5 bpw pack (`src/models/exl3_matrix.hpp`, `src/kernels/exl3_codec.cuh`,
`src/kernels/exl3_gemv.cu`) implements the format of
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3):
the `mul1` procedural codebook and its three-instruction decode
(`exllamav3_ext/quant/codebook.cuh`), the tile bit-stream layout and the
per-lane state windows (`pack.cu`, `exl3_dq.cuh`), and the 128-point
Hadamard rotation's stage order with the input and output scale vectors
(`hadamard_inner.cuh`). The constants and the operation order are kept so
the engine's values are the reference's. The code here is a re-implementation
on this engine's kernel cores; the arithmetic it reproduces is exllamav3's.

exllamav3 is distributed under the MIT License:

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

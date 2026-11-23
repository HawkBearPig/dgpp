#pragma once
// A device-to-device copy as a KERNEL. A memcpy node may not enter a
// captured decode graph (it executes on the copy-engine queue), so every
// copy a decode pass records goes through here: a family's draft state
// snapshot, a ring's rollback copy, the verify logits a sampled request
// keeps.
//
// Any size and any alignment: the copy moves 16-byte units when both
// pointers and the size are whole units, 4-byte words when they are whole
// words, and bytes otherwise. A caller never states which. Logits rows of a
// vocabulary that is not a multiple of four values are not whole 16-byte
// units, and an odd number of them does not end on one.
#include <cstddef>

#include <cuda_runtime.h>

namespace dgpp {

void device_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream);

}  // namespace dgpp

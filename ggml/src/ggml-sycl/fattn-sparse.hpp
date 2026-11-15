#ifndef GGML_SYCL_FATTN_SPARSE_HPP
#define GGML_SYCL_FATTN_SPARSE_HPP

#include "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// ggml_flash_attn_ext_rows: each q slice attends to its own list of K/V rows, read in place.
bool ggml_sycl_flash_attn_ext_rows_supported(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_SPARSE_HPP

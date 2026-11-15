#include "fattn.hpp"
#include "fattn-sparse.hpp"
#include "dmmv.hpp"

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
#    include <sycl/ext/intel/esimd.hpp>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

static constexpr int64_t SPARSE_FA_PAD       = 256;
static constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

extern int g_ggml_sycl_enable_sparse_fa;
extern int g_ggml_sycl_debug_sparse_fa;
extern int g_ggml_sycl_sparse_fa_margin;

static int sparse_fa_enabled(void) {
    return g_ggml_sycl_enable_sparse_fa;
}

static int sparse_fa_debug(void) {
    return g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
static int sparse_fa_margin(void) {
    return g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
static void sparse_fa_compact_mask(sycl::queue * stream,
                                   const sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   const int64_t n_kv,
                                   const int64_t n_kv_g) {
    constexpr size_t WG = 256;
    const size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = (int64_t) item.get_global_id(0);
            if (i >= n_kv || !sycl::isfinite((float) mask[i])) {
                return;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            const int32_t pos = ctr.fetch_add(1);
            if (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
static void sparse_fa_gather_rows(sycl::queue * stream,
                                  const uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const size_t row_size,
                                  const size_t src_nb1,
                                  const size_t src_nb2,
                                  const int64_t n_kv_g,
                                  const int64_t n_head) {
    GGML_ASSERT(row_size % sizeof(uint32_t) == 0);
    const size_t words = row_size / sizeof(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, words),
        [=](sycl::id<3> id) {
            const int64_t h    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];
            const size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            if (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                return;
            }

            const uint32_t * src_row =
                (const uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

static void sparse_fa_gather_mask(sycl::queue * stream,
                                  const sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const int64_t n_kv_g,
                                  const int64_t n_rows,
                                  const size_t mask_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            const int64_t r    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            if (slot < (int64_t) *count) {
                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

static bool sparse_fa_applicable(const ggml_tensor * dst, int64_t & n_kv_g_out) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V || !mask) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (n_kv_max <= 0) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // single-token decode only; prefill amortises the scan already
    if (Q->ne[1] != 1) {
        return false;
    }
    if (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        return false;
    }
    if (K->ne[2] != V->ne[2]) {
        return false;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);
    if (k_row % sizeof(uint32_t) || v_row % sizeof(uint32_t)) {
        return false;
    }

    const int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    if (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        return false;
    }

    n_kv_g_out = n_kv_g;
    return true;
}

bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    if (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        return false;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int64_t n_kv     = K->ne[1];
    const int64_t n_head_k = K->ne[2];
    const int64_t n_rows_m = mask->ne[1];

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_rows_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_mask = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_mask(stream, (const sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_rows(stream, (const uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_rows(stream, (const uint8_t *) V->data, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_mask(stream, (const sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m,
                          mask->nb[1] / sizeof(sycl::half));

    if (sparse_fa_debug() && stream->ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::recording) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, sizeof(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (long long) n_kv, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (int) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.data      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (size_t) n_kv_g * k_row;
    K_g.nb[3]     = (size_t) n_head_k * n_kv_g * k_row;
    K_g.view_src  = nullptr;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.data      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (size_t) n_kv_g * v_row;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = nullptr;
    V_g.view_offs = 0;

    ggml_tensor M_g = *mask;
    M_g.data      = d_mask;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * sizeof(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * mask->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = nullptr;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    return true;
}

// Flash attention over a list of K/V rows per q slice (ggml_flash_attn_ext_rows): one work-group per (slice, token,
// head) reads its rows straight from the cache, so the cost follows the length of the list, not the size of the cache.
static constexpr int64_t FA_ROWS_WG       = 256;
static constexpr int64_t FA_ROWS_MAX_COLS = 4096;

bool ggml_sycl_flash_attn_ext_rows_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * rows = dst->src[5];
    if (rows == nullptr || dst->src[4] != nullptr) { // no sinks
        return false;
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    return max_bias == 0.0f && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
           (mask == nullptr || mask->type == GGML_TYPE_F16) && rows->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 &&
           q->nb[0] == sizeof(float) && k->nb[0] == sizeof(sycl::half) && v->nb[0] == sizeof(sycl::half) &&
           rows->nb[0] == sizeof(int32_t) && ggml_is_contiguous(dst) && k->ne[3] == 1 && v->ne[3] == 1 &&
           q->ne[2] % k->ne[2] == 0 && q->ne[2] % v->ne[2] == 0 && q->ne[0] == k->ne[0] &&
           rows->ne[0] >= 1 && rows->ne[0] <= FA_ROWS_MAX_COLS && rows->ne[1] == q->ne[3] &&
           (mask == nullptr || (mask->ne[0] == rows->ne[0] && mask->ne[1] == q->ne[1] && mask->nb[0] == sizeof(sycl::half)));
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// One thread per (slice, token, head): the query stays in registers, each listed row of K and V is read with one wide
// load, and the softmax is accumulated on the fly as in flash attention. D = head size of K and V.
template <int D>
static void fa_rows_esimd(const char * q_d, const char * k_d, const char * v_d, const char * m_d, const char * r_d,
                          float * o_d, const int64_t N, const int64_t H, const int64_t S, const int64_t n_cols,
                          const int64_t rk2, const int64_t rv2, const float scale, const size_t nbq1, const size_t nbq2,
                          const size_t nbq3, const size_t nbk1, const size_t nbk2, const size_t nbv1, const size_t nbv2,
                          const size_t nbr1, const size_t nbm1, const size_t nbm2, const size_t nbm3, const int64_t nem2,
                          const int64_t nem3, dpct::queue_ptr stream) {
    constexpr int CH = D < 128 ? D : 128; // floats per load
    constexpr int W  = 32;                 // K and V values per step
    stream->parallel_for(sycl::range<1>((size_t) (S * N * H)), [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
        using namespace sycl::ext::intel::esimd;
        const int64_t g  = (int64_t) idx[0];
        const int64_t i3 = g / (N * H);
        const int64_t i1 = (g / H) % N;
        const int64_t i2 = g % H;

        const float *    qp = (const float *) (q_d + i1 * nbq1 + i2 * nbq2 + i3 * nbq3);
        const int32_t *  rp = (const int32_t *) (r_d + i3 * nbr1);
        const char *     kb = k_d + (i2 / rk2) * nbk2;
        const char *     vb = v_d + (i2 / rv2) * nbv2;
        const uint16_t * mp = m_d ? (const uint16_t *) (m_d + i1 * nbm1 + (i2 % nem2) * nbm2 + (i3 % nem3) * nbm3) : nullptr;

        simd<float, D> qv;
#pragma unroll
        for (int o = 0; o < D; o += CH) {
            qv.template select<CH, 1>(o) = block_load<float, CH>(qp + o, element_aligned_tag{});
        }

        simd<float, D> out(0.0f);
        float          M   = -INFINITY;
        float          sum = 0.0f;
        for (int64_t c = 0; c < n_cols; ++c) {
            const int32_t r = rp[c];
            float         m = 0.0f;
            if (mp) {
                const uint16_t mb = mp[c];
                if (mb == 0xFC00) { // -inf
                    continue;
                }
                if (mb != 0) {
                    simd<uint16_t, 8> mv(mb);
                    m = (float) convert<float>(mv.template bit_cast_view<sycl::half>().read())[0];
                }
            }
            if (r < 0) {
                continue;
            }
            // 32 values at a time: whole-row registers for K and V as well would not fit the register file
            const sycl::half * kr = (const sycl::half *) (kb + (size_t) r * nbk1);
            simd<float, W>     acc(0.0f);
#pragma unroll
            for (int o = 0; o < D; o += W) {
                const simd<float, W> qc = qv.template select<W, 1>(o);
                acc += qc * convert<float>(block_load<sycl::half, W>(kr + o, element_aligned_tag{}));
            }
            const float s = reduce<float>(acc, std::plus<>()) * scale + m;
            if (s > M) {
                if (M > -INFINITY) {
                    const float f = (float) exp(simd<float, 8>(M - s))[0];
                    out *= f;
                    sum *= f;
                }
                M = s;
            }
            const float w = (float) exp(simd<float, 8>(s - M))[0];
            sum += w;
            const sycl::half * vr = (const sycl::half *) (vb + (size_t) r * nbv1);
#pragma unroll
            for (int o = 0; o < D; o += W) {
                simd<float, W> oc = out.template select<W, 1>(o);
                oc += convert<float>(block_load<sycl::half, W>(vr + o, element_aligned_tag{})) * w;
                out.template select<W, 1>(o) = oc;
            }
        }
        out *= sum > 0.0f ? 1.0f / sum : 0.0f;

        float * op = o_d + (i3 * N * H + i1 * H + i2) * D;
#pragma unroll
        for (int o = 0; o < D; o += CH) {
            const simd<float, CH> t = out.template select<CH, 1>(o);
            t.copy_to(op + o);
        }
    });
}
#endif

// Row lists by gathering: the listed K and V rows of a few slices are copied next to each other (one thread per word, so
// the cache misses of scattered rows overlap instead of queueing up in one thread), then the dense kernels run on that
// small cache with one stream per slice. Returns false if the dense kernels do not take the gathered shape.
static bool fa_rows_gather(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * rows = dst->src[5];
    if ((mask && (mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != q->ne[3]))) || k->ne[2] != v->ne[2]) {
        return false;
    }
    const size_t k_row = ggml_row_size(k->type, k->ne[0]);
    const size_t v_row = ggml_row_size(v->type, v->ne[0]);
    if (k_row % sizeof(uint32_t) || v_row % sizeof(uint32_t) || (uintptr_t) k->data % 4 || (uintptr_t) v->data % 4 ||
        k->nb[1] % 4 || k->nb[2] % 4 || v->nb[1] % 4 || v->nb[2] % 4) {
        return false;
    }
    const int64_t N = q->ne[1], S = q->ne[3], Hk = k->ne[2];
    const int64_t n_cols = rows->ne[0];
    // GGML_SYCL_FA_ROWS_UNION=1 (single-token slices only): the slices of a pass become the batch of ONE stream over the
    // lists of all of them laid end to end, each token masked to its own list. That is the shape the dense kernels are
    // tuned for (a few queries over about a thousand keys); =0 keeps one stream per slice.
    const bool    uni    = N == 1 && ggml_live_switch("GGML_SYCL_FA_ROWS_UNION", 1) != 0;
    const int64_t cs_max = std::min<int64_t>(S, uni ? 8 : 32); // slices per pass, bounds the gathered copy
    const int64_t n_g    = GGML_PAD(uni ? n_cols * cs_max : n_cols, SPARSE_FA_PAD);

    ggml_tensor K_g = *k;
    K_g.ne[1] = n_g;   K_g.ne[3] = uni ? 1 : cs_max;
    K_g.nb[1] = k_row; K_g.nb[2] = (size_t) n_g * k_row; K_g.nb[3] = (size_t) Hk * n_g * k_row;
    K_g.view_src = nullptr; K_g.view_offs = 0;
    ggml_tensor V_g = *v;
    V_g.ne[1] = n_g;   V_g.ne[3] = uni ? 1 : cs_max;
    V_g.nb[1] = v_row; V_g.nb[2] = (size_t) n_g * v_row; V_g.nb[3] = (size_t) Hk * n_g * v_row;
    V_g.view_src = nullptr; V_g.view_offs = 0;
    ggml_tensor M_g = mask ? *mask : *rows;
    M_g.type  = GGML_TYPE_F16;
    M_g.ne[0] = n_g; M_g.ne[1] = uni ? cs_max : N; M_g.ne[2] = 1; M_g.ne[3] = uni ? 1 : cs_max;
    M_g.nb[0] = sizeof(sycl::half); M_g.nb[1] = (size_t) n_g * sizeof(sycl::half); M_g.nb[2] = M_g.nb[1] * M_g.ne[1]; M_g.nb[3] = M_g.nb[2];
    M_g.view_src = nullptr; M_g.view_offs = 0;
    ggml_tensor q_g = *q;
    q_g.ne[3] = cs_max; q_g.view_src = nullptr; q_g.view_offs = 0;
    ggml_tensor dst_g = *dst;
    dst_g.ne[3] = cs_max;
    if (uni) { // [DK, 1, H, cs] and [DV, H, 1, cs] are the same memory as [DK, cs, H, 1] and [DV, H, cs, 1]
        q_g.ne[1] = cs_max; q_g.ne[3] = 1; q_g.nb[1] = q->nb[3]; q_g.nb[3] = q->nb[3] * cs_max;
        dst_g.ne[2] = cs_max; dst_g.ne[3] = 1; dst_g.nb[3] = dst->nb[3] * cs_max;
    }
    dst_g.src[0] = &q_g; dst_g.src[1] = &K_g; dst_g.src[2] = &V_g; dst_g.src[3] = &M_g; dst_g.src[4] = nullptr; dst_g.src[5] = nullptr;
    dst_g.op_params[4] = 0; // no row hint: the gathered cache is already small
    dst_g.view_src = nullptr; dst_g.view_offs = 0;
    dst_g.data = nullptr;
    const bool ok = ggml_sycl_flash_attn_ext_supported(ctx.device, &dst_g);
    {   // once per shape: which way the row lists go
        static int64_t seen_s = -1;
        if (seen_s != S) {
            seen_s = S;
            GGML_LOG_INFO("%s: %lld slices x %lld queries, %lld rows each -> %s, %lld keys per pass of %lld slices\n", __func__,
                          (long long) S, (long long) N, (long long) n_cols, !ok ? "NOT taken by the dense kernels" : uni ? "one stream" : "a stream per slice",
                          (long long) n_g, (long long) cs_max);
        }
    }
    if (!ok) {
        return false;
    }
    // the dense kernels keep their scratch right after the output, so the output of a pass goes to a buffer with that room
    const size_t out_bytes = (size_t) cs_max * dst->nb[3];
    const size_t out_alloc = ggml_sycl_flash_attn_ext_get_alloc_size(&dst_g) + 256;

    const int64_t n_str = uni ? 1 : cs_max; // streams of a full pass
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_str * Hk * n_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_str * Hk * n_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) cs_max * N * n_g);
    ggml_sycl_pool_alloc<uint8_t>    o_alloc(ctx.pool(), std::max(out_alloc, out_bytes));
    uint8_t *    d_K = k_alloc.get();
    uint8_t *    d_V = v_alloc.get();
    sycl::half * d_M = m_alloc.get();
    uint8_t *    d_O = o_alloc.get();
    K_g.data = d_K; V_g.data = d_V; M_g.data = d_M; dst_g.data = d_O;

    const char *       k_d  = (const char *) k->data;
    const char *       v_d  = (const char *) v->data;
    const char *       r_d  = (const char *) rows->data;
    const sycl::half * m_d  = mask ? (const sycl::half *) mask->data : nullptr;
    const size_t       nbk1 = k->nb[1], nbk2 = k->nb[2], nbv1 = v->nb[1], nbv2 = v->nb[2], nbr1 = rows->nb[1];
    const size_t       ms1  = mask ? mask->nb[1] / sizeof(sycl::half) : 0, ms3 = mask ? mask->nb[3] / sizeof(sycl::half) : 0;
    const int64_t      nem3 = mask ? mask->ne[3] : 1;
    const size_t       kw   = k_row / sizeof(uint32_t), vw = v_row / sizeof(uint32_t);

    dpct::queue_ptr stream = ctx.stream();
    for (int64_t s0 = 0; s0 < S; s0 += cs_max) {
        const int64_t cs = std::min<int64_t>(cs_max, S - s0);
        q_g.data = (char *) q->data + (size_t) s0 * q->nb[3];
        if (uni) {
            q_g.ne[1] = M_g.ne[1] = dst_g.ne[2] = cs;
            M_g.nb[2] = M_g.nb[3] = M_g.nb[1] * cs;
        } else {
            K_g.ne[3] = V_g.ne[3] = M_g.ne[3] = q_g.ne[3] = dst_g.ne[3] = cs;
        }

        // All index arithmetic below is 32-bit and free of division by run-time values where it can be: 64-bit division
        // is emulated on this hardware and was most of the cost of the first version of these kernels.
        // key slot = slice * n_cols + column in one stream holding the lists of all slices, else a stream per slice
        const uint32_t u_cs = (uint32_t) cs, u_hk = (uint32_t) Hk, u_nc = (uint32_t) n_cols, u_ng = (uint32_t) n_g, u_n = (uint32_t) N;
        const uint32_t n_used = uni ? u_cs * u_nc : u_nc;   // listed slots of a stream; the rest up to n_g is padding
        const auto gather = [&](const char * src, uint8_t * out, const size_t nb1, const size_t nb2, const size_t words, const size_t row_size) {
            // (slice, head) x column: one work-item copies a whole row. One item per word meant about ten thousand
            // threads of 32 words each per call, and starting the threads cost more than the copy.
            const uint32_t n_words = (uint32_t) words;
            stream->parallel_for(sycl::range<2>((size_t) u_cs * u_hk, (size_t) u_nc), [=](sycl::id<2> id) {
                const uint32_t sh = (uint32_t) id[0], col = (uint32_t) id[1];
                const uint32_t sl = sh / u_hk, h = sh - sl * u_hk;
                const uint32_t str = uni ? h : sh;                  // stream and head of the output
                const uint32_t slot = uni ? sl * u_nc + col : col;
                uint32_t *     o   = (uint32_t *) (out + ((size_t) str * u_ng + slot) * row_size);
                const int32_t  r   = ((const int32_t *) (r_d + (size_t) (s0 + sl) * nbr1))[col];
                if (r >= 0) {
                    const uint32_t * in = (const uint32_t *) (src + (size_t) r * nb1 + (size_t) h * nb2);
                    for (uint32_t w = 0; w < n_words; ++w) {
                        o[w] = in[w];
                    }
                } else {
                    for (uint32_t w = 0; w < n_words; ++w) {
                        o[w] = 0;
                    }
                }
            });
            if (u_ng > n_used) { // padding rows must be zero: a masked key still enters the dot product
                const uint32_t n_str = uni ? u_hk : u_cs * u_hk;
                stream->parallel_for(sycl::range<3>((size_t) n_str, (size_t) (u_ng - n_used), words), [=](sycl::id<3> id) {
                    ((uint32_t *) (out + ((size_t) id[0] * u_ng + n_used + id[1]) * row_size))[id[2]] = 0;
                });
            }
        };
        gather(k_d, d_K, nbk1, nbk2, kw, k_row);
        gather(v_d, d_V, nbv1, nbv2, vw, v_row);
        // mask: (slice, query) x slot
        stream->parallel_for(sycl::range<2>((size_t) u_cs * u_n, (size_t) u_ng), [=](sycl::id<2> id) {
            const uint32_t sq = (uint32_t) id[0], slot = (uint32_t) id[1];
            const uint32_t sl = sq / u_n, n = sq - sl * u_n;
            const uint32_t col = uni ? slot - sl * u_nc : slot;      // column inside this slice's own list
            sycl::half     m   = sycl::half(-INFINITY);
            if (col < u_nc) { // unsigned: slots before or after the own list fall out
                const int32_t r = ((const int32_t *) (r_d + (size_t) (s0 + sl) * nbr1))[col];
                if (r >= 0) {
                    m = m_d ? m_d[(size_t) col + (size_t) n * ms1 + (size_t) (nem3 == 1 ? 0 : s0 + sl) * ms3] : sycl::half(0.0f);
                }
            }
            d_M[(size_t) sq * u_ng + slot] = m;
        });

        ggml_sycl_flash_attn_ext(ctx, &dst_g);

        // a kernel, not a queue memcpy: on Level Zero that is not always ordered behind the kernels before it
        const uint32_t * so = (const uint32_t *) d_O;
        uint32_t *       to = (uint32_t *) ((char *) dst->data + (size_t) s0 * dst->nb[3]);
        stream->parallel_for(sycl::range<1>((size_t) cs * dst->nb[3] / sizeof(uint32_t)), [=](sycl::id<1> i) { to[i] = so[i]; });
    }
    return true;
}

void ggml_sycl_flash_attn_ext_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    // GGML_SYCL_FA_ROWS_GATHER=0: the single-kernel versions below
    static const bool gather_on = getenv("GGML_SYCL_FA_ROWS_GATHER") == nullptr || atoi(getenv("GGML_SYCL_FA_ROWS_GATHER")) != 0;
    if (gather_on && fa_rows_gather(ctx, dst)) {
        return;
    }
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * rows = dst->src[5];

    const int64_t DK = q->ne[0], N = q->ne[1], H = q->ne[2], S = q->ne[3];
    const int64_t DV = v->ne[0];
    const int64_t n_cols = rows->ne[0];
    const int64_t rk2 = H / k->ne[2], rv2 = H / v->ne[2];

    float scale = 1.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const char * q_d = (const char *) q->data;
    const char * k_d = (const char *) k->data;
    const char * v_d = (const char *) v->data;
    const char * m_d = mask ? (const char *) mask->data : nullptr;
    const char * r_d = (const char *) rows->data;
    float *      o_d = (float *) dst->data;

    const size_t nbq1 = q->nb[1], nbq2 = q->nb[2], nbq3 = q->nb[3];
    const size_t nbk1 = k->nb[1], nbk2 = k->nb[2];
    const size_t nbv1 = v->nb[1], nbv2 = v->nb[2];
    const size_t nbr1 = rows->nb[1];
    const size_t nbm1 = mask ? mask->nb[1] : 0, nbm2 = mask ? mask->nb[2] : 0, nbm3 = mask ? mask->nb[3] : 0;
    const int64_t nem2 = mask ? mask->ne[2] : 1, nem3 = mask ? mask->ne[3] : 1;

    dpct::queue_ptr stream = ctx.stream();
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // GGML_SYCL_FA_ROWS_ESIMD=1: the ESIMD variant (slower than the kernel below as measured, kept for comparison)
    static const bool esimd_on = getenv("GGML_SYCL_FA_ROWS_ESIMD") != nullptr && atoi(getenv("GGML_SYCL_FA_ROWS_ESIMD")) != 0;
    if (esimd_on && g_ggml_sycl_enable_esimd && DK == DV && logit_softcap == 0.0f && (DK == 64 || DK == 128 || DK == 256)) {
#define GGML_SYCL_FA_ROWS_ARGS q_d, k_d, v_d, m_d, r_d, o_d, N, H, S, n_cols, rk2, rv2, scale, nbq1, nbq2, nbq3, nbk1, nbk2, nbv1, \
        nbv2, nbr1, nbm1, nbm2, nbm3, nem2, nem3, stream
        switch (DK) {
            case 64:  fa_rows_esimd<64>(GGML_SYCL_FA_ROWS_ARGS); break;
            case 128: fa_rows_esimd<128>(GGML_SYCL_FA_ROWS_ARGS); break;
            default:  fa_rows_esimd<256>(GGML_SYCL_FA_ROWS_ARGS); break;
        }
#undef GGML_SYCL_FA_ROWS_ARGS
        return;
    }
#endif
    // GGML_SYCL_FA_ROWS_WORDS=0: 16-bit loads of K and V
    static const bool words_on = getenv("GGML_SYCL_FA_ROWS_WORDS") == nullptr || atoi(getenv("GGML_SYCL_FA_ROWS_WORDS")) != 0;
    const bool words = words_on && DK % 2 == 0 && DV % 2 == 0 && (uintptr_t) k_d % 4 == 0 && (uintptr_t) v_d % 4 == 0 &&
                       nbk1 % 4 == 0 && nbk2 % 4 == 0 && nbv1 % 4 == 0 && nbv2 % 4 == 0;
    const auto h2f = [](const uint32_t bits) { return (float) sycl::bit_cast<sycl::half>((uint16_t) bits); };
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> w(sycl::range<1>((size_t) n_cols), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) (S * N * H), (size_t) FA_ROWS_WG), sycl::range<2>(1, (size_t) FA_ROWS_WG)),
            [=](sycl::nd_item<2> item) {
                const int64_t g  = (int64_t) item.get_global_id(0);
                const int64_t l  = (int64_t) item.get_local_id(1);
                const int64_t i3 = g / (N * H);
                const int64_t i1 = (g / H) % N;
                const int64_t i2 = g % H;

                const float *      qp = (const float *) (q_d + i1 * nbq1 + i2 * nbq2 + i3 * nbq3);
                const int32_t *    rp = (const int32_t *) (r_d + i3 * nbr1);
                const char *       kb = k_d + (i2 / rk2) * nbk2;
                const char *       vb = v_d + (i2 / rv2) * nbv2;
                const sycl::half * mp = m_d ? (const sycl::half *) (m_d + i1 * nbm1 + (i2 % nem2) * nbm2 + (i3 % nem3) * nbm3) : nullptr;

                // scores of this lane's columns
                float lmax = -INFINITY;
                for (int64_t c = l; c < n_cols; c += FA_ROWS_WG) {
                    const int32_t r = rp[c];
                    float         m = 0.0f;
                    if (mp) {
                        const uintptr_t pm = (uintptr_t) (mp + c);
                        m = h2f(*(const uint32_t *) (pm & ~(uintptr_t) 3) >> ((uint32_t) (pm & 3) * 8));
                    }
                    float s = -INFINITY;
                    if (r >= 0 && m > -INFINITY) {
                        float acc = 0.0f;
                        if (words) {
                            // two halves per aligned 32-bit load: 16-bit loads are several times slower here
                            const uint32_t * kw = (const uint32_t *) (kb + (size_t) r * nbk1);
                            for (int64_t i = 0; i < DK / 2; ++i) {
                                const uint32_t x = kw[i];
                                acc += qp[2 * i] * h2f(x) + qp[2 * i + 1] * h2f(x >> 16);
                            }
                        } else {
                            const sycl::half * kr = (const sycl::half *) (kb + (size_t) r * nbk1);
                            for (int64_t d = 0; d < DK; ++d) {
                                acc += qp[d] * (float) kr[d];
                            }
                        }
                        s = acc * scale;
                        if (logit_softcap != 0.0f) {
                            s = logit_softcap * sycl::tanh(s);
                        }
                        s += m;
                    }
                    w[c] = s;
                    lmax = sycl::fmax(lmax, s);
                }
                const float M = sycl::reduce_over_group(item.get_group(), lmax, sycl::maximum<float>());

                float lsum = 0.0f;
                for (int64_t c = l; c < n_cols; c += FA_ROWS_WG) {
                    const float e = w[c] > -INFINITY ? sycl::exp(w[c] - M) : 0.0f;
                    w[c] = e;
                    lsum += e;
                }
                const float sum = sycl::reduce_over_group(item.get_group(), lsum, sycl::plus<float>());
                sycl::group_barrier(item.get_group());

                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *     out = o_d + (i3 * N * H + i1 * H + i2) * DV;
                if (words) {
                    // a lane owns two output values, the two halves of one word of every listed V row
                    for (int64_t i = l; i < DV / 2; i += FA_ROWS_WG) {
                        float acc0 = 0.0f, acc1 = 0.0f;
                        for (int64_t c = 0; c < n_cols; ++c) {
                            const float e = w[c];
                            if (e != 0.0f) {
                                const uint32_t x = ((const uint32_t *) (vb + (size_t) rp[c] * nbv1))[i];
                                acc0 += e * h2f(x);
                                acc1 += e * h2f(x >> 16);
                            }
                        }
                        out[2 * i]     = acc0 * inv;
                        out[2 * i + 1] = acc1 * inv;
                    }
                } else {
                    for (int64_t d = l; d < DV; d += FA_ROWS_WG) {
                        float acc = 0.0f;
                        for (int64_t c = 0; c < n_cols; ++c) {
                            const float e = w[c];
                            if (e != 0.0f) {
                                acc += e * (float) ((const sycl::half *) (vb + (size_t) rp[c] * nbv1))[d];
                            }
                        }
                        out[d] = acc * inv;
                    }
                }
            });
    });
}

#ifndef GGML_SYCL_ROT8_HPP
#define GGML_SYCL_ROT8_HPP

// Prompt batches as a rotated int8 matmul (GGML_SYCL_Q4_0_ROT8=1, off by default).
//
// The weights and the activations of a matmul are rotated by the same Walsh-Hadamard transform over blocks of 1024 values
// along K. That leaves every dot product unchanged up to the factor 1024 and spreads the values, so that one scale per
// weight row and one per token keep the values accurate when they are rounded to int8. The products are then one int8
// matmul of oneDNN, which applies the two scales, and a following ADD or SwiGLU of the result, as part of its output.
//
// - Reordered Q4_0 weights are converted on every pass, into a staging buffer. As many of them as fit the free device
//   memory are converted once and kept (GGML_SYCL_ROT8_RESIDENT_MIB caps this, GGML_SYCL_ROT8_RESERVE_MIB, default 3000,
//   stays free). The other quantized types are taken only when their converted form can be kept
//   (GGML_SYCL_ROT8_OTHER_MIB, default 2400).
// - What is kept is given back when a device buffer does not fit, and dropped when the buffer of the weights is freed.
// - Batches under GGML_SYCL_Q4_0_ROT8_MIN_COLS tokens (default 64) and shapes the kernels do not take use the other paths.
// - The result differs slightly from the f16 paths: on Qwen3.8-27B Q4_0, KLD 0.0013 and 98.5 % same top token.

#include <array>
#include <map>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdio>
#include "common.hpp"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace ggml_sycl_rot8 {

static constexpr int SG = 32;        // one sub-group holds one block
static constexpr int BL = 1024;      // rotation block
static constexpr int EW = BL / SG;   // values of a block per lane

// lane L of the sub-group keeps the elements i * SG + L of the block in reg[i]
static inline void fwht(float * reg, const sycl::sub_group & sg, const int lane) {
#pragma unroll
    for (int h = 1; h < SG; h *= 2) {
#pragma unroll
        for (int j = 0; j < EW; ++j) {
            const float v  = reg[j];
            const float v2 = sycl::permute_group_by_xor(sg, v, h);
            reg[j]         = (lane & h) == 0 ? v + v2 : v2 - v;
        }
    }
#pragma unroll
    for (int h = SG; h < BL; h *= 2) {
        const int step = h / SG;
#pragma unroll
        for (int j = 0; j < EW; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x     = reg[j + k];
                const float y     = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// The converted weights are written [n][k], as they are read, and the matmul gets them transposed: the conversion is
// 3.6 times faster than writing [k][n] with a stride of N and the matmul as fast (27B, B70). GGML_SYCL_ROT8_ROW_MAJOR=0: [k][n]
static inline bool row_major() {
    static const bool on = std::getenv("GGML_SYCL_ROT8_ROW_MAJOR") == nullptr || std::atoi(std::getenv("GGML_SYCL_ROT8_ROW_MAJOR")) != 0;
    return on;
}

// Reordered Q4_0: the nibbles of all rows (K / 2 bytes a row), then the f16 scales (K / 32 a row). A block of 1024 is
// 32 stored blocks, so reg[i] of every lane is stored block i. out == nullptr: the largest rotated value of each
// block to bmax; else int8 with the row scales, K major ([k][n], the layout the matmul reads fastest).
static inline void weights(const uint8_t * qs, const sycl::half * d, const int N, const int K, float * bmax,
                           const float * rscale, int8_t * out, dpct::queue_ptr stream) {
    const int   nblk = K / BL;
    const bool  rm   = row_major();
    const float wl   = 127.0f;
    stream->parallel_for(sycl::nd_range<2>({ (size_t) N * nblk, SG }, { 4, SG }),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto   sg   = it.get_sub_group();
        const int    lane = sg.get_local_linear_id();
        const size_t id   = it.get_global_id(0);
        const int    r    = (int) (id / nblk);
        const int    b    = (int) (id % nblk);
        const uint8_t *    p  = qs + (size_t) r * (K / 2) + (size_t) b * (BL / 2);
        const sycl::half * dr = d + (size_t) r * (K / 32) + (size_t) b * EW;
        float reg[EW];
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            const uint8_t byte = p[i * 16 + (lane & 15)];
            reg[i]             = (float) ((int) (lane < 16 ? byte & 15 : byte >> 4) - 8) * (float) dr[i];
        }
        fwht(reg, sg, lane);
        if (out == nullptr) {
            float m = 0.0f;
#pragma unroll
            for (int i = 0; i < EW; ++i) {
                m = sycl::fmax(m, sycl::fabs(reg[i]));
            }
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            if (lane == 0) {
                bmax[id] = m;
            }
        } else {
            const float s  = rscale[r];
            if (rm) {
                int8_t * o = out + (size_t) r * K + (size_t) b * BL;
#pragma unroll
                for (int i = 0; i < EW; ++i) {
                    o[i * SG + lane] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -wl, wl);
                }
            } else {
                int8_t * o = out + (size_t) b * BL * N + r;
#pragma unroll
                for (int i = 0; i < EW; ++i) {
                    o[(size_t) (i * SG + lane) * N] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -wl, wl);
                }
            }
        }
    });
}

// activations [M][K] f32 -> rotated f32 in tmp and the largest value of each block
static inline void activations(const float * x, const int M, const int K, float * tmp, float * bmax, dpct::queue_ptr stream) {
    const int nblk = K / BL;
    stream->parallel_for(sycl::nd_range<2>({ (size_t) M * nblk, SG }, { 4, SG }),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto    sg   = it.get_sub_group();
        const int     lane = sg.get_local_linear_id();
        const size_t  id   = it.get_global_id(0);
        const float * p    = x + id * BL;
        float *       o    = tmp + id * BL;
        float reg[EW];
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            reg[i] = p[i * SG + lane];
        }
        fwht(reg, sg, lane);
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            m = sycl::fmax(m, sycl::fabs(reg[i]));
            if (tmp != nullptr) {
                o[i * SG + lane] = reg[i];
            }
        }
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        if (lane == 0) {
            bmax[id] = m;
        }
    });
}

// activations [M][K] f32 -> rotated and rounded to int8 with the token scales `to`. With activations(x, .., nullptr, ..)
// before it for the scales, the values are rotated twice but never written out as f32: 9 bytes of memory traffic a
// value against 13.
static inline void activations_q(const float * x, const int M, const int K, const float * to, int8_t * out, dpct::queue_ptr stream) {
    const int   nblk = K / BL;
    const float al   = 127.0f;
    stream->parallel_for(sycl::nd_range<2>({ (size_t) M * nblk, SG }, { 4, SG }),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto    sg   = it.get_sub_group();
        const int     lane = sg.get_local_linear_id();
        const size_t  id   = it.get_global_id(0);
        const float * p    = x + id * BL;
        int8_t *      o    = out + id * BL;
        const float   s    = to[id / nblk];
        float reg[EW];
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            reg[i] = p[i * SG + lane];
        }
        fwht(reg, sg, lane);
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            o[i * SG + lane] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -al, al);
        }
    });
}

// activations [M][K] f32 -> rotated int8 and the token scales in one pass: a work-group holds one token's row, each of
// its sub-groups one block of it in registers; the largest value of the row goes round the work-group, then every
// sub-group rounds what it holds. The values are read once (5 bytes of memory traffic a value against 9).
static inline void activations_row(const float * x, const int M, const int K, int8_t * out, float * back, dpct::queue_ptr stream) {
    const int   nblk = K / BL;
    const float al   = 127.0f, clip = 1.0f;
    stream->parallel_for(sycl::nd_range<2>({ (size_t) M, (size_t) nblk * SG }, { 1, (size_t) nblk * SG }),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto    sg   = it.get_sub_group();
        const int     lane = sg.get_local_linear_id();
        const size_t  m    = it.get_global_id(0);
        const size_t  b    = it.get_local_id(1) / SG;
        const float * p    = x + (m * nblk + b) * BL;
        int8_t *      o    = out + (m * nblk + b) * BL;
        float reg[EW];
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            reg[i] = p[i * SG + lane];
        }
        fwht(reg, sg, lane);
        float mx = 0.0f;
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            mx = sycl::fmax(mx, sycl::fabs(reg[i]));
        }
        const float row = sycl::fmax(sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>()), 1e-30f) * clip;
        const float s   = al / row;
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            o[i * SG + lane] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -al, al);
        }
        if (it.get_local_id(1) == 0) {
            back[m] = row / al;
        }
    });
}

// to[i] = 127 / (largest value of row i), back[i] = its inverse times post
static inline void row_scales(const float * bmax, const int rows, const int nblk, float * to, float * back, const float post,
                              const float lvl, const float clip, dpct::queue_ptr stream) {
    stream->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> i) {
        float m = 1e-30f;
        for (int b = 0; b < nblk; ++b) {
            m = sycl::fmax(m, bmax[i[0] * nblk + b]);
        }
        m *= clip;
        to[i[0]]   = lvl / m;
        back[i[0]] = m / lvl * post;
    });
}

struct tensor_state {
    float *  to    = nullptr;  // [N] 127 / row max
    float *  back  = nullptr;  // [N] row max / 127 / 1024
    int8_t * w8    = nullptr;  // [K][N] when resident
    bool     declined = false; // not Q4_0 and no room to keep it: the caller's other path takes it
    size_t   w8_bytes = 0;     // size of w8
    bool     other    = false; // converted from a type other than Q4_0 (counted in resident_other)
};

struct device_state {
    std::unordered_map<const void *, tensor_state> tensors;
    int8_t * stage       = nullptr;
    size_t   stage_size  = 0;
    size_t   resident    = 0;  // bytes of resident int8 weights converted from Q4_0
    size_t   resident_other = 0;  // and from the other types, which are only taken when they can be kept
    bool     keep_off = false;    // the device ran short of memory once: nothing is kept any more (release_kept)
    // the converted activations of the last call: the multiplies that read one tensor (query / key / value, gate / up)
    // follow each other and share them
    uint64_t            epoch   = 0;        // graph computations so far
    const ggml_tensor * act_id  = nullptr;
    uint64_t            act_epoch = 0;
    int                 act_m = 0, act_k = 0;
    int8_t *            a8 = nullptr;
    float *             aback = nullptr;
    float *             ato = nullptr;    // scratch of the conversion, kept so that it needs no wait for a release
    float *             abmax = nullptr;
    size_t              a8_size = 0, aback_size = 0, abmax_size = 0;
    // an ADD that follows the matmul in the graph, offered by the graph loop: the pass that applies the scales adds
    // add_other and writes to add_out, so the ADD is not run on its own
    const float * add_other = nullptr;
    float *       add_out   = nullptr;
    bool          add_done  = false;
    // the same for a SwiGLU that follows a gate and an up matmul: po_swish makes this matmul write silu(result),
    // po_mul multiplies by another tensor, po_out is where the result goes (nullptr: the matmul's own output)
    bool          po_swish = false;
    const float * po_mul   = nullptr;
    float *       po_out   = nullptr;
    bool          po_done  = false;
};

static inline device_state & state(int device) {
    static device_state st[GGML_SYCL_MAX_DEVICES];
    return st[device];
}

// g_ggml_sycl_free_hook: a device buffer is freed. What is kept for the weights in it goes, as the tensors are keyed by
// their address and the next buffer may get the same one. size 0: everything kept for the device.
static inline void forget_range(int device, const void * ptr, size_t size) {
    device_state & st = state(device);
    if (st.tensors.empty()) {
        return;
    }
    auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
    bool waited = false;
    for (auto it = st.tensors.begin(); it != st.tensors.end();) {
        const char * key = (const char *) it->first;
        if (size != 0 && (key < (const char *) ptr || key >= (const char *) ptr + size)) {
            ++it;
            continue;
        }
        if (!waited) {
            q.wait();
            waited = true;
        }
        tensor_state & ts = it->second;
        if (ts.w8 != nullptr) {
            (ts.other ? st.resident_other : st.resident) -= ts.w8_bytes;
            sycl::free(ts.w8, q);
        }
        if (ts.to != nullptr) {
            sycl::free(ts.to, q);
        }
        if (ts.back != nullptr) {
            sycl::free(ts.back, q);
        }
        it = st.tensors.erase(it);
    }
    st.act_id = nullptr;
}

// g_ggml_sycl_low_mem_hook: a device buffer did not fit. The converted weights go, and none are kept again in this
// process: the Q4_0 ones are converted on every pass from then on, the other types take their usual path.
static inline bool release_kept(int device) {
    device_state & st = state(device);
    auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
    q.wait();
    size_t freed = 0;
    for (auto & e : st.tensors) {
        if (e.second.w8 != nullptr) {
            sycl::free(e.second.w8, q);
            e.second.w8       = nullptr;
            e.second.w8_bytes = 0;
            freed++;
        }
    }
    st.resident = st.resident_other = 0;
    st.keep_off = true;
    if (freed > 0) {
        GGML_LOG_WARN("%s: the device ran short of memory: the %zu converted weight tensors kept on it were freed and none are kept from now on\n",
                      __func__, freed);
    }
    return freed > 0;
}

// Converted weights are kept only in device memory that is really free. Past its own memory the driver backs device
// allocations with host memory, which is slower than not keeping them and can take all the host has.
static inline size_t env_mib(const char * name, const long long def) {
    return (size_t) (std::getenv(name) ? std::atoll(std::getenv(name)) : def) << 20;
}

// true if `bytes` more can be kept and GGML_SYCL_ROT8_RESERVE_MIB (default 3000) stay free for the working memory
static inline bool room_for(const int device, const size_t bytes) {
    static const size_t reserve = env_mib("GGML_SYCL_ROT8_RESERVE_MIB", 3000);
    size_t free_b = 0, total_b = 0;
    ggml_backend_sycl_get_device_memory(device, &free_b, &total_b);
    return free_b >= bytes + reserve;
}

// the most Q4_0 weights kept: GGML_SYCL_ROT8_RESIDENT_MIB if set, else what is free at the first call less the reserve
// and less GGML_SYCL_ROT8_OTHER_MIB (default 2400) for the weights of other types, which are only taken when kept
static inline size_t q4_budget(const int device) {
    static size_t budget[GGML_SYCL_MAX_DEVICES];
    static bool   known[GGML_SYCL_MAX_DEVICES];
    if (!known[device]) {
        known[device] = true;
        size_t free_b = 0, total_b = 0;
        ggml_backend_sycl_get_device_memory(device, &free_b, &total_b);
        const size_t keep_free = env_mib("GGML_SYCL_ROT8_RESERVE_MIB", 3000) + env_mib("GGML_SYCL_ROT8_OTHER_MIB", 2400);
        const size_t fits      = free_b > keep_free ? free_b - keep_free : 0;
        budget[device]         = std::getenv("GGML_SYCL_ROT8_RESIDENT_MIB") ? std::min(fits, env_mib("GGML_SYCL_ROT8_RESIDENT_MIB", 0)) : fits;
        GGML_LOG_INFO("%s: %.0f MiB of the device free, up to %.0f MiB of converted Q4_0 weights are kept\n", __func__, free_b / 1048576.0,
                      budget[device] / 1048576.0);
    }
    return budget[device];
}

static inline void offer_add(const int device, const float * other, float * out) {
    device_state & st = state(device);
    st.add_other = other; st.add_out = out; st.add_done = false;
}

// true if the matmul just computed also did the offered ADD; the offer ends either way
static inline bool take_add(const int device) {
    device_state & st = state(device);
    const bool done = st.add_done;
    st.add_other = nullptr; st.add_out = nullptr; st.add_done = false;
    return done;
}

static inline void offer_glu(const int device, const bool swish, const float * mul, float * out) {
    device_state & st = state(device);
    st.po_swish = swish; st.po_mul = mul; st.po_out = out; st.po_done = false;
}

static inline bool take_glu(const int device) {
    device_state & st = state(device);
    const bool done = st.po_done;
    st.po_swish = false; st.po_mul = nullptr; st.po_out = nullptr; st.po_done = false;
    return done;
}

// a graph computation starts: tensors of the last one may hold other values now
static inline void new_graph(int device) {
    state(device).epoch++;
}

// the shapes the conversion kernels take: they launch work-groups of four blocks of 1024 values, of the weights always
// and of the activations when a token's row is too long for one work-group
static inline bool shape_ok(const int M, const int K, const int N) {
    const size_t nblk = (size_t) K / BL;
    return (N * nblk) % 4 == 0 && (nblk * SG <= 1024 || (M * nblk) % 4 == 0);
}

// the part every weight type shares: dst[M][N] = x[M][K] * w8[K][N] with the scales of both sides
static inline bool run(ggml_backend_sycl_context & ctx, device_state & st, const tensor_state & ts, const int8_t * w8, const float * x,
                       const ggml_tensor * x_id, float * dst, const int M, const int K, const int N, dpct::queue_ptr stream) {
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const int    nblk = K / BL;
    static const bool share = std::getenv("GGML_SYCL_ROT8_SHARE_ACT") == nullptr || std::atoi(std::getenv("GGML_SYCL_ROT8_SHARE_ACT")) != 0;
    if (!(share && x_id != nullptr && st.act_id == x_id && st.act_epoch == st.epoch && st.act_m == M && st.act_k == K)) {
        if (st.a8_size < (size_t) M * K || st.aback_size < (size_t) M || st.abmax_size < (size_t) M * nblk) {
            stream->wait();
            if (st.a8 != nullptr) {
                sycl::free(st.a8, *stream);
                sycl::free(st.aback, *stream);
                sycl::free(st.ato, *stream);
                sycl::free(st.abmax, *stream);
            }
            const size_t n_a8 = std::max(st.a8_size, (size_t) M * K), n_m = std::max(st.aback_size, (size_t) M);
            const size_t n_bm = std::max(st.abmax_size, (size_t) M * nblk);
            st.a8         = (int8_t *) sycl::aligned_alloc_device(4096, n_a8, *stream);
            st.aback      = sycl::malloc_device<float>(n_m, *stream);
            st.ato        = sycl::malloc_device<float>(n_m, *stream);
            st.abmax      = sycl::malloc_device<float>(n_bm, *stream);
            const bool ok = st.a8 != nullptr && st.aback != nullptr && st.ato != nullptr && st.abmax != nullptr;
            st.a8_size    = ok ? n_a8 : 0;
            st.aback_size = ok ? n_m : 0;
            st.abmax_size = ok ? n_bm : 0;
            st.act_id     = nullptr;
            if (!ok) {
                return false;
            }
        }
        float *  ato_p   = st.ato;
        float *  aback_w = st.aback;
        int8_t * a8_w    = st.a8;
        // GGML_SYCL_ROT8_ACT_1PASS=0: the largest values in one pass and the rounding in a second
        static const bool one_pass = std::getenv("GGML_SYCL_ROT8_ACT_1PASS") == nullptr || std::atoi(std::getenv("GGML_SYCL_ROT8_ACT_1PASS")) != 0;
        if (one_pass && nblk * SG <= 1024) {
            activations_row(x, M, K, a8_w, aback_w, stream);
        } else {
            activations(x, M, K, nullptr, st.abmax, stream);
            row_scales(st.abmax, M, nblk, ato_p, aback_w, 1.0f, 127.0f, 1.0f, stream);
            activations_q(x, M, K, ato_p, a8_w, stream);
        }
        st.act_id = x_id; st.act_epoch = st.epoch; st.act_m = M; st.act_k = K;
    }
    const int8_t * a8_p    = st.a8;
    const float *  aback_p = st.aback;


    auto dstream = ctx.stream_dnnl(stream);
    auto eng     = ctx.engine_dnnl(stream);
    const auto a_md = dnnl::memory::desc({ M, K }, dt::s8, tag::ab);
    const auto w_md = dnnl::memory::desc({ K, N }, dt::s8, row_major() ? tag::ba : tag::ab);
    const auto c_md = dnnl::memory::desc({ M, N }, dt::f32, tag::ab);
    const float * wback = ts.back;
    // the ADD is taken here only if its output is not also its other operand
    const bool    add   = st.add_other != nullptr && st.add_out != nullptr && (const float *) st.add_out != st.add_other;
    struct prim_t { dnnl::engine eng; dnnl::matmul::primitive_desc pd; dnnl::matmul prim; bool ok; };
    static std::unordered_map<std::string, prim_t> cache;

    // GGML_SYCL_ROT8_POST_OPS=0: a pass of its own applies the scales (and the ADD). Default: the matmul applies them,
    // the column scales as the scales of its weights and the token scales and the ADD as post-ops.
    static const bool post_on = std::getenv("GGML_SYCL_ROT8_POST_OPS") == nullptr || std::atoi(std::getenv("GGML_SYCL_ROT8_POST_OPS")) != 0;
    const bool    glu   = st.po_swish || st.po_mul != nullptr;
    if (post_on) {
        const auto s_md = dnnl::memory::desc({ N }, dt::f32, tag::a);
        const auto r_md = dnnl::memory::desc({ M, 1 }, dt::f32, tag::ab);
        char key[112];
        snprintf(key, sizeof(key), "p%d%d%d:%p:%d:%d:%d", (int) (add && !glu), (int) st.po_swish, (int) (st.po_mul != nullptr), (void *) eng.get(), M, K, N);
        auto it = cache.find(key);
        if (it == cache.end()) {
            prim_t pr{ eng, {}, {}, false };
            try {
                dnnl::primitive_attr attr;
                attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
                dnnl::post_ops po;
                po.append_binary(dnnl::algorithm::binary_mul, r_md);
                if (st.po_swish) {
                    po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
                }
                if (st.po_mul != nullptr) {
                    po.append_binary(dnnl::algorithm::binary_mul, c_md);
                }
                if (add && !glu) {
                    po.append_binary(dnnl::algorithm::binary_add, c_md);
                }
                attr.set_post_ops(po);
                pr.pd   = dnnl::matmul::primitive_desc(eng, a_md, w_md, c_md, attr);
                pr.prim = dnnl::matmul(pr.pd);
                pr.ok   = true;
                GGML_LOG_INFO("%s: %d x %d x %d%s%s%s with the scales in the matmul: %s\n", __func__, M, K, N, add && !glu ? " + add" : "",
                              st.po_swish ? " + silu" : "", st.po_mul != nullptr ? " * other" : "", pr.pd.impl_info_str());
            } catch (const dnnl::error & e) {
                GGML_LOG_WARN("%s: %d x %d x %d: no matmul with the scales in it (%s), a pass of its own applies them\n", __func__, M, K, N, e.what());
            }
            it = cache.emplace(key, pr).first;
        }
        if (it->second.ok) {
            const auto scratchpad_md = it->second.pd.scratchpad_desc();
            ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
            void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, dnnl::memory(a_md, eng, const_cast<int8_t *>(a8_p)) });
            args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<int8_t *>(w8)) });
            float * out = glu ? (st.po_out != nullptr ? st.po_out : dst) : (add ? st.add_out : dst);
            args.insert({ DNNL_ARG_DST, dnnl::memory(c_md, eng, out) });
            args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(scratchpad_md, eng, scratchpad_ptr) });
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(s_md, eng, const_cast<float *>(wback)) });
            args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, dnnl::memory(r_md, eng, const_cast<float *>(aback_p)) });
            int po_i = 1;
            if (st.po_swish) {
                po_i++;
            }
            if (st.po_mul != nullptr) {
                args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i) | DNNL_ARG_SRC_1, dnnl::memory(c_md, eng, const_cast<float *>(st.po_mul)) });
                po_i++;
            }
            if (add && !glu) {
                args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i) | DNNL_ARG_SRC_1, dnnl::memory(c_md, eng, const_cast<float *>(st.add_other)) });
            }
            it->second.prim.execute(dstream, args);
            if (glu) {
                st.po_done = true;
            } else if (add) {
                st.add_done = true;
            }
            return true;
        }
    }

    char key[96];
    snprintf(key, sizeof(key), "%p:%d:%d:%d", (void *) eng.get(), M, K, N);
    auto it = cache.find(key);
    if (it == cache.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, w_md, c_md, attr);
        it      = cache.emplace(key, prim_t{ eng, pd, dnnl::matmul(pd), true }).first;
    }
    const auto scratchpad_md = it->second.pd.scratchpad_desc();
    ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
    void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, dnnl::memory(a_md, eng, const_cast<int8_t *>(a8_p)) });
    args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<int8_t *>(w8)) });
    args.insert({ DNNL_ARG_DST, dnnl::memory(c_md, eng, dst) });
    args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(scratchpad_md, eng, scratchpad_ptr) });
    it->second.prim.execute(dstream, args);

    if (st.add_other != nullptr && st.add_out != nullptr) {
        const float * other = st.add_other;
        float *       out   = st.add_out;
        stream->parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> i) {
            const size_t j = i[0] * N + i[1];
            out[j] = dst[j] * (aback_p[i[0]] * wback[i[1]]) + other[j];
        });
        st.add_done = true;
    } else {
        stream->parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> i) { dst[i[0] * N + i[1]] *= aback_p[i[0]] * wback[i[1]]; });
    }
    return true;
}

// dst[M][N] (f32) = x[M][K] (f32) * W[N][K]^T for reordered Q4_0 weights. false: not taken, nothing written.
// x_id: the tensor x is the data of (nullptr: unknown), to share its converted form between multiplies
static inline bool mul_mat(ggml_backend_sycl_context & ctx, const void * w_reordered, const float * x, const ggml_tensor * x_id, float * dst,
                           const int M, const int K, const int N, dpct::queue_ptr stream) {
    // (the conversion kernels run in work-groups of four blocks)
    if (K % BL != 0 || !shape_ok(M, K, N)) {
        return false;
    }
    const int      device = ggml_sycl_get_device();
    device_state & st     = state(device);
    const int      nblk   = K / BL;
    const uint8_t *    qs = (const uint8_t *) w_reordered;
    const sycl::half * d  = (const sycl::half *) (qs + (size_t) N * K / 2);

    tensor_state & ts = st.tensors[w_reordered];
    if (ts.to == nullptr) {
        ts.to   = sycl::malloc_device<float>(N, *stream);
        ts.back = sycl::malloc_device<float>(N, *stream);
        ggml_sycl_pool_alloc<float> bmax(ctx.pool(), (size_t) N * nblk);
        weights(qs, d, N, K, bmax.get(), nullptr, nullptr, stream);
        row_scales(bmax.get(), N, nblk, ts.to, ts.back, 1.0f / BL, 127.0f, 1.0f, stream);
        stream->wait();
        g_ggml_sycl_low_mem_hook = release_kept;
        g_ggml_sycl_free_hook    = forget_range;
        if (!st.keep_off && st.resident + (size_t) N * K <= q4_budget(device) && room_for(device, (size_t) N * K)) {
            ts.w8 = (int8_t *) sycl::aligned_alloc_device(4096, (size_t) N * K, *stream);
            if (ts.w8 != nullptr) {
                weights(qs, d, N, K, nullptr, ts.to, ts.w8, stream);
                ts.w8_bytes = (size_t) N * K;
                st.resident += (size_t) N * K;
            }
        }
    }
    const int8_t * w8 = ts.w8;
    if (w8 == nullptr) {
        if (st.stage_size < (size_t) N * K) {
            stream->wait();
            if (st.stage != nullptr) {
                sycl::free(st.stage, *stream);
            }
            st.stage      = (int8_t *) sycl::aligned_alloc_device(4096, (size_t) N * K, *stream);
            st.stage_size = st.stage != nullptr ? (size_t) N * K : 0;
            if (st.stage == nullptr) {
                return false;
            }
        }
        weights(qs, d, N, K, nullptr, ts.to, st.stage, stream);
        w8 = st.stage;
    }
    return run(ctx, st, ts, w8, x, x_id, dst, M, K, N, stream);
}

// Weights of any other type are taken only when their converted form can stay on the device (the conversion goes
// through the f16 values and is too slow to repeat every pass). wanted(): should the caller hand the f16 values to keep()?
static inline bool kept(const void * key) {
    const device_state & st = state(ggml_sycl_get_device());
    const auto it = st.tensors.find(key);
    return it != st.tensors.end() && it->second.w8 != nullptr;
}

static inline bool wanted(const void * key, const int K) {
    if (K % BL != 0) {
        return false;
    }
    const device_state & st = state(ggml_sycl_get_device());
    if (st.keep_off) {
        return false;
    }
    const auto it = st.tensors.find(key);
    return it == st.tensors.end() || (it->second.w8 == nullptr && !it->second.declined);
}

// w16: the weights as f16 values [N][K]
static inline void keep(ggml_backend_sycl_context & ctx, const void * key, const sycl::half * w16, const int K, const int N, dpct::queue_ptr stream) {
    device_state & st = state(ggml_sycl_get_device());
    tensor_state & ts = st.tensors[key];
    static const size_t budget = env_mib("GGML_SYCL_ROT8_OTHER_MIB", 2400);
    ts.declined = true;
    // (the conversion needs two f32 copies of the weights for a moment)
    // (their room was set aside when the Q4_0 budget was fixed, see q4_budget)
    g_ggml_sycl_low_mem_hook = release_kept;
    g_ggml_sycl_free_hook    = forget_range;
    if (K % BL != 0 || !shape_ok(4, K, N) || st.keep_off || st.resident_other + (size_t) N * K > budget) {
        return;
    }
    const int nblk = K / BL;
    float *  to   = sycl::malloc_device<float>(N, *stream);
    float *  back = sycl::malloc_device<float>(N, *stream);
    int8_t * w8   = (int8_t *) sycl::aligned_alloc_device(4096, (size_t) N * K, *stream);
    if (to == nullptr || back == nullptr || w8 == nullptr) {
        if (to)   { sycl::free(to, *stream); }
        if (back) { sycl::free(back, *stream); }
        if (w8)   { sycl::free(w8, *stream); }
        return;
    }
    {
        ggml_sycl_pool_alloc<float> wf(ctx.pool(), (size_t) N * K);
        ggml_sycl_pool_alloc<float> rot(ctx.pool(), (size_t) N * K);
        ggml_sycl_pool_alloc<float> bmax(ctx.pool(), (size_t) N * nblk);
        float * wf_p  = wf.get();
        float * rot_p = rot.get();
        stream->parallel_for(sycl::range<1>((size_t) N * K), [=](sycl::id<1> i) { wf_p[i] = (float) w16[i]; });
        activations(wf_p, N, K, rot_p, bmax.get(), stream);
        row_scales(bmax.get(), N, nblk, to, back, 1.0f / BL, 127.0f, 1.0f, stream);
        const bool  rm = row_major();
        const float wl = 127.0f;
        stream->parallel_for(sycl::range<2>(K, N), [=](sycl::id<2> i) {
            w8[rm ? i[1] * K + i[0] : i[0] * N + i[1]] = (int8_t) sycl::clamp(sycl::rint(rot_p[i[1] * K + i[0]] * to[i[1]]), -wl, wl);
        });
        stream->wait(); // the scratch is released here
    }
    ts.to = to; ts.back = back; ts.w8 = w8; ts.declined = false; ts.w8_bytes = (size_t) N * K; ts.other = true;
    st.resident_other += (size_t) N * K;
}

// f16 weights [N][K] (row major): the same rotation as weights() for Q4_0, value i * SG + lane of a block in reg[i].
// out == nullptr: the largest rotated value of each block to bmax; else int8 with the row scales in the matmul's layout.
static inline void weights_f16(const sycl::half * w16, const int N, const int K, float * bmax, const float * rscale, int8_t * out,
                               dpct::queue_ptr stream) {
    const int   nblk = K / BL;
    const bool  rm   = row_major();
    const float wl   = 127.0f;
    stream->parallel_for(sycl::nd_range<2>({ (size_t) N * nblk, SG }, { 4, SG }),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto   sg   = it.get_sub_group();
        const int    lane = sg.get_local_linear_id();
        const size_t id   = it.get_global_id(0);
        const int    r    = (int) (id / nblk);
        const int    b    = (int) (id % nblk);
        const sycl::half * p = w16 + (size_t) r * K + (size_t) b * BL;
        float reg[EW];
#pragma unroll
        for (int i = 0; i < EW; ++i) {
            reg[i] = (float) p[i * SG + lane];
        }
        fwht(reg, sg, lane);
        if (out == nullptr) {
            float m = 0.0f;
#pragma unroll
            for (int i = 0; i < EW; ++i) {
                m = sycl::fmax(m, sycl::fabs(reg[i]));
            }
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            if (lane == 0) {
                bmax[id] = m;
            }
        } else {
            const float s = rscale[r];
            if (rm) {
                int8_t * o = out + (size_t) r * K + (size_t) b * BL;
#pragma unroll
                for (int i = 0; i < EW; ++i) {
                    o[i * SG + lane] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -wl, wl);
                }
            } else {
                int8_t * o = out + (size_t) b * BL * N + r;
#pragma unroll
                for (int i = 0; i < EW; ++i) {
                    o[(size_t) (i * SG + lane) * N] = (int8_t) sycl::clamp(sycl::rint(reg[i] * s), -wl, wl);
                }
            }
        }
    });
}

// Weights of the other types that are not kept (no room for their int8 copy, e.g. a 27B whose copy would be ~27 GB):
// converted every pass from the f16 values the caller made into the shared scratch, as Q4_0 is. Costs a read of the f16
// weights and a write of the int8 ones per pass; the row scales are worked out once per tensor.
// GGML_SYCL_ROT8_TRANSIENT=0 turns this off (those weights then take the caller's other path).
static inline bool transient_on() {
    static const bool on = env_mib("GGML_SYCL_ROT8_TRANSIENT", 1) != 0;
    return on;
}

static inline bool mul_mat_f16(ggml_backend_sycl_context & ctx, const void * key, const sycl::half * w16, const float * x,
                               const ggml_tensor * x_id, float * dst, const int M, const int K, const int N, dpct::queue_ptr stream) {
    if (!transient_on() || K % BL != 0 || !shape_ok(M, K, N)) {
        return false;
    }
    device_state & st = state(ggml_sycl_get_device());
    tensor_state & ts = st.tensors[key];
    if (ts.w8 != nullptr) {
        return run(ctx, st, ts, ts.w8, x, x_id, dst, M, K, N, stream);
    }
    if (ts.to == nullptr) {
        const int nblk = K / BL;
        ts.to   = sycl::malloc_device<float>(N, *stream);
        ts.back = sycl::malloc_device<float>(N, *stream);
        if (ts.to == nullptr || ts.back == nullptr) {
            return false;
        }
        ggml_sycl_pool_alloc<float> bmax(ctx.pool(), (size_t) N * nblk);
        weights_f16(w16, N, K, bmax.get(), nullptr, nullptr, stream);
        row_scales(bmax.get(), N, nblk, ts.to, ts.back, 1.0f / BL, 127.0f, 1.0f, stream);
        stream->wait();
        g_ggml_sycl_low_mem_hook = release_kept;
        g_ggml_sycl_free_hook    = forget_range;
    }
    if (st.stage_size < (size_t) N * K) {
        stream->wait();
        if (st.stage != nullptr) {
            sycl::free(st.stage, *stream);
        }
        st.stage      = (int8_t *) sycl::aligned_alloc_device(4096, (size_t) N * K, *stream);
        st.stage_size = st.stage != nullptr ? (size_t) N * K : 0;
        if (st.stage == nullptr) {
            return false;
        }
    }
    weights_f16(w16, N, K, nullptr, ts.to, st.stage, stream);
    return run(ctx, st, ts, st.stage, x, x_id, dst, M, K, N, stream);
}

static inline bool mul_mat_kept(ggml_backend_sycl_context & ctx, const void * key, const float * x, const ggml_tensor * x_id, float * dst,
                                const int M, const int K, const int N, dpct::queue_ptr stream) {
    device_state & st = state(ggml_sycl_get_device());
    const auto it = st.tensors.find(key);
    if (it == st.tensors.end() || it->second.w8 == nullptr || !shape_ok(M, K, N)) {
        return false;
    }
    return run(ctx, st, it->second, it->second.w8, x, x_id, dst, M, K, N, stream);
}

}  // namespace ggml_sycl_rot8

#endif  // GGML_SYCL_DNNL
#endif  // GGML_SYCL_ROT8_HPP

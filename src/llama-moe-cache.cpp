#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mutex>
#include <string>
#endif

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token
    size_t n_skipped = 0;             // misses of the last plan() dropped by LLAMA_MOE_TAIL_SKIP

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // Local experiment, LOSSY, off by default (LLAMA_MOE_TAIL_SKIP=K): a miss that every token of the batch routes at
        // rank K or lower (0 = the strongest expert) is neither fetched nor computed, it reads as zero like a miss left
        // to the CPU but nothing computes it. The weakest experts carry the least weight and misses are what cost time
        // when the experts live off the card. Needs zero slots, so a CPU share above 0 (a tiny one is enough).
        n_skipped = 0;
        token_zero.assign(n_tokens, 0);
        {
            static const int tail_k = getenv("LLAMA_MOE_TAIL_SKIP") ? atoi(getenv("LLAMA_MOE_TAIL_SKIP")) : 0;
            if (tail_k > 0 && n_zero > 0 && (size_t) n_tokens*n_used == n_ids) {
                size_t n_keep = 0;
                for (size_t i = 0; i < misses.size(); ++i) {
                    const int32_t e = misses[i];
                    int32_t  best = n_used;
                    uint64_t used = 0;
                    for (int64_t it = 0; it < n_tokens; ++it) {
                        for (int32_t k = 0; k < n_used; ++k) {
                            if (ids[it*n_used + k] == e) {
                                best  = std::min(best, k);
                                used |= token_zero[it];
                            }
                        }
                    }
                    int32_t z = 0;
                    while (z < 64 && (used >> z & 1)) {
                        z++;
                    }
                    if (best < tail_k || z >= n_zero) {
                        misses[n_keep++] = e;
                        continue;
                    }
                    for (int64_t it = 0; it < n_tokens; ++it) {
                        for (int32_t k = 0; k < n_used; ++k) {
                            if (ids[it*n_used + k] == e) {
                                token_zero[it] |= 1ull << z;
                            }
                        }
                    }
                    slots[e] = n_slots + z;
                    n_skipped++;
                }
                misses.resize(n_keep);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

// Reads `size` bytes at `src`, an address inside a file mapping, into dst with read calls on the file itself: one request
// per range with the kernel's normal read-ahead, where copying from the mapping takes a page fault per 4 KiB because
// the lazy tensors are mapped with random-access advice. Falls back to memcpy if the mapping cannot be resolved.
static void llama_moe_cache_read_file(const uint8_t * src, uint8_t * dst, size_t size) {
#if defined(__linux__)
    struct map_entry { uintptr_t beg, end; off_t offs; int fd; };
    static std::mutex             mtx;
    static std::vector<map_entry> maps;
    auto find = [&](uintptr_t p) -> const map_entry * {
        for (const map_entry & m : maps) {
            if (p >= m.beg && p < m.end) {
                return &m;
            }
        }
        return nullptr;
    };
    map_entry m = { 0, 0, 0, -1 };
    {
        std::lock_guard<std::mutex> lock(mtx);
        const map_entry * e = find((uintptr_t) src);
        if (e == nullptr) {
            if (FILE * f = fopen("/proc/self/maps", "r")) {
                char line[4096];
                while (fgets(line, sizeof(line), f)) {
                    unsigned long beg, end, offs;
                    char          perms[8], path[3072];
                    path[0] = 0;
                    if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %3071[^\n]", &beg, &end, perms, &offs, path) >= 4 &&
                        (uintptr_t) src >= beg && (uintptr_t) src < end && path[0] == '/') {
                        const int fd = open(path, O_RDONLY);
                        if (fd >= 0) {
                            maps.push_back({ (uintptr_t) beg, (uintptr_t) end, (off_t) offs, fd });
                        }
                    }
                }
                fclose(f);
            }
            e = find((uintptr_t) src);
        }
        if (e != nullptr) {
            m = *e;
        }
    }
    if (m.fd >= 0 && (uintptr_t) src + size <= m.end) {
        off_t  offs = m.offs + (off_t) ((uintptr_t) src - m.beg);
        size_t done = 0;
        while (done < size) {
            const ssize_t n = pread(m.fd, dst + done, size - done, offs + (off_t) done);
            if (n <= 0) {
                break;
            }
            done += (size_t) n;
        }
        if (done == size) {
            return;
        }
    }
#endif
    memcpy(dst, src, size);
}

// an expert tensor that is not in RAM but a view of the model file (the disk tier, see moe_host_tier in the loader)
static bool llama_moe_cache_is_file_backed(const ggml_tensor * t) {
    return t->buffer != nullptr && strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)), "CPU_Mapped") == 0;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // Disk tier: experts of file-backed tensors are read into this pinned buffer by helper threads and uploaded from
    // it, instead of the device copying them page fault by page fault out of the file mapping.
    ggml_backend_buffer_ptr stage_buf;
    size_t                  stage_size  = 0;
    size_t                  stage_bytes = 0; // read from the file so far
    size_t                  stage_reads = 0;

    // RAM tier by usage (LLAMA_MOE_RAM_CACHE_MIB=N): the file-backed experts used most recently are kept in pinned
    // memory, one slot per (layer, expert) holding all its tensors. A device-cache miss is uploaded from its RAM slot;
    // an expert that is in neither is read from the file into the least recently used slot first.
    struct ram_tier {
        bool                                 ready      = false;
        size_t                               slot_bytes = 0; // all tensors of one expert
        std::vector<size_t>                  bank_off;   // start of each tensor's bank in the buffer: bank ip holds n_slots experts of tensor ip
        std::vector<ggml_tensor *>           bank;       // host tensors over the banks, read by the CPU part of the graph
        ggml_context_ptr                     ctx;
        int32_t                              n_slots    = 0;
        ggml_backend_buffer_ptr              buf;
        uint8_t *                            base       = nullptr;
        std::unordered_map<int64_t, int32_t> slot_of;
        std::vector<int64_t>                 key_of;
        std::vector<uint64_t>                stamp;
        std::vector<uint8_t>                 ref;        // CLOCK reference count, 0..3: bumped on a hit, aged by the hand
        uint64_t                             tick       = 0;
        int32_t                              hand       = 0;
    };
    std::vector<ram_tier> ram;       // per group
    size_t                ram_hits  = 0;
    size_t                tail_skipped = 0;
    size_t                ram_reads = 0;

    ram_tier & ram_of(ggml_backend_t backend, size_t ig) {
        if (ram.size() < groups.size()) {
            ram.resize(groups.size());
        }
        ram_tier & r = ram[ig];
        if (!r.ready) {
            r.ready = true;
            static const long long mib = getenv("LLAMA_MOE_RAM_CACHE_MIB") ? atoll(getenv("LLAMA_MOE_RAM_CACHE_MIB")) : 0;
            size_t host_total = 0;
            for (const group & gg : groups) {
                host_total += gg.host_bytes;
            }
            const group & g = groups[ig];
            for (const ggml_tensor * t : g.ref) {
                r.slot_bytes += t->nb[2];
            }
            const size_t budget = host_total > 0 ? (size_t) ((double) mib*1024*1024*g.host_bytes/host_total) : 0;
            r.n_slots = r.slot_bytes > 0 ? (int32_t) (budget/r.slot_bytes) : 0;
            // at least two layers' worth of experts: one call can then always place every expert it touches, which the
            // CPU part relies on (its ids are RAM slots, there is no fallback once the graph reads the RAM bank)
            if (r.n_slots >= 2*(int32_t) g.ref[0]->ne[2]) {
                ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
                r.buf.reset(ggml_backend_buft_alloc_buffer(host ? host : ggml_backend_cpu_buffer_type(), (size_t) r.n_slots*r.slot_bytes));
            }
            if (r.buf) {
                ggml_backend_buffer_set_usage(r.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                r.base = (uint8_t *) ggml_backend_buffer_get_base(r.buf.get());
                r.key_of.assign(r.n_slots, -1);
                r.stamp.assign(r.n_slots, 0);
                r.ref.assign(r.n_slots, 0);
                ggml_init_params params = { g.ref.size()*ggml_tensor_overhead(), nullptr, true };
                r.ctx.reset(ggml_init(params));
                size_t off = 0;
                for (const ggml_tensor * t : g.ref) {
                    r.bank_off.push_back(off);
                    ggml_tensor * b = ggml_new_tensor_3d(r.ctx.get(), t->type, t->ne[0], t->ne[1], r.n_slots);
                    b->buffer = r.buf.get();
                    b->data   = r.base + off;
                    ggml_set_name(b, "moe_ram_bank");
                    r.bank.push_back(b);
                    off += (size_t) r.n_slots*t->nb[2];
                }
                LLAMA_LOG_INFO("llama_moe_cache: RAM tier of group %zu: %d experts, %.2f MiB pinned\n", ig, r.n_slots, r.n_slots*r.slot_bytes/1024.0/1024.0);
            } else {
                r.n_slots = 0;
            }
        }
        return r;
    }

    // the RAM bank that holds the experts of host tensor w, if w is file-backed and its group has a RAM tier
    ggml_tensor * get_ram_experts(const ggml_tensor * w) {
        const auto it = bindings.find(w);
        if (it == bindings.end() || !llama_moe_cache_is_file_backed(w)) {
            return nullptr;
        }
        const int32_t ig = layers[it->second.il].ig;
        ram_tier & r = ram_of(devices[groups[ig].id].backend, ig);
        return r.n_slots > 0 ? r.bank[it->second.ip] : nullptr;
    }

    // RAM slot of (layer il, expert e): looked up, or taken from the least recently used of the next 64 slots that this
    // call does not use. read is set when the expert has to be read from the file into the slot.
    int32_t ram_slot(ram_tier & r, int32_t il, int32_t e, uint64_t call_tick, bool & read) {
        const int64_t key = ((int64_t) il << 20) | e;
        auto it = r.slot_of.find(key);
        int32_t s;
        if (it != r.slot_of.end()) {
            s    = it->second;
            read = false;
            ram_hits++;
            if (r.ref[s] < 3) {
                r.ref[s]++;
            }
        } else {
            // CLOCK: the hand ages the slots it passes and takes the first one with no references left that this call
            // does not use, so experts that keep being hit survive several passes and one-off experts go first
            s = -1;
            for (int64_t k = 0; k < 5*(int64_t) r.n_slots && s < 0; ++k) {
                const int32_t cand = r.hand;
                r.hand = (r.hand + 1) % r.n_slots;
                if (r.stamp[cand] == call_tick) {
                    continue;
                }
                if (r.ref[cand] > 0) {
                    r.ref[cand]--;
                } else {
                    s = cand;
                }
            }
            GGML_ASSERT(s >= 0 && "the RAM tier is smaller than the experts of one call");
            if (r.key_of[s] >= 0) {
                r.slot_of.erase(r.key_of[s]);
            }
            r.key_of[s]    = key;
            r.slot_of[key] = s;
            r.ref[s]       = 0;
            read = true;
        }
        r.stamp[s] = call_tick;
        return s;
    }

    // Large batches upload long runs of experts. For a file-backed tensor they go through the pinned staging buffer in
    // chunks read by helper threads, never straight from the file mapping (slow, and it lost the device on Arc cards).
    size_t stage_used = SIZE_MAX; // SIZE_MAX: the buffer may hold data of queued copies, synchronize before reuse

    bool upload_file_backed(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, size_t offset, size_t size) {
        if (!llama_moe_cache_is_file_backed(w)) {
            return false;
        }
        // experts of the run that sit in the RAM tier are uploaded from their pinned slots, the rest come from the file
        const auto it = bindings.find(w);
        if (it != bindings.end()) {
            const int32_t il = it->second.il;
            const size_t  ip = it->second.ip;
            ram_tier & r = ram_of(backend, layers[il].ig);
            const size_t expert_size = w->nb[2];
            if (r.n_slots > 0 && offset % expert_size == 0) {
                const size_t end = offset + size;
                size_t disk_beg = offset; // start of the pending run that is not in RAM
                for (size_t o = offset; o < end; o += expert_size) {
                    const size_t n = std::min(expert_size, end - o);
                    const auto s = n == expert_size ? r.slot_of.find(((int64_t) il << 20) | (int64_t) (o/expert_size)) : r.slot_of.end();
                    if (s == r.slot_of.end()) {
                        continue;
                    }
                    if (disk_beg < o && !upload_staged(backend, w, dst, disk_beg, o - disk_beg)) {
                        return false;
                    }
                    ggml_backend_tensor_set_async(backend, dst, r.base + r.bank_off[ip] + (size_t) s->second*expert_size, o, expert_size);
                    ram_hits++;
                    disk_beg = o + expert_size;
                }
                return disk_beg >= end || upload_staged(backend, w, dst, disk_beg, end - disk_beg);
            }
        }
        return upload_staged(backend, w, dst, offset, size);
    }

    bool upload_staged(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, size_t offset, size_t size) {
        constexpr size_t arena = 256ull*1024*1024;
        constexpr size_t chunk = 32ull*1024*1024;
        uint8_t * base = stage_reserve(backend, arena);
        if (base == nullptr) {
            return false;
        }
        const uint8_t * src = (const uint8_t *) w->data + offset;
        for (size_t done = 0; done < size; ) {
            const size_t n = std::min(chunk, size - done);
            if (stage_used == SIZE_MAX || stage_used + n > arena) {
                ggml_backend_synchronize(backend);
                stage_used = 0;
            }
            uint8_t * d = base + stage_used;
            const size_t n_thr = n >= 8*1024*1024 ? 8 : 1;
            const size_t part  = (n + n_thr - 1)/n_thr;
            std::vector<std::thread> thr;
            for (size_t t = 1; t < n_thr; ++t) {
                const size_t o = t*part;
                if (o < n) {
                    thr.emplace_back([=] { llama_moe_cache_read_file(src + done + o, d + o, std::min(part, n - o)); });
                }
            }
            llama_moe_cache_read_file(src + done, d, std::min(part, n));
            for (auto & t : thr) {
                t.join();
            }
            ggml_backend_tensor_set_async(backend, dst, d, offset + done, n);
            stage_used  += n;
            stage_bytes += n;
            done        += n;
        }
        stage_reads++;
        return true;
    }

    uint8_t * stage_reserve(ggml_backend_t backend, size_t size) {
        if (size > stage_size) {
            ggml_backend_synchronize(backend); // copies out of the old buffer may still be queued
            const size_t want = size + size / 2;
            ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            stage_buf.reset(ggml_backend_buft_alloc_buffer(host ? host : ggml_backend_cpu_buffer_type(), want));
            stage_size = stage_buf ? want : 0;
        }
        return stage_buf ? (uint8_t *) ggml_backend_buffer_get_base(stage_buf.get()) : nullptr;
    }

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        usage_save();
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    struct read_job { const uint8_t * src; uint8_t * dst; size_t size; };

    // read experts from the file into pinned memory with helper threads, one request per expert tensor
    void run_reads(const std::vector<read_job> & jobs) {
        if (jobs.empty()) {
            return;
        }
        const size_t n_thr = std::min<size_t>(8, jobs.size());
        std::vector<std::thread> thr;
        for (size_t t = 1; t < n_thr; ++t) {
            thr.emplace_back([&jobs, t, n_thr] {
                for (size_t j = t; j < jobs.size(); j += n_thr) {
                    llama_moe_cache_read_file(jobs[j].src, jobs[j].dst, jobs[j].size);
                }
            });
        }
        for (size_t j = 0; j < jobs.size(); j += n_thr) {
            llama_moe_cache_read_file(jobs[j].src, jobs[j].dst, jobs[j].size);
        }
        for (auto & t : thr) {
            t.join();
        }
        for (const read_job & j : jobs) {
            stage_bytes += j.size;
        }
        stage_reads += jobs.size();
        ram_reads   += jobs.size();
    }

    // upload the experts of layer il that the last plan() put in fills (and stage those it left to the CPU), returns the bytes uploaded
    size_t upload_fills(ggml_backend_t backend, int32_t il, bool has_cpu) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        // Where each missing expert of each tensor is uploaded from: the host tensor itself (pinned experts), a slot of
        // the RAM tier, or the staging buffer. File-backed experts are read by helper threads, one request per expert.
        // The misses that the CPU computes are read from the RAM tier too, their cpu_map entries are the RAM slots.
        std::vector<read_job> jobs;
        std::vector<std::vector<const uint8_t *>> srcs(l.experts.size());
        {
            std::vector<size_t> file_ips;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                srcs[ip].resize(fills.size());
                for (size_t i = 0; i < fills.size(); ++i) {
                    srcs[ip][i] = (const uint8_t *) w->data + fills[i].expert*w->nb[2];
                }
                if (llama_moe_cache_is_file_backed(w)) {
                    file_ips.push_back(ip);
                }
            }
            if (!file_ips.empty() && (!fills.empty() || !cpu.empty())) {
                ram_tier & r = ram_of(backend, l.ig);
                const bool use_ram = r.n_slots > 0; // a call touches at most one layer's experts, the tier holds two or more
                uint8_t * stage = nullptr;
                size_t    stage_offs = 0;
                if (!use_ram && !fills.empty()) {
                    size_t need = 0;
                    for (size_t ip : file_ips) {
                        need += fills.size()*l.experts[ip]->nb[2];
                    }
                    stage = stage_reserve(backend, need);
                }
                const uint64_t call_tick = ++r.tick;
                auto queue_read = [&](size_t ip, int32_t expert, uint8_t * dst) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t expert_size = w->nb[2];
                    const uint8_t * src = (const uint8_t *) w->data + expert*expert_size;
#if defined(__linux__)
                    const uintptr_t beg = (uintptr_t) src & ~(uintptr_t) 4095;
                    madvise((void *) beg, (uintptr_t) src + expert_size - beg, MADV_WILLNEED);
#endif
                    jobs.push_back({ src, dst, expert_size });
                };
                for (size_t i = 0; i < fills.size(); ++i) {
                    bool    read = true;
                    int32_t s    = use_ram ? ram_slot(r, il, fills[i].expert, call_tick, read) : -1;
                    for (size_t ip : file_ips) {
                        const size_t expert_size = l.experts[ip]->nb[2];
                        uint8_t * dst = use_ram ? r.base + r.bank_off[ip] + (size_t) s*expert_size : (stage ? stage + stage_offs : nullptr);
                        if (dst == nullptr) {
                            continue; // no pinned memory: upload from the file mapping as before
                        }
                        if (read) {
                            queue_read(ip, fills[i].expert, dst);
                        }
                        srcs[ip][i] = dst;
                        stage_offs += use_ram ? 0 : expert_size;
                    }
                }
                if (use_ram && has_cpu && file_ips.size() == l.experts.size()) {
                    int32_t * cpu_map = (int32_t *) l.cpu_map->data;
                    for (const auto & f : cpu) {
                        bool          read = true;
                        const int32_t s    = ram_slot(r, il, f.expert, call_tick, read);
                        if (read) {
                            for (size_t ip : file_ips) {
                                queue_read(ip, f.expert, r.base + r.bank_off[ip] + (size_t) s*l.experts[ip]->nb[2]);
                            }
                        }
                        cpu_map[f.expert] = s; // the CPU part reads the RAM bank, see get_ram_experts()
                    }
                }
            }
            run_reads(jobs);
        }

        stage_used = SIZE_MAX; // the staging buffer may now be read by queued copies

        // upload the missing experts, consecutive experts going to consecutive slots from consecutive memory are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n &&
                       srcs[ip][i + n] == srcs[ip][i] + n*expert_size) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, srcs[ip][i], fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        return bytes;
    }

    // Usage memory, local, off by default (LLAMA_MOE_USAGE_FILE=path): on shutdown the experts held in the device cache
    // and in the RAM tier are written to the file, coldest first per layer. The next start reads the file and, at the
    // first slot lookup on each device, loads those experts back before the first token instead of one miss at a time.
    // It changes which requests are fast, never the output. Keep it off for speed measurements: a warmed cache hides
    // the cold start.
    struct usage_layer {
        std::vector<int32_t> vram; // coldest first
        std::vector<int32_t> ram;
    };
    std::vector<usage_layer> usage;        // [n_layer], empty if there is nothing to load
    std::vector<uint8_t>     usage_warmed; // [n_device]
    bool                     usage_read = false;

    static const char * usage_path() {
        static const char * path = getenv("LLAMA_MOE_USAGE_FILE");
        return path != nullptr && path[0] != '\0' ? path : nullptr;
    }

    // identifies the model and its split over the devices: expert bytes of each group
    std::string usage_signature() const {
        std::string sig = std::to_string(layers.size());
        for (const group & g : groups) {
            sig += "," + std::to_string(g.id) + ":" + std::to_string(g.layers.size()) + ":" + std::to_string(g.host_bytes);
        }
        return sig;
    }

    void usage_load() {
        usage_read = true;
        FILE * f = usage_path() ? fopen(usage_path(), "r") : nullptr;
        if (f == nullptr) {
            return;
        }
        char sig[1024] = "";
        std::vector<usage_layer> res(layers.size());
        bool ok = fscanf(f, "moe-usage 1 %1023s", sig) == 1 && usage_signature() == sig;
        int  il;
        char tier;
        int  n;
        while (ok && fscanf(f, " L %d %c %d", &il, &tier, &n) == 3) {
            ok = il >= 0 && il < (int) layers.size() && layers[il].ig >= 0 && (tier == 'V' || tier == 'R') && n >= 0;
            const int n_expert = ok ? (int) groups[layers[il].ig].ref[0]->ne[2] : 0;
            for (int i = 0; ok && i < n; ++i) {
                int e;
                ok = fscanf(f, "%d", &e) == 1 && e >= 0 && e < n_expert;
                if (ok) {
                    (tier == 'V' ? res[il].vram : res[il].ram).push_back(e);
                }
            }
        }
        fclose(f);
        if (!ok) {
            LLAMA_LOG_WARN("llama_moe_cache: usage memory: %s is for another model or setup, ignored\n", usage_path());
            return;
        }
        usage = std::move(res);
    }

    void usage_save() const {
        if (usage_path() == nullptr || stats_small.hits + stats_small.misses + stats_large.hits + stats_large.misses == 0) {
            return; // nothing was generated: keep the previous file
        }
        const std::string tmp = std::string(usage_path()) + ".tmp";
        FILE * f = fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache: usage memory: cannot write %s\n", tmp.c_str());
            return;
        }
        fprintf(f, "moe-usage 1 %s\n", usage_signature().c_str());
        size_t n_v = 0;
        size_t n_r = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            for (int32_t il : g.layers) {
                const int32_t * slots = g.lru.slot_map[il];
                if (slots == nullptr) {
                    continue;
                }
                std::vector<int32_t> v;
                for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                    if (slots[e] < g.n_slots) {
                        v.push_back(e);
                    }
                }
                std::sort(v.begin(), v.end(), [&](int32_t a, int32_t b) {
                    return g.lru.decayed_crf(il, g.lru.make_key(il, a)) < g.lru.decayed_crf(il, g.lru.make_key(il, b));
                });
                std::vector<int32_t> r;
                if (ig < ram.size()) {
                    const ram_tier & rt = ram[ig];
                    for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                        if (slots[e] >= g.n_slots && rt.slot_of.count(((int64_t) il << 20) | e)) {
                            r.push_back(e);
                        }
                    }
                    std::stable_sort(r.begin(), r.end(), [&](int32_t a, int32_t b) {
                        return rt.ref[rt.slot_of.at(((int64_t) il << 20) | a)] < rt.ref[rt.slot_of.at(((int64_t) il << 20) | b)];
                    });
                }
                for (const auto * list : { &v, &r }) {
                    fprintf(f, "L %d %c %zu", il, list == &v ? 'V' : 'R', list->size());
                    for (int32_t e : *list) {
                        fprintf(f, " %d", e);
                    }
                    fprintf(f, "\n");
                }
                n_v += v.size();
                n_r += r.size();
            }
        }
        fclose(f);
        if (rename(tmp.c_str(), usage_path()) == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: usage memory: saved %zu device and %zu RAM experts to %s\n", n_v, n_r, usage_path());
        }
    }

    // load the remembered experts of the layers cached on device id, at its first slot lookup
    void warm(ggml_backend_t backend, int32_t id) {
        if (usage_path() == nullptr) {
            return;
        }
        if (!usage_read) {
            usage_load();
            usage_warmed.assign(devices.size(), 0);
        }
        if (usage.empty() || usage_warmed[id]) {
            return;
        }
        usage_warmed[id] = 1;

        constexpr size_t chunk = 16; // experts per upload, bounds the staging buffer
        const int64_t t_start = ggml_time_us();
        const size_t  bytes0  = stage_bytes;
        size_t n_v = 0;
        size_t n_r = 0;
        auto no_cpu = [](size_t) { return (size_t) 0; };
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.id != id) {
                continue;
            }
            // a smaller cache than last time keeps the same share of each layer, the rest moves to the RAM list
            size_t total = 0;
            for (int32_t il : g.layers) {
                total += usage[il].vram.size();
            }
            const size_t room = g.lru.n_slots - g.lru.n_filled;
            const double keep = total > room ? (double) room/total : 1.0;
            for (int32_t il : g.layers) {
                usage_layer & u = usage[il];
                const size_t drop = u.vram.size() - (size_t) (u.vram.size()*keep);
                u.ram.insert(u.ram.end(), u.vram.begin(), u.vram.begin() + drop);
                for (size_t i = drop; i < u.vram.size(); i += chunk) {
                    const size_t n = std::min(chunk, u.vram.size() - i);
                    size_t n_hit = 0;
                    if (!g.lru.plan(il, u.vram.data() + i, n, 1, (int32_t) n, 0, no_cpu, fills, cpu, n_hit)) {
                        break;
                    }
                    upload_fills(backend, il, false);
                    ggml_backend_synchronize(backend);
                    n_v += fills.size();
                }
            }
            // RAM tier: after the device experts, which pass through it on their way up
            bool file_backed = !g.ref.empty();
            for (const ggml_tensor * t : g.ref) {
                file_backed = file_backed && llama_moe_cache_is_file_backed(t);
            }
            ram_tier & r = ram_of(backend, ig);
            if (!file_backed || r.n_slots == 0) {
                continue;
            }
            total = 0;
            for (int32_t il : g.layers) {
                total += usage[il].ram.size();
            }
            const double keep_ram = total > (size_t) r.n_slots ? (double) r.n_slots/total : 1.0;
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                const std::vector<int32_t> & list = usage[il].ram;
                for (size_t i = list.size() - (size_t) (list.size()*keep_ram); i < list.size(); i += chunk) {
                    const size_t n = std::min(chunk, list.size() - i);
                    const uint64_t call_tick = ++r.tick;
                    std::vector<read_job> jobs;
                    for (size_t k = 0; k < n; ++k) {
                        bool          read = true;
                        const int32_t s    = ram_slot(r, il, list[i + k], call_tick, read);
                        for (size_t ip = 0; read && ip < l.experts.size(); ++ip) {
                            const size_t expert_size = l.experts[ip]->nb[2];
                            jobs.push_back({ (const uint8_t *) l.experts[ip]->data + list[i + k]*expert_size,
                                             r.base + r.bank_off[ip] + (size_t) s*expert_size, expert_size });
                        }
                        n_r += read;
                    }
                    run_reads(jobs);
                }
            }
        }
        // a warning so that it shows at the server's default verbosity: a run with a preloaded cache must be recognisable
        LLAMA_LOG_WARN("llama_moe_cache: usage memory: device %d preloaded %zu device and %zu RAM experts from last run's list, %.2f MiB read, %.1f s\n",
            id, n_v, n_r, (stage_bytes - bytes0)/1024.0/1024.0, (ggml_time_us() - t_start)/1e6);
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_moe_cache_slots

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        warm(backend, g.id);

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert; // replaced by the RAM slot below when the layer's experts are file-backed
                l.cpu_experts.push_back(f.expert);
            }
        }

        const size_t bytes = upload_fills(backend, il, has_cpu);

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        tail_skipped += g.lru.n_skipped;
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    void log_stats() const {
        if (tail_skipped > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: tail skip: %zu misses neither fetched nor computed\n", tail_skipped);
        }
        if (stage_reads > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: disk tier: %zu expert tensors read from the file, %.2f MiB; RAM tier: %zu hits\n", stage_reads, stage_bytes/1024.0/1024.0, ram_hits);
        }
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

void llama_moe_cache::scope_counters(uint64_t out[6]) const {
    const impl & p = *pimpl;
    out[0] = p.stats_small.hits   + p.stats_large.hits;
    out[1] = p.stats_small.misses + p.stats_large.misses;
    out[2] = p.ram_hits;
    out[3] = p.stage_reads;
    out[4] = p.stage_bytes;
    out[5] = p.stats_small.bytes  + p.stats_large.bytes;
}

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_ram_experts(const ggml_tensor * w) const {
    return pimpl->get_ram_experts(w);
}

bool llama_moe_cache::upload_file_backed(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, size_t offset, size_t size) {
    return pimpl->upload_file_backed(backend, w, dst, offset, size);
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_map(il, n_tokens);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

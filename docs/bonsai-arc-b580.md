# Bonsai 2 27B (PTQ1_0 ternary) on Intel Arc B580

https://github.com/user-attachments/assets/b6df0d56-1492-4d38-8f28-8fd323ffb92b

*Real time in the llama.cpp web UI on a B580: a new ~1000-token answer at ~65 t/s, then a whole-file edit at ~205 t/s (temperature 0, 128K context).*

This branch makes PrismML's ternary Bonsai 2 27B run fast on a 12 GB Intel Arc B580 (Xe2, "Battlemage") with the SYCL
backend, at the full 128K context. The Vulkan backend got most of the same kernel work, but SYCL is clearly faster on
this card. **Windows:** a prebuilt SYCL build is under [Releases](https://github.com/Torchit1/llama.cpp/releases) (experimental,
needs a current Intel Arc driver; unzip and run `run-bonsai.bat`).

## Results

B580 12 GB, Linux, oneAPI 2025.3, Level Zero driver 26.35. Everything below is at 131072 context with a q4_0 KV cache,
speculative decoding on (MTP head, 4 drafts, plus n-gram drafts), thinking off. Public mtp-lean model (see Model).

|                                   | SYCL (this branch) | Vulkan (this branch, older) |
|-----------------------------------|--------------------|-----------------------------|
| Fresh code answer                 | 90 t/s             | 53 t/s                      |
| Rename a symbol in pasted code    | 365 t/s            | 194 t/s                     |
| Small edit to pasted code         | 258 t/s            | 135 t/s                     |
| Plain generation, no speculation  | 42 t/s             | 32 t/s                      |
| 48K tokens of code in context: new code / edit | 63 / 34 t/s | -                  |
| ~115K tokens in context: new code / edit       | 44 / 27 t/s | -                  |

These are greedy (temperature 0) numbers on short benchmark prompts. Speculation speed depends on the text and the
sampling: in the chat UI at temperature 0 a ~1000-token new answer ran at about 65 t/s and returning a whole edited file
at about 205 t/s (video above, recorded before the latest changes). With sampling (temperature 0.6) new code runs at about
75-80 t/s. New prose drafts worse than code.

Long-context recall (needle in a haystack, a passphrase hidden in Pride and Prejudice at 10%, 50% and 90% depth):
exact at 32K, 64K and 120K tokens of context, 9 of 9.

Against the first working SYCL port of this model (same card, same settings, 32K context): fresh code 55.8 -> 85.5 t/s,
rename 217.5 -> 368.8, edit 143.0 -> 255.1, plain generation 31.4 -> 40.7. Quality: KL divergence against the reference logits is 0.00022
(99.2% same top token; the plain PTQ1_0 kernels score 0.0003), and greedy outputs on our test prompts are byte-identical to the plain PTQ1_0 kernels.

Against PrismML's own fork, which added basic SYCL support for PTQ1_0 / PQ2_0 on 24 September (prism branch at 8444536,
27 Sep). Same B580, same public mtp-lean model, q4_0 KV cache, flash attention; the chat rows use identical server flags
(32K context, MTP depth 3, no n-gram drafts):

|                                         | PrismML fork | this branch |       |
|-----------------------------------------|--------------|-------------|-------|
| Prompt reading (llama-bench pp512)      | 207 t/s      | 895 t/s     | 4.3x  |
| Plain generation (tg128)                | 20.6 t/s     | 42.3 t/s    | 2.1x  |
| Generation with 32K tokens in context   | 8.1 t/s      | 35.8 t/s    | 4.4x  |
| Chat, fresh code, MTP drafts            | 37.6 t/s     | 88.4 t/s    | 2.35x |
| Chat, fresh code, no drafts             | 20.3 t/s     | 40.7 t/s    | 2.0x  |

Draft acceptance was the same on both (166/210 vs 165/213), so the gap is the kernels: the XMX units for the weights and
for attention.

## Model

The speculative numbers need a PTQ1_0 GGUF that includes the MTP head. PrismML's own
[Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) PTQ1_0 file has no MTP head
(use `--spec-type ngram-mod` with it). A public build with the head grafted on is
[sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF),
file `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` (6.3 GB).

The table above was measured on my own derivative of Bonsai 2 27B with the same MTP head. With the public mtp-lean file
on the same card, build and settings (128K): fresh code 77 t/s, rename 306 t/s, edit 205 t/s, plain generation 38 t/s.
Different weights produce different text, so speculation lands a little less often.

## What changed

- **Ternary weights on the XMX matrix units.** At load, every PTQ1_0 weight is repacked in place to a 2-bit layout and
  multiplied with the int8 x int2 DPAS kernels from [libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL) (BSD 3-Clause,
  vendored in `ggml/src/ggml-sycl/ternsycl`). The base-3 PTQ1_0 packing has to be decoded on the ALUs first, which made
  the multi-token verify step of speculative decoding compute-bound. The 2-bit layout costs about 31% more weight memory.
  Activations are quantized to int8 with round-to-nearest (TernSYCL truncates; rounding to nearest cut KLD 5x).
- **Decode attention straight from the q4_0 KV cache on XMX** (`GGML_SYCL_FA_DEC_DPAS=1`): Q.K uses an int8 x int4 DPAS
  builtin that IGC provides on Xe2 (`intel_sub_group_i8_i4_matrix_mad_k32`, not in the public extension list) with the raw
  q4_0 bytes as the B operand, P.V runs on f16 DPAS, and one kernel serves 1-32 query tokens, so MTP and n-gram verify
  batches no longer convert the whole cache to f16. Generation at 48K context +14-16%; it also removed an out-of-VRAM crash
  near 128K. Without it, a q4_0 cache kernel serves 1-4 tokens (GQA-aware).
- **Long prompts:** full 1024-token prompt batches convert the 2-bit weights to int8 and use oneDNN's int8 GEMM
  (`GGML_SYCL_T2_W8A8_MIN=1024`), which beats the TernSYCL GEMM at that size: a 48K-token code prompt reads at ~800 t/s.
- **Long prompts above the oneDNN cap:** the chunked attention path's softmax gave each query row to one work-item;
  now a work-group per row, 2.5x faster generation at ~115K context.
- **Gated delta-net** blocked kernel, fused state writes, and several fusions for single-token decode (same-input mat-vecs
  in one launch, narrow concat, fused gate/up).
- **Memory for 128K on 12 GB:** the MTP draft context now takes its own KV cache type (`-ctkd/-ctvd q4_0`) and a smaller
  physical batch (`LLAMA_ARG_SPEC_DRAFT_UBATCH=512`), and the main prompt batch is 1024.

## Build (SYCL)

Needs the Intel oneAPI Base Toolkit 2025.3 or newer and a recent Level Zero GPU driver.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_TARGET=INTEL
cmake --build build-sycl -j --target llama-server llama-bench llama-cli
```

## Run

```sh
source /opt/intel/oneapi/setvars.sh
export GGML_SYCL_PTQ1_T2=all              # PTQ1_0 weights on XMX (ffn = feed-forward only, unset = off)
export LLAMA_ARG_SPEC_DRAFT_UBATCH=512    # smaller compute buffer for the MTP draft context
export GGML_SYCL_FA_ONEDNN_MAX_KV=98304   # see below: without it a ~120K-token prompt runs out of VRAM
export GGML_SYCL_FA_DEC_DPAS=1            # decode / verify attention on XMX straight from the q4_0 cache (Xe2)
export GGML_SYCL_T2_W8A8_MIN=1024         # full 1024-token prompt batches via oneDNN's int8 GEMM: ~10% faster long prompts
./build-sycl/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-ngram-mod-n-max 256 \
  -ub 1024 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080 \
  --jinja --chat-template-file docs/bonsai-arc-b580-windows/bonsai-chat-template.jinja
```

Thinking is off in this command (and in `run-bonsai.bat`) because it is faster for coding. For maths and logic puzzles
turn it on with `--chat-template-kwargs '{"enable_thinking":true}'`: in one tester's run a timer puzzle that the model
brute-forced for ~6,000 tokens with thinking off was solved first try in ~1,000 tokens with it on (42 vs 55 t/s per token,
but about 4x sooner overall).

The chat template file is the model's own template with one change: a system message in the middle of a conversation
becomes a note in a user turn instead of an error. Coding agents such as Claude Code send those (see below); plain chat
is unaffected.

`GGML_SYCL_FA_ONEDNN_MAX_KV` matters at 128K: the fast prompt-attention path converts the whole KV cache to f16 (about
4 KB per token of context), and with this configuration's ~0.5 GB of spare VRAM a prompt of about 119K tokens ran out of
memory and took the server down. Above the cap (98304 tokens) prompt attention uses a chunked path instead: slower
(~330 t/s prompt reading at 120K) but safe. Shorter prompts are unaffected.

With a shorter context you can use `-ub 2048` for faster prompt reading. `GGML_SYCL_PTQ1_T2=ffn` puts only the
feed-forward weights on XMX: about 380 MiB less weight memory, enough for `-ub 2048` at 128K, and most of the speed
(fresh 80, rename 314, edit 213).

The first long prompt after the very first start compiles the XMX kernels (about 30 s); the GPU driver caches them after
that. If the server ever hangs during start-up in GPU initialisation after being killed mid-compile, move
`~/.cache/neo_compiler_cache` aside.

## As a coding agent (Claude Code)

llama-server speaks the Anthropic Messages API, so Claude Code can use Bonsai as its model: file reads and edits, shell
commands and tests, all running locally on the card. Start the server as above (with the chat template file), then:

```sh
export ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=bonsai
export ANTHROPIC_DEFAULT_HAIKU_MODEL=bonsai ANTHROPIC_DEFAULT_SONNET_MODEL=bonsai ANTHROPIC_DEFAULT_OPUS_MODEL=bonsai
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

(Windows PowerShell: `$env:ANTHROPIC_BASE_URL="http://localhost:8080"` and so on.) Claude Code warns that it doesn't
know the model name; that's harmless.

It works well. Asked for a Snake game with its logic unit-tested in Node, it wrote the game and a test suite, ran the
tests, fixed the one failure and re-ran them to a clean pass, in about 5.5 minutes. Things that helped:

- A `CLAUDE.md` in the project with a few rules. Without them it tends to build more than asked:

  ```markdown
  # Rules
  - Build only what was asked. No extra features, themes, settings or polish unless requested.
  - Prefer the simplest code that works. Small files, few functions.
  - After every change, run the tests. Never say a test passed unless you ran it this turn and saw the output.
  - If a tool call fails, say so and retry. Never report a failed edit as done.
  - End each task by listing exactly which commands you ran and their results.
  ```

- Patience on the first turn: Claude Code's instructions and tool list are about 20-27K tokens, which take about 30 s to
  read. Later turns reuse the cache and start quickly.
- One user at a time (`-np 1`): the web UI and the agent share the one slot.

Believe the tool output rather than the summary, as with any local model. For comparison, Gemma 4 12B (below) is faster
but skipped steps and reported tests as passed without running them, so Bonsai is the one to use as an agent.

## Build and run (Vulkan)

SYCL is faster on the B580 (see the table), but the Vulkan backend has most of the same kernel work and runs on any
Vulkan driver. Tested on Mesa ANV 26.2. Needs the Vulkan SDK (glslc and headers).

```sh
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j --target llama-server llama-bench llama-cli
./build-vk/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-ngram-mod-n-max 256 \
  -ub 2048 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080
```

At 128K this left about 1.2 GB of VRAM free on the B580. The `GGML_SYCL_*` switches below do not apply to Vulkan; its
new paths can be turned off with `GGML_VK_PTQ1_MC_OFF=1` (multi-column PTQ1_0 mat-vec).

## Ternary Bonsai 8B

PrismML's smaller [Ternary Bonsai 8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf) (`Ternary-Bonsai-8B-PQ2_0.gguf`,
2 GB) also runs on the XMX path. It is the older Qwen3-based Bonsai, so it has no MTP head. Use n-gram drafts, or
[Ternary Bonsai 1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) (same vocabulary) as a draft model:

```sh
source /opt/intel/oneapi/setvars.sh
export GGML_SYCL_PTQ1_T2=all
./build-sycl/bin/llama-server -m Ternary-Bonsai-8B-PQ2_0.gguf -ngl 99 -c 32768 -ctk q4_0 -ctv q4_0 -np 1 \
  --spec-type ngram-mod --spec-ngram-mod-n-max 256 \
  -ub 1024 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080
# or add the 1.7B as drafter: --spec-type draft-simple,ngram-mod -md Ternary-Bonsai-1.7B-PQ2_0.gguf -ngld 99 --spec-draft-n-max 4
```

On the B580 (32K context, temperature 0):

| | new code | rename | edit |
|---|---|---|---|
| no drafts | 100 t/s | 75 | 80 |
| n-gram drafts | 102 | 506 | 658 |
| 1.7B drafter + n-gram | 110 | 471 | 608 |

The XMX path gives 134 t/s plain generation (118 without it) and 765 vs 408 t/s on 8-token batches, with identical
perplexity. At 2 GB it should also suit 8 GB cards, but I've only tested it on the B580.

Tool calling (first 100 of each [BFCL v3](https://gorilla.cs.berkeley.edu/leaderboard.html) category, my simplified scorer, so
compare rows with each other only):

| model | simple | multiple | parallel | no call needed |
|---|---|---|---|---|
| Ternary Bonsai 2 27B | 98 | 95 | 92 | 79 |
| Ternary Bonsai 8B | 96 | 92 | 83 | 92 |
| Gemma 4 12B (QAT q4_0) | 96 | 92 | 84 | 83 |

## Gemma 4 12B (bonus)

Google publishes a small "assistant" draft model for the QAT Gemma 4 12B, and the same XMX work covers Gemma's shapes:
the int8 x int4 decode attention handles its global (head 512) and sliding (head 256) layers, and a q4_0 small-batch GEMM
on XMX streams the weights once for verify batches of 5-256 tokens. Together Gemma 4 12B QAT q4_0 goes from ~40 to ~100
t/s on new code on the B580:

```sh
# once: convert Google's assistant (846 MB, not gated) to GGUF
huggingface-cli download google/gemma-4-12B-it-qat-q4_0-unquantized-assistant --local-dir gemma4-12b-assistant
python convert_hf_to_gguf.py gemma4-12b-assistant --outtype q8_0 --outfile gemma-4-12b-it-qat-assistant-Q8_0.gguf

source /opt/intel/oneapi/setvars.sh
export GGML_SYCL_FA_DEC_DPAS=1            # XMX decode attention (Gemma global + sliding layers too)
export GGML_SYCL_Q4_0_DPAS=1              # q4_0 verify batches of 5-256 tokens on XMX (weights streamed once)
export GGML_SYCL_MMVQ_CHUNK_MAX=128       # anything else up to 128 tokens as chunked mat-vecs, not dequantize-everything
./build-sycl/bin/llama-server -m gemma-4-12b-it-qat-q4_0.gguf -md gemma-4-12b-it-qat-assistant-Q8_0.gguf -ngl 99 -ngld 99 \
  -c 131072 -ctk q4_0 -ctv q4_0 -np 1 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-ngram-mod-n-max 96 \
  -ub 1024 -b 2048 --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080
```

| Gemma 4 12B QAT q4_0, B580, 128K context | new code | rename | edit | new code / edit at 48K |
|---|---|---|---|---|
| n-gram drafts only | 40 t/s | 164 | 64 | - |
| + assistant (3 drafts) | 92 | 186 | 136 | 47 / 44 |
| + XMX attention and q4_0 GEMM (above) | **102** | **275** | **168** | **65 / 57** |

About 100 t/s on new code at temperature 0.6 too. Gemma writes slightly better code (HumanEval+ 152 vs 141 of 164) and is
faster on new code; Bonsai is faster on edits (~255 vs 168 t/s) and renames (~370 vs 275).

## Switches (SYCL)

All optimisations are on by default except the XMX path. The XMX path needs an Xe2 or newer GPU (Arc B-series, Lunar Lake, Panther Lake); on
others it turns itself off with a warning. Set any of these to turn a piece off for comparison:
`GGML_SYCL_PTQ1_T2_GEMM_OFF`, `GGML_SYCL_PTQ1_MULTI=0`, `GGML_SYCL_PTQ1_MULTI_NCOLS=0`, `GGML_SYCL_PTQ1_GLU1=0`,
`GGML_SYCL_PTQ1_PAIRS=0`, `GGML_SYCL_PTQ1_NCOLS_DEC_OFF`, `GGML_SYCL_FA_DEC_OFF`, `GGML_SYCL_GDN_BLOCKED_OFF`,
`GGML_SYCL_GLU_FUSE_OFF`, `GGML_SYCL_TOPK_OLD=1`. Opt-in: `GGML_SYCL_FA_DEC_DPAS=1` (XMX decode attention, Xe2),
`GGML_SYCL_MMVQ_CHUNK_MAX=N` (largest quantized batch run as chunked mat-vecs, default 32), `GGML_SYCL_Q4_0_DPAS=1`
(q4_0 verify batches on XMX, Xe2), `GGML_SYCL_T2_W8A8_MIN=N` (prompt batches of N+ tokens via oneDNN int8, 0 = off). An MTP GGUF can carry a
trimmed draft LM head (`blk.<n>.nextn.draft_head`, top-K frequent tokens; ~5% faster drafting on the B580);
`LLAMA_MTP_DRAFT_HEAD_OFF=1` ignores it.

## Feedback and your numbers

If you run this on a B580 or another Arc card, please post your results (card, driver, context, the numbers you get) in
this repository's Discussions, and report problems as issues. Results from other setups are the most useful thing
right now.

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), PrismML's PTQ1_0 support, and the `bonsai-combo`
branch of [professorpalmer/llama.cpp-ada-ternary](https://github.com/professorpalmer/llama.cpp-ada-ternary) (PrismML
PR #221) that this branch started from, with the ternary DPAS
kernels from [libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL) (BSD 3-Clause; licence and notice included).

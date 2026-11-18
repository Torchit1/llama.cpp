# Arc Cascade: tester notes (facts only)

Arc Cascade is the llama.cpp SYCL backend plus local changes for Intel Arc cards. For a mixture-of-experts model that does
not fit in VRAM, the experts are split across tiers: an expert cache in VRAM (on one or two cards), the next most used
experts pinned in RAM, and the rest read from the model file. Dense models are not helped by the tiers.

This tree is mainline llama.cpp (12 Nov 2026) with several open upstream pull requests merged in and local commits on
top. It is not an upstream release.

## Tested

Hardware: Arc Pro B70 (32 GB) and Arc B580 (12 GB) in one machine, Linux, oneAPI 2026.1, legacy Level Zero adapter
(`SYCL_UR_USE_LEVEL_ZERO_V2=0`). All numbers are decode speed in tokens per second, 15 Nov 2026.

| Model | Placement | Short prompt, first answer / asked again | 4.5K prompt |
|---|---|---|---|
| Qwen3.8 Flash-Next Coder IQ1_M + draft head | B70, all in VRAM | 99.8-101.5 | 90.8-94.7 (82.4-83.1 at 16K) |
| Qwen3.5-35B-A3B Q4_K_M (22 GB) | B70, all in VRAM | 107-119 | |
| Qwen3.5-35B-A3B Q4_K_M (22 GB) | B580, tiers, 12 GB RAM budget | 18.4 / 33.3 | 31.7 |
| same, mainline options: expert cache only | B580 | 15.7 | 18.7 |
| same, mainline options: experts on the CPU | B580 | 5.2-5.9 | did not finish in 15 min |
| Qwen3.5-122B-A10B UD-IQ3_XXS (44.7 GB) | B70, tiers, 13 GB RAM budget | 13.4 / 27.1 | 25.4 |
| same, mainline options: expert cache only | B70 | 11.0 / 16.2 | 12.5 |
| same, mainline options: experts on the CPU | B70 | 0.9-1.1 | |
| Qwen3.5-122B-A10B UD-IQ3_XXS (44.7 GB) | B70 + B580, tiers, 13 GB RAM budget | 17.6 / 35.0 | 33.1 |
| same, 8 GB RAM budget | B70 + B580, tiers | 15.7 / 38.5 | 32.5 |

- The tier runs give the same output as the same model with everything on a card.
- A model that fits one card is 17-25% slower when split over two. The launcher uses one card when the model fits.
- The B580 and two-card rows were measured on the pushed build. The B70 rows for the 35B and 122B are from before
  small host-to-card copies went through pinned memory (`GGML_SYCL_PIN_SMALL`, on by default); that change took the 122B
  on the B70 from 37.6 to 40.7 t/s on a repeated prompt, switch off and on inside one server.
- The mainline-options rows were measured before that change too.
- Expert weight formats with a kernel on the card: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ1_S,
  IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS. Only IQ1_M, Q4_K_M and IQ3_XXS models were run end to end.
- Repeatability: with a fresh server per run, the same request gave identical output in every repeat tried (Flash-Next,
  empty, 4.5K and 16K context). Inside one server, state carried between requests can change a later answer.

## Not tested

Windows, other Intel cards, the integrated GPU, AMD or NVIDIA cards, models from other families, more than one request
at a time, and the `Level Zero v2` adapter.

## Build

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
          -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=ON -DLLAMA_CURL=OFF
    cmake --build build --target llama-server llama-bench -j

## Run with the launcher

`tools/arc-cascade/arc_cascade.py` reads the model's sizes from the GGUF header, decides the placement and starts
`llama-server`. `plan` only prints the decision.

    python3 tools/arc-cascade/arc_cascade.py plan --model MODEL.gguf --cards B70,B580 --ram-mib 13000
    python3 tools/arc-cascade/arc_cascade.py run  --model MODEL.gguf --cards B580 --ram-mib 12000 --bin build/bin \
        -- --host 127.0.0.1 --port 8080

- `--cards`: names from the table at the top of the script (B70, B580). For another card give its usable VRAM with
  `--vram-mib`, one value per card.
- `--ram-mib`: the most RAM the server may use. The pinned tier is this minus about 3.5 GB. Without it the budget is
  the free RAM minus 8 GB, and 8 GB always stay free whatever `--ram-mib` asks for.
- `--fast-prompt` / `--no-fast-prompt`: when the model fits one card and most of its weights are dense Q4_0, the
  launcher turns on the 8-bit prompt path (`GGML_SYCL_Q4_0_ROT8=1`) with prompt chunks of 4,096 tokens, if that
  chunk's working memory still fits. Qwen3.8-27B Q4_0 on a B70: 1,347 -> 1,889 t/s on a 19K prompt. These switch
  it on or off whatever the launcher decides.
- `--head HEAD.gguf`: a multi-token draft head for the model, if one exists.
- `--remember`: on shutdown the server writes which experts were in VRAM and RAM, and loads them back at the next
  start. Off by default. Output is unchanged. 122B on both cards: the first answer after a start runs at 37.6 t/s
  instead of 7.4, and the start takes 34 s instead of 15 s. Do not use it for speed measurements.
- Order of decisions: fits the first card, use it alone; fits across the cards, split it; otherwise tiers. A model
  that is mostly dense and does not fit is refused.

## Run by hand (tiers)

    export SYCL_UR_USE_LEVEL_ZERO_V2=0 GGML_SYCL_MMID_SCHED=3
    LLAMA_MOE_HOST_PIN_MIB=0 LLAMA_MOE_RAM_CACHE_MIB=<MiB of RAM to pin> \
    build/bin/llama-server -m MODEL.gguf -ngl 99 -fa on -lm none -lzm on -c 8192 \
        -cmoe --moe-cache-mib <MiB of VRAM for the expert cache> -b 1024 -ub 1024 -t <cores>

- `LLAMA_MOE_RAM_CACHE_MIB`: RAM tier. The server needs about 3.5 GB of RAM beyond it. It must hold at least two
  layers of experts.
- `--moe-cache-mib`: VRAM tier, over all cards in use. Leave room for the rest of the model and the context.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`.
- A model that fits: `-ngl 99 -lm dio` (direct reads load a 22 GB model in 22 s instead of 92-112 s with mmap).
- `LLAMA_MOE_USAGE_FILE=path`: what `--remember` sets.
- `LLAMA_MOE_TAIL_SKIP=K` is lossy and off by default. Leave it off for tests.

## What to report

Card, driver and oneAPI version, model and quant, the plan the launcher printed, decode and prompt t/s, free RAM
before and during the run, and whether the answer to a fixed prompt matches a run with everything on a card or on CPU.

# arc-deterministic-v0.1

This is llama.cpp for Intel Arc cards that gives you the same answer every time you ask the same question.

That sounds like it should already be true, but it isn't. On my Arc Pro B70 I could start a fresh server, send it the same prompt at temperature 0 three times, and get three different answers back. Nothing was broken and nothing was random on purpose. The GPU was adding up numbers in a slightly different order each run, and over a few hundred tokens that's enough to change the text.

If you're just chatting with a model you'll never notice. If you're testing a change, comparing two builds, scoring a benchmark or trying to show that a result is real, you need to be able to run the same thing twice and get the same thing back. That's what this release is for.

It's built on upstream llama.cpp master from 15 November 2026, with four changes on top. Everything else is stock.

## What it does

**Same answer every run, out of the box.** The oneDNN matrix multiplies now run in their deterministic mode. On the B70 this took the same prompt from three different answers in three fresh servers to the same answer every time. I couldn't measure any speed cost. If you want the old behaviour back, set `GGML_SYCL_DNN_DETERMINISTIC=0`.

**Same answer at any batch size, if you ask for it.** Set `GGML_SYCL_DETERMINISTIC=1` and the output doesn't change when you change the batch or ubatch size either. Normally it does, because the GPU uses different kernels for different batch sizes. In this mode every token goes through the same path as a single token would. Generation runs at about the same speed, but prompt processing is a lot slower, so this is for testing and checking results, not for everyday use.

**Ties broken the same way every time.** When the top-k selection has two values exactly level, it now always takes them in column order. Before, it took whichever one the GPU got to first, which changed from run to run on long prompts.

**A tool to check it yourself.** `scripts/detcheck.py` runs a model through a build a few times and tells you whether the answers match, whether they hold across batch sizes, and whether two builds give the same result.

## Download

Two Linux builds, the same way upstream packages its SYCL releases:

- `llama-arc-deterministic-v0.1-bin-ubuntu-sycl-fp32-x64.tar.gz`
- `llama-arc-deterministic-v0.1-bin-ubuntu-sycl-fp16-x64.tar.gz`

The fp16 build processes prompts a lot faster on these cards, so that's the one I'd use. The fp32 build does its maths at full precision, if you'd rather have that. Checksums are in `SHA256SUMS`.

You need the Intel GPU driver and the oneAPI runtime (2026.x) installed, the same as for upstream's SYCL builds. Before you run anything:

```
source /opt/intel/oneapi/setvars.sh
```

If llama-server crashes as soon as the model loads, your GPU driver is older than the oneAPI runtime expects. Updating the driver fixes it, or you can set this before you run anything:

```
export SYCL_UR_USE_LEVEL_ZERO_V2=0
```

My own box needed that with driver 26.35 and oneAPI 2026.2.

## Use it

Nothing changes from normal llama.cpp. Repeatable output is on by default:

```
./llama-server -m your-model.gguf -ngl 99
```

For the same output at any batch size:

```
GGML_SYCL_DETERMINISTIC=1 ./llama-server -m your-model.gguf -ngl 99
```

To check a model on your own card:

```
python3 scripts/detcheck.py -m your-model.gguf
```

It prints whether the build is repeatable and whether it's batch-invariant. Give it a second build with `-B other/build/bin` and it tells you if the two give the SAME or DIFFERENT results. If you have more than one GPU, pick one first with `ONEAPI_DEVICE_SELECTOR=level_zero:0` (or 1, and so on).

## What I tested

On an Arc B580 and an Arc Pro B70.

- SmolLM2-360M (BF16), Llama 2 7B (Q4_0) and Llama 2 7B (Q4_K_M) on the B580: the same answers and the same logits every run, and with `GGML_SYCL_DETERMINISTIC=1` the same at ubatch 512 and 128.
- Qwen3.8 Flash-Next Coder (IQ1_M, a large MoE model, with MTP drafting) on the B70: three fresh servers gave three different answers before the change and the same answer after, on a short prompt and a 4,500-token one. On a 17,700-token prompt it went from four different answers in eight runs to one.
- The released files themselves, downloaded and unpacked fresh, on both cards: `detcheck.py` says repeatable and batch-invariant for SmolLM2-360M and Llama 2 7B (Q4_0) on the B580 and the B70, and for the fp16 build on the B580.

Speed with the fp32 build, Llama 2 7B Q4_0, tokens per second (prompt of 512 / generating 64):

| Card | Default | Old oneDNN behaviour | `GGML_SYCL_DETERMINISTIC=1` |
|---|---|---|---|
| B580 | 680 / 89.1 | 679 / 89.1 | 148 / 88.1 |
| B70 | 1,004 / 112.0 | 1,005 / 112.2 | 266 / 109.2 |

So the repeatable default costs nothing, and the batch-invariant mode keeps generation within a few percent but makes prompts about four times slower.

## What it doesn't do

I'd rather be clear about this than have someone find out the hard way.

- It won't make two different cards agree with each other, or a GPU agree with a CPU. They do the maths in a different order and the results differ slightly.
- A different build, driver or oneAPI version can also give different results. Repeatable means the same build on the same card.
- A server that's been running a while can still give a slightly different answer to a later request, because earlier requests leave state behind. For results you want to repeat exactly, use a fresh server per request, or send the same requests in the same order.
- By default it's repeatable, but not batch-invariant. Change the ubatch size and the numbers can change. Use `GGML_SYCL_DETERMINISTIC=1` if you need that.
- I haven't tested batch invariance on a MoE model yet, or anything on Windows, laptop GPUs, the older A-series cards, or a model split across two GPUs.

## Licence

MIT, the same as llama.cpp. See `LICENSE`.

Jesse (Torchit1)

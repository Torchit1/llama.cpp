# GPU Farm — Context & Game Plan

Handoff for another Claude session taking over the llama.cpp Vulkan-on-Intel-Arc-Xe2 optimization work. Read this before doing anything. No secrets live in this file.

## 1. Machines and cards
Everything is on **TOWER** (Unraid, `192.168.1.51`). Reach it: `ssh root@192.168.1.51` (key auth from the desktop), then `docker exec arc-fn ...` to run inside the lab container.

The Vulkan work runs in the **arc-fn** container (image `local/arc-oneapi:2026.1`); it mounts the source tree at `/lab`. Physical cards (shared with a second sandbox container `gpu-sandbox` at ssh port 2222 — don't collide with it):

| card | VRAM | Vulkan device | SYCL device |
|------|------|---------------|-------------|
| Intel Arc Pro B70 (BMG-G31) | 32 GB | `GGML_VK_VISIBLE_DEVICES=0` | `level_zero:0` |
| Intel Arc B580 (BMG-G21) | 12 GB | `GGML_VK_VISIBLE_DEVICES=1` | `level_zero:1` |
| iGPU (UHD 730) | shared | `=2` | `level_zero:2` (unused here) |

GPU check: `ls /dev/dri` (or `docker exec arc-fn sycl-ls` after sourcing setvars).

## 2. How a job runs today
A long-lived **runner** (`runner.sh`, PID 1 inside arc-fn) watches `/lab/dev26/jobs/*.sh`. When a `*.sh` appears it runs `bash <job>` and writes `out/<name>.log` + `out/<name>.done` (exit code), then removes the job. **It is serial — one job at a time.**

Submit from the host (`/workspace/homelab/arc-lab/dev26`):
```
cp dev26/<job>.sh dev26/jobs/<job>.sh.tmp && mv dev26/jobs/<job>.sh.tmp dev26/jobs/<job>.sh
```
(or `dev26/run.sh <job>.sh <limit_seconds> <tail_lines>`). Every job script needs a `# progress: <one-liner>` header and should append progress lines to `dev26/out/<job>.progress`.

**Guard** (host, for every GPU job):
```
(GUARD_MIN=20 nohup bash dev26/guard.sh > dev26/out/<job>-guard.log 2>&1 </dev/null &)
```
It kills arc-fn job processes if host `MemAvailable < 10 GB` or 1-min load > 40.

Exact commands:
- One test-backend-ops op (perf): `GGML_VK_VISIBLE_DEVICES=0 /lab/upstream-stock/build-vk/bin/test-backend-ops perf -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_K=f16.*type_V=f16'`
- One llama-bench run: `GGML_VK_VISIBLE_DEVICES=0 /lab/upstream-stock/build-vk/bin/llama-bench -m /fastmodels/qwen38-27b/Qwen3.8-27B-Q4_0.gguf -ngl 99 -fa 1 -p 4096 -n 128 -r 2`

For a second card in parallel (runner is serial), launch directly:
```
docker exec -d arc-fn bash -c "export GGML_VK_VISIBLE_DEVICES=1; <cmd> > /lab/dev26/out/x.log 2>&1"
```

## 3. Where things live
- Source (git worktree, branch `vulkan-xe2-fa`): `/workspace/homelab/arc-lab/upstream-stock` (host) = `/lab/upstream-stock` (in arc-fn). Main repo `prism-llama.cpp`. Remotes: `mine`=github.com/Torchit1/llama.cpp, `origin`=github.com/PrismML-Eng, upstream-master ref = ggml-org/llama.cpp.
- Vulkan build: `build-vk/` (configured `-DGGML_VULKAN=ON -DGGML_SYCL=OFF`).
- Models: `/fastmodels/qwen38-27b/Qwen3.8-27B-Q4_0.gguf` (27B Q4_0, main bench); `/fastpool/arc-fn/keep/Qwen3-0.6B-Q8_0.gguf` (small fast-iteration model); `/fastpool/arc-fn/keep/Qwen3.8-27B-IQ3_S.gguf` (27B IQ3_S, decode work). `/fastmodels/` also has `bonsai/`, `flash-next-coder/`, `fn-mtp/`, `qwen35/` subdirs (not inventoried here).
- Results: `dev26/out/` (`.progress`, `.log`, `.done`, `vkprof-*.txt`). Running numbers: `/workspace/homelab/arc-lab/RESULTS.md`.
- WINS.md (headline scoreboard): `F:\deepseek-sandbox\WINS.md` (desktop).
- Shared team notes: `F:\Claude-DeepSeek\la\Claude-3p\side-session-notes\local_2ca404cf-d9c5-4c62-9b46-1863e532d963\*.md` (desktop).

**Small models on disk:** `Qwen3-0.6B-Q8_0.gguf` — 761.80 MiB, Q8_0. (That is the only small one; the rest are 27B Q4_0 and 27B IQ3_S.)

## 4. How long things take (measured)
- Incremental build (shaders): ~2 min.
- One test-backend-ops op (perf): seconds.
- test-backend-ops correctness, `-o MUL_MAT`: ~6 min (~2600 cases).
- Full test-backend-ops correctness: ~10 min.
- 0.6B llama-bench (pp512/tg128): ~30 s.
- 27B llama-bench (pp4096 + 128 tokens): ~2 min.

## 5. Team sessions
Seven side sessions (message them via SendMessage by id). Models: all on `claude-dspro` (= deepseek-pro via LiteLLM) except wins-reporter.
- `vk-review` — Vulkan code reviewer + coder (writes/edits the Vulkan source). `claude-dspro`.
- `perf-analyst` — perf-data analyst, prioritizes next optimization. `claude-dspro`.
- `xe2-expert` — Intel Xe2 / Vulkan-coopmat / Mesa-ANV hardware research. `claude-dspro`.
- `test-harness` — benchmark/measurement harness owner. `claude-dspro`.
- `results-curator` — maintains `RESULTS.md`. `claude-dspro`.
- `orchestrator` — dispatches work, tracks `PIPELINE.md`. `claude-dspro`.
- `wins-reporter` — maintains `WINS.md` scoreboard. `claude-sonnet-5-5`.

## 6. What breaks / gets in the way
- **`/workspace` symlink on TOWER** (`/workspace -> /mnt/user/claude-workspace`) is required by every git worktree + `run.sh`; it is lost on Unraid reboot. Symptom: git says "gitfile does not point to a valid repository". Fix: `ln -s /mnt/user/claude-workspace /workspace`.
- **Git-as-root flips file ownership** on checkout. Fix: `git ls-files -z | xargs -0 chown 1000:1000`.
- **DeepSeek billing**: team sessions on deepseek-pro fail with 402 "Insufficient Balance" when the account is empty. Top up, or switch their model.
- **Runner is serial** — one job at a time; parallel second-card work must go through `docker exec -d`.
- **FLASH_ATTN_EXT head dim 512 + large KV → `VK_ERROR_DEVICE_LOST`** on all FA paths. Exclude `hsk=512` from test filters.
- **VRAM rule**: size jobs ≤ ~75% of the card's VRAM (overflow spills into host RAM and can freeze the box). Guard + save outputs to `/work`.

## 7. Branch, commits, and what's in flight
- Branch: `vulkan-xe2-fa` (off mainline `5a4afd807`).
- Committed: `5b9e833af` — m8 coopmat flash-attn (MatBc=8/MatK=16, `FA_COOPMAT_M8`); `2cd775ddd` — FA auto-select (skip xe_fa for head dim ≥256 prefill). Verified: 27B pp4096 471 → 534 t/s.
- Uncommitted (in the worktree, do NOT commit): int8 coopmat matmul enable — three AMD-only gates in `ggml-vulkan.cpp` (lines ~2135, ~2816, ~2889) now also allow `INTEL_XE2`. **Currently WRONG results** (76 MUL_MAT FAIL, ERR ~1.0 on q3_K/q6_K/nvfp4).
- In flight: fix `cm_elem_row()` in `vulkan-shaders/mul_mmq_cm1.comp:130` — the Xe2 DPAS accumulator element→row map (RDNA3 "interleaved" branch is wrong for Xe2). `vk-review` implements, `xe2-expert` researches Mesa ANV. After the fix: re-run `test-backend-ops test -b Vulkan0 -o MUL_MAT` to 0 failures, then benchmark the matmuls (the expected int8-coopmat win; matmuls are ~5.9 s of the 8.7 s prompt pass).

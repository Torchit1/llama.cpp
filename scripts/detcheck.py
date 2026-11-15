#!/usr/bin/env python3
"""detcheck: check that a llama.cpp build gives the same answer every time.

Runs a model through a build's llama-server (greedy, fresh process per prompt) and llama-perplexity (exact logits), then
reports:
  repeatable       same settings, run twice                    -> identical?
  batch-invariant  GGML_SYCL_DETERMINISTIC=1 at two ubatch sizes -> identical?
  SAME/DIFFERENT   two builds (optional), deterministic mode    -> do they compute the same thing?

Usage:
  python3 scripts/detcheck.py -m model.gguf [-b build/bin] [-B other/build/bin] [--ub 512 128] [--ctx 512]
Environment passes through, so pick a device with ONEAPI_DEVICE_SELECTOR=level_zero:N as usual.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "List five uses for a paperclip.",
]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def text_file(tmp):
    # scoring text for the logits check: llama.cpp's own docs (MIT), so nothing third-party is shipped
    root = os.path.dirname(HERE)
    parts = []
    for rel in ("docs/build.md", "docs/backend/SYCL.md", "README.md"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8", errors="replace").read())
    if not parts:
        sys.exit("no docs found next to scripts/ to score; run from a llama.cpp checkout")
    path = os.path.join(tmp, "score.txt")
    open(path, "w").write("\n\n".join(parts))
    return path


def run_arm(bin_dir, model, ub, env_extra, tmp, label, ntok, ctx, chunks):
    env = dict(os.environ, **env_extra)
    env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    out = {"tokens": [], "logits": None}
    port = "18291"
    for i, prompt in enumerate(PROMPTS):
        # a fresh server per prompt: a server that already answered other requests carries state that can change later answers
        logf = open(os.path.join(tmp, f"server-{label}-{i}.log"), "w")
        srv = subprocess.Popen([os.path.join(bin_dir, "llama-server"), "-m", model, "-ngl", "99", "-c", str(max(ctx, 2048)),
                                "-b", str(max(ub, 512)), "-ub", str(ub), "-np", "1", "--host", "127.0.0.1", "--port", port,
                                "--seed", "1"], env=env, stdout=logf, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                try:
                    if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read():
                        break
                except Exception:
                    pass
                if srv.poll() is not None:
                    sys.exit(f"llama-server exited; see {logf.name}")
                time.sleep(1)
            req = {"prompt": prompt, "n_predict": ntok, "temperature": 0, "seed": 1, "cache_prompt": False,
                   "return_tokens": True, "ignore_eos": True}
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                          json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=1800))
            out["tokens"].append(hashlib.sha256(json.dumps(r.get("tokens") or []).encode()).hexdigest()[:12])
        finally:
            srv.terminate(); srv.wait(); logf.close()
    kld = os.path.join(tmp, f"logits-{label}.kld")
    # one sequence per pass (-b = -c): packing several sequences into one pass changes logits by itself
    r = subprocess.run([os.path.join(bin_dir, "llama-perplexity"), "-m", model, "-ngl", "99", "-f", text_file(tmp),
                        "-c", str(ctx), "-b", str(ctx), "-ub", str(min(ub, ctx)), "--chunks", str(chunks),
                        "--kl-divergence-base", kld], env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(kld):
        sys.exit(f"llama-perplexity failed for {label}:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    h = hashlib.sha256()
    with open(kld, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    out["logits"] = h.hexdigest()[:12]
    os.remove(kld)
    log(f"  {label}: tokens {' '.join(out['tokens'])} | logits {out['logits']}")
    return out


def check_build(bin_dir, model, ubs, tmp, tag, a):
    log(f"build {tag}: {bin_dir}")
    r1 = run_arm(bin_dir, model, ubs[0], {}, tmp, f"{tag}-default-1", a.ntok, a.ctx, a.chunks)
    r2 = run_arm(bin_dir, model, ubs[0], {}, tmp, f"{tag}-default-2", a.ntok, a.ctx, a.chunks)
    det = [run_arm(bin_dir, model, ub, {"GGML_SYCL_DETERMINISTIC": "1"}, tmp, f"{tag}-det-ub{ub}", a.ntok, a.ctx, a.chunks)
           for ub in ubs]
    rep = r1 == r2
    inv = all(d == det[0] for d in det)
    return {"repeatable": rep, "batch_invariant": inv, "det": det[0]}


def main():
    ap = argparse.ArgumentParser(description="Does this llama.cpp build give the same answer every time?")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-b", "--bin", default=os.path.join(os.path.dirname(HERE), "build", "bin"), help="build A bin dir")
    ap.add_argument("-B", "--bin2", help="optional build B bin dir: SAME/DIFFERENT against A")
    ap.add_argument("--ub", type=int, nargs="+", default=[512, 128], help="ubatch sizes for the batch-invariance check")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ntok", type=int, default=128, help="tokens generated per prompt")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="detcheck-") as tmp:
        ra = check_build(a.bin, a.model, a.ub, tmp, "A", a)
        rb = check_build(a.bin2, a.model, a.ub, tmp, "B", a) if a.bin2 else None
        print()
        print(f"model: {a.model}")
        for tag, r in (("A", ra), ("B", rb)):
            if r:
                print(f"build {tag}: repeatable {'YES' if r['repeatable'] else 'NO'} | "
                      f"batch-invariant (GGML_SYCL_DETERMINISTIC=1, ub {' vs '.join(map(str, a.ub))}) "
                      f"{'YES' if r['batch_invariant'] else 'NO'}")
        if rb:
            print(f"A vs B (deterministic mode): {'SAME' if ra['det'] == rb['det'] else 'DIFFERENT'}")
        ok = ra["repeatable"] and ra["batch_invariant"] and (rb is None or (rb["repeatable"] and rb["batch_invariant"]))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

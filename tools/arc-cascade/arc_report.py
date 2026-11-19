#!/usr/bin/env python3
"""Arc Cascade test report: one command that runs a short, fixed test of a model through the launcher on your machine and
writes the results, with your hardware and software versions, to one Markdown file you can paste into an issue or a comment.

    python3 tools/arc-cascade/arc_report.py --model /path/to/model.gguf [--head /path/to/mtp-head.gguf] [--bin build/bin]

What it does, in about 5-15 minutes:
  1. Lists the GPUs llama-server sees and picks the first Intel Arc card (or use --cards).
  2. Prints the launcher's plan for the model (card / RAM / disk placement).
  3. Starts the server through the launcher and runs: a short code answer (twice, cold and warm), a short prose answer,
     and a ~4,000-token prompt with a short answer. Greedy decoding, thinking off.
  4. Stops the server and writes arc-cascade-report-<date>.md in the current folder.

The report holds: OS, kernel, CPU model, RAM size, GPU names and memory, oneAPI / driver versions as sycl-ls prints them,
the git commit you built, the model file name and size, the plan and the speeds. No host name, user name or paths outside
the model file name. Read it before you share it.

Needs: a build of this branch (llama-server in --bin), the oneAPI environment loaded (source setvars.sh), Python 3.8+.
Standard library only. Nothing is downloaded or installed.
"""
# progress-ok: the script prints each step ([1/4] .. [4/4], test N of 4) as it goes; the server's own output goes to a
# log file only so it can be quoted in the report if the run fails, it is not the progress.
import argparse, datetime, json, os, platform, re, shutil, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, 'arc_cascade.py')

def sh(cmd, env=None, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return (r.stdout or '') + (r.stderr or '')
    except Exception as e:
        return 'failed: %s' % e

def system_info(bin_dir):
    info = {}
    info['os'] = platform.platform(terse=True)
    if os.path.exists('/etc/os-release'):
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)', open('/etc/os-release').read(), re.M)
        if m: info['os'] = m.group(1)
    info['kernel'] = platform.release()
    cpu = platform.processor()
    if os.path.exists('/proc/cpuinfo'):
        m = re.search(r'^model name\s*:\s*(.+)$', open('/proc/cpuinfo').read(), re.M)
        if m: cpu = m.group(1).strip()
    info['cpu'] = cpu or 'unknown'
    if os.path.exists('/proc/meminfo'):
        mi = open('/proc/meminfo').read()
        tot = int(re.search(r'MemTotal:\s+(\d+)', mi).group(1)) // 1024
        av = int(re.search(r'MemAvailable:\s+(\d+)', mi).group(1)) // 1024
        info['ram'] = '%d MiB total, %d MiB available at start' % (tot, av)
    env = dict(os.environ, ONEAPI_DEVICE_SELECTOR='level_zero:*')
    sl = sh(['sycl-ls'], env=env) if shutil.which('sycl-ls') else 'sycl-ls not found (is the oneAPI environment loaded?)'
    info['sycl_ls'] = '\n'.join(l for l in sl.splitlines() if l.startswith('[')) or sl.strip()[:500]
    icpx = sh(['icpx', '--version']) if shutil.which('icpx') else ''
    info['compiler'] = icpx.splitlines()[0] if icpx else 'icpx not on PATH'
    commit = sh(['git', '-C', HERE, 'log', '-1', '--format=%h %s']).strip() if shutil.which('git') else ''
    branch = sh(['git', '-C', HERE, 'rev-parse', '--abbrev-ref', 'HEAD']).strip() if commit else ''
    bad = lambda s: not s or 'fatal' in s or 'failed' in s
    info['commit'] = 'unknown (not a git checkout)' if bad(commit) else commit.splitlines()[0][:120]
    info['branch'] = '' if bad(branch) else branch.splitlines()[0]
    exe = os.path.join(bin_dir, 'llama-server')
    ver = [l for l in sh([exe, '--version'], timeout=120).splitlines() if l.startswith('version')]
    info['server_version'] = ver[0] if ver else 'unknown'
    lst = sh([exe, '--list-devices'], env=dict(env, SYCL_UR_USE_LEVEL_ZERO_V2='0'), timeout=120)
    info['devices'] = [l.strip() for l in lst.splitlines() if re.match(r'\s*SYCL\d+:', l)]
    if not info['devices']:
        info['devices_error'] = lst.strip()[-600:]
    return info

def pick_card(devices):
    names = [d.split(':', 1)[1].rsplit('(', 1)[0].strip() for d in devices]
    for n in names:
        if 'Arc' in n: return n
    return names[0] if names else None

def post(port, path, obj, timeout=900):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path), json.dumps(obj).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def chat(port, prompt, n):
    t0 = time.time()
    r = post(port, '/v1/chat/completions', {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': n, 'temperature': 0,
                                            'chat_template_kwargs': {'enable_thinking': False}})
    t = r.get('timings', {})
    return {'prompt_tokens': t.get('prompt_n'), 'prompt_tps': round(t.get('prompt_per_second') or 0, 1),
            'tokens': t.get('predicted_n'), 'tps': round(t.get('predicted_per_second') or 0, 1),
            'drafted': t.get('draft_n'), 'accepted': t.get('draft_n_accepted'), 'wall_s': round(time.time() - t0, 1)}

def long_document(port, target=4000):
    # a fixed, made-up ledger: the same text on every machine, sized to about `target` tokens with the server's tokenizer
    names = ['Ada', 'Ben', 'Cleo', 'Dev', 'Eli', 'Fay', 'Gus', 'Hana', 'Ivo', 'Jun']
    items = ['lamp', 'kettle', 'bike', 'radio', 'drill', 'tent', 'camera', 'violin', 'ladder', 'clock']
    rows = ['Entry %d. %s lent the %s to %s on day %d.' % (k, names[(k * 7) % 10], items[(k * 3) % 10], names[(k * 5 + 1) % 10], (k * 37) % 365 + 1)
            for k in range(2000)]
    n = len(post(port, '/tokenize', {'content': ' '.join(rows)})['tokens'])
    return ' '.join(rows[:max(10, int(len(rows) * target / n))])

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--head', help='optional MTP draft head GGUF')
    ap.add_argument('--bin', default=None, help='folder with llama-server (default: build/bin next to this checkout)')
    ap.add_argument('--cards', help='card names as the launcher takes them (default: the first Intel Arc card llama-server lists)')
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--port', type=int, default=8379)
    ap.add_argument('--out', default=None, help='report file (default: arc-cascade-report-<date>.md in the current folder)')
    ap.add_argument('--launcher-args', default='', help='extra launcher options, quoted, e.g. "--ram-mib 12000"')
    a = ap.parse_args()

    bin_dir = a.bin or os.path.normpath(os.path.join(HERE, '..', '..', 'build', 'bin'))
    if not os.path.exists(os.path.join(bin_dir, 'llama-server')):
        sys.exit('llama-server not found in %s: build this branch first, or pass --bin' % bin_dir)
    if not os.path.exists(a.model):
        sys.exit('model not found: %s' % a.model)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    out = a.out or 'arc-cascade-report-%s.md' % stamp
    log_path = out.replace('.md', '') + '-server.log'

    print('[1/4] looking at this machine...', flush=True)
    info = system_info(bin_dir)
    card = a.cards or pick_card(info['devices'])
    if not card:
        sys.exit('llama-server lists no SYCL devices:\n%s' % info.get('devices_error', ''))
    print('      card: %s' % card, flush=True)

    base = [sys.executable, LAUNCHER]
    opts = ['--model', a.model, '--cards', card, '--ctx', str(a.ctx), '--bin', bin_dir] + (['--head', a.head] if a.head else []) + a.launcher_args.split()
    print("[2/4] the launcher's plan:", flush=True)
    plan = sh(base + ['plan'] + opts, timeout=300)
    print('\n'.join('      ' + l for l in plan.strip().splitlines()), flush=True)

    print('[3/4] starting the server (big tiered models can take a few minutes to fill their caches)...', flush=True)
    srv = subprocess.Popen(base + ['run'] + opts + ['--', '-np', '1', '--host', '127.0.0.1', '--port', str(a.port), '--temp', '0', '--cache-ram', '0'],
                           stdout=open(log_path, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    results, error = {}, None
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if srv.poll() is not None: raise RuntimeError('the server stopped while starting (see %s)' % log_path)
            try:
                if json.load(urllib.request.urlopen('http://127.0.0.1:%d/health' % a.port, timeout=2)).get('status') == 'ok': break
            except Exception: pass
            time.sleep(2)
        else:
            raise RuntimeError('the server did not become ready in 15 minutes')
        results['start_s'] = round(time.time() - t0)
        print('      ready after %d s' % results['start_s'], flush=True)
        code = 'Write a Python function that merges two sorted lists into one sorted list, with a docstring and two doctests.'
        prose = 'Explain in three short paragraphs why the sky is blue, for a ten-year-old.'
        tests = [('code, first ask', code, 256), ('code, asked again', code, 256), ('prose', prose, 256)]
        for k, (name, p, n) in enumerate(tests, 1):
            print('      test %d of 4: %s' % (k, name), flush=True)
            results[name] = chat(a.port, p, n)
            print('        %s t/s generation' % results[name]['tps'], flush=True)
        print('      test 4 of 4: ~4,000-token prompt', flush=True)
        doc = long_document(a.port)
        results['4K prompt'] = chat(a.port, doc + '\n\nWho lent the first item listed above, and what was it? Answer in one sentence.', 64)
        print('        %s t/s prompt, %s t/s generation' % (results['4K prompt']['prompt_tps'], results['4K prompt']['tps']), flush=True)
    except Exception as e:
        error = str(e)
    finally:
        try: os.killpg(srv.pid, signal.SIGTERM); srv.wait(30)
        except Exception:
            try: os.killpg(srv.pid, signal.SIGKILL)
            except Exception: pass

    print('[4/4] writing %s' % out, flush=True)
    L = ['# Arc Cascade test report', '',
         '- Date: %s' % datetime.date.today().isoformat(),
         '- Build: %s%s; llama-server %s' % (info['commit'], ' (%s)' % info['branch'] if info['branch'] else '', info['server_version']),
         '- OS: %s, kernel %s' % (info['os'], info['kernel']),
         '- CPU: %s' % info['cpu'], '- RAM: %s' % info.get('ram', 'unknown'),
         '- Compiler: %s' % info['compiler'],
         '- Model: %s (%.1f GB)%s' % (os.path.basename(a.model), os.path.getsize(a.model) / 1e9,
                                       ', draft head %s' % os.path.basename(a.head) if a.head else ''),
         '- Context: %d, card(s): %s%s' % (a.ctx, card, ', launcher options: %s' % a.launcher_args if a.launcher_args else ''), '',
         '## Devices', '```'] + info['devices'] + ['```', '', '## sycl-ls', '```', info['sycl_ls'], '```', '',
         '## Launcher plan', '```'] + [l for l in plan.strip().splitlines() if not l.startswith('command:')] + ['```', '']
    if error:
        tail = open(log_path, errors='replace').read().splitlines()[-25:] if os.path.exists(log_path) else []
        L += ['## Result: FAILED', '', error, '', '```'] + [re.sub(r'/[^ ]*/', '.../', l)[:200] for l in tail] + ['```']
    else:
        L += ['## Results', '', 'Server ready after %s s.' % results['start_s'], '',
              '| Test | Prompt tokens | Prompt t/s | Tokens out | Generation t/s | Drafted / accepted |', '|---|---|---|---|---|---|']
        for name in ('code, first ask', 'code, asked again', 'prose', '4K prompt'):
            r = results[name]
            d = '%s / %s' % (r['drafted'], r['accepted']) if r.get('drafted') else '-'
            L.append('| %s | %s | %s | %s | %s | %s |' % (name, r['prompt_tokens'], r['prompt_tps'], r['tokens'], r['tps'], d))
    open(out, 'w').write('\n'.join(L) + '\n')
    print('done. Read %s before sharing it; the server log is %s' % (out, log_path))
    if error: sys.exit(1)

if __name__ == '__main__':
    main()

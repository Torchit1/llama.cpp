#!/usr/bin/env python3
# Cascade Scope recorder: sends one streamed request to llama-server and merges what the client saw (each text piece and
# when it arrived) with the server's scope timeline (LLAMA_SCOPE_FILE: server loop phases, decodes, MoE cache activity)
# into one session file that the Cascade Scope page draws. No dependencies beyond Python 3.
#
#   LLAMA_SCOPE_FILE=/path/scope.jsonl llama-server ...          (start the server with the timeline on)
#   cascade_scope.py record http://127.0.0.1:8080 /path/scope.jsonl out.json --label NAME [--prompt TEXT | --prompt-file F]
#                    [--max-tokens 200] [--extra '{"cache_prompt": true}']
#
# Times in the session are milliseconds from the moment the request was sent. The timeline costs two short file writes
# per decode; it does not add waits on the GPU, so the "decode" phase is the time to queue the work and the GPU time
# shows up in the phase that reads the result (sample).
import argparse, json, sys, time, urllib.request

PHASES = ['outside', 'pre', 'draft', 'checkpoint', 'decode', 'sample', 'post']

def read_scope(path, w0, w1):
    """records between wall times w0 and w1 (seconds), times converted to ms after w0"""
    off = {}; last = {}; phases = []; decodes = []
    for line in open(path, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        k = r.get('k')
        if k == 'h':
            off[r['src']] = (r['wall_us'] - r['us']) / 1e6
            if r['src'] == 'llama': last = {}
        elif k == 'p' and 'server' in off:
            t = r['t'] / 1e6 + off['server']
            if w0 <= t <= w1 + 0.05: phases.append([r['id'], round((t - w0) * 1000, 3)])
        elif k == 'd' and 'llama' in off:
            prev = last.get(r['c']); last[r['c']] = r
            t3 = r['t3'] / 1e6 + off['llama']
            if not (w0 <= t3 <= w1 + 0.05): continue
            ms = lambda key: round((r[key] / 1e6 + off['llama'] - w0) * 1000, 3)
            d = {'c': r['c'], 'layers': r['layers'], 'n': r['n'], 't0': ms('t0'), 't1': ms('t1'), 't2': ms('t2'), 't3': ms('t3')}
            for key in ('hit', 'miss', 'ram', 'reads', 'read_b', 'up_b'):
                d[key] = r[key] - (prev[key] if prev else 0)
            decodes.append(d)
    return phases, decodes

def record(a):
    prompt = open(a.prompt_file, errors='ignore').read() if a.prompt_file else a.prompt
    req = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}, "cache_prompt": False, "timings_per_token": True,
           "chat_template_kwargs": {"enable_thinking": False}}
    req.update(json.loads(a.extra or '{}'))
    chunks = []; usage = {}; tm = {}
    w0 = time.time()
    resp = urllib.request.urlopen(urllib.request.Request(a.server.rstrip('/') + '/v1/chat/completions', json.dumps(req).encode(),
                                  {"Content-Type": "application/json", "Authorization": "Bearer local"}), timeout=1800)
    for line in resp:
        line = line.decode('utf-8', 'replace').strip()
        if not line.startswith('data:') or line.endswith('[DONE]'): continue
        d = json.loads(line[5:]); now = time.time()
        if d.get('usage'): usage = d['usage']
        if d.get('timings'): tm = d['timings']
        for ch in d.get('choices', []):
            delta = ch.get('delta', {})
            piece = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
            if piece:
                chunks.append([round((now - w0) * 1000, 3), piece, tm.get('predicted_n'), tm.get('draft_n'), tm.get('draft_n_accepted')])
    w1 = time.time()
    phases, decodes = read_scope(a.scope, w0, w1)
    ctxs = {}
    for d in decodes: ctxs.setdefault(d['c'], d['layers'])
    main = max(ctxs, key=lambda c: ctxs[c]) if ctxs else None      # the model with the most layers is the main one
    for d in decodes: d['main'] = d['c'] == main; del d['c']
    out = {'version': 1, 'label': a.label, 'recorded': time.strftime('%Y-%m-%d %H:%M', time.localtime(w0)), 'note': a.note or '',
           'prompt_chars': len(prompt), 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
           'total_ms': round((w1 - w0) * 1000, 1), 'timings': tm, 'phase_names': PHASES, 'chunks': chunks, 'phases': phases, 'decodes': decodes}
    json.dump(out, open(a.out, 'w'), separators=(',', ':'))
    n = usage.get('completion_tokens') or len(chunks)
    gen = (chunks[-1][0] - chunks[0][0]) / 1000 if len(chunks) > 1 else 0
    print('%s: %d tokens, first after %.2f s, %.1f t/s, %d phase marks, %d decodes -> %s' % (
        a.label, n, chunks[0][0] / 1000 if chunks else 0, (n - 1) / gen if gen > 0 else 0, len(phases), len(decodes), a.out))

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['record'])
ap.add_argument('server'); ap.add_argument('scope'); ap.add_argument('out')
ap.add_argument('--label', default='run'); ap.add_argument('--note')
ap.add_argument('--prompt', default='Write a Python function that merges overlapping intervals, with a docstring and three test cases. Then explain its complexity.')
ap.add_argument('--prompt-file'); ap.add_argument('--max-tokens', type=int, default=200); ap.add_argument('--extra')
record(ap.parse_args())

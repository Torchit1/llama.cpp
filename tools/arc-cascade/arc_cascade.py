#!/usr/bin/env python3
# Arc Cascade launcher: give it a model, the cards to use and a RAM budget; it decides where the model goes
# (one card / two cards / VRAM expert cache + pinned RAM + the file on disk), prints the plan, and with `run` starts
# llama-server that way. No dependencies beyond Python 3; sizes come from the GGUF header.
#
#   arc_cascade.py plan --model M.gguf [--head HEAD.gguf] [--cards B70,B580] [--ram-mib N] [--ctx 8192]
#   arc_cascade.py run  --model M.gguf ... [--bin DIR] [--remember | --usage-file F] [-- extra llama-server arguments]
#
# Rules, in order (measured on the B70 + B580 box, 15 Nov 2026):
#   1. The model fits on the first card: use that card alone. A second card makes a model that fits 17-25% slower.
#   2. It fits across the listed cards: split it over them, no tiers.
#   3. Otherwise mixture-of-experts models get tiers: the non-expert weights and an expert cache in VRAM (over all
#      listed cards), the most recently used of the other experts pinned in RAM, the rest read from the file.
#      Dense models that do not fit are refused: the tiers only hold experts.
# RAM: the server needs about 3.5 GiB beyond the pinned tier (measured 2.0-3.4), so the tier is the budget minus that.
# Without --ram-mib the budget is what is free now minus a 5 GiB margin, and never more than is free.
import argparse, glob, os, shutil, struct, subprocess, sys

CARDS = {'B70': 32656, 'B580': 12216}          # usable MiB as the driver reports it
SERVER_RAM_MIB = 3500                         # host memory the server uses beyond the pinned RAM tier
# context: 32 MiB of VRAM per 1,024 tokens, working memory that grows with it included (Flash-Next at 262K, 16 Nov 2026)
RESERVE_MIB = 600                             # per card, left free
TWO_CARD_MARGIN_MIB = 4000                    # extra room when the cache is split over cards (one run so far, at 32000)
WIDE_PASS_MIB = 700                           # working memory of a wide (up to 128-token) checking pass
MiB = 2 ** 20

def gguf_tensors(path):
    """(name, bytes) of every tensor, from the GGUF header alone: a tensor's size is the gap to the next offset."""
    with open(path, 'rb') as f:
        def u32(): return struct.unpack('<I', f.read(4))[0]
        def u64(): return struct.unpack('<Q', f.read(8))[0]
        def string(): return f.read(u64()).decode('utf-8', 'replace')
        assert f.read(4) == b'GGUF', '%s is not a GGUF file' % path
        u32(); n_tensors, n_kv = u64(), u64()
        scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        def skip(t):
            if t == 8: f.seek(u64(), 1)
            elif t == 9:
                et, n = u32(), u64()
                if et in scalar: f.seek(scalar[et] * n, 1)
                else:
                    for _ in range(n): skip(et)
            else: f.seek(scalar[t], 1)
        align = 32
        for _ in range(n_kv):
            key = string(); t = u32()
            if key == 'general.alignment' and t == 4: align = u32()
            else: skip(t)
        infos = []
        for _ in range(n_tensors):
            name = string(); nd = u32(); f.seek(8 * nd, 1); u32(); infos.append((name, u64()))
        data_start = (f.tell() + align - 1) // align * align
        size = os.path.getsize(path) - data_start
    infos.sort(key=lambda x: x[1])
    return [(n, (infos[k + 1][1] if k + 1 < len(infos) else size) - o) for k, (n, o) in enumerate(infos)]

def sizes(path):
    files = sorted(glob.glob(path.replace('-00001-of-', '-*-of-'))) if '-00001-of-' in path else [path]
    s = {'exps': 0, 'lazy': 0, 'other': 0}
    for f in files:
        for name, nbytes in gguf_tensors(f):
            k = 'exps' if '_exps.' in name else ('lazy' if 'per_layer_token_embd' in name else 'other')
            s[k] += nbytes
    return {k: v / MiB for k, v in s.items()}

def free_ram_mib():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable:'): return int(l.split()[1]) // 1024
    return 0

def plan(a):
    m = sizes(a.model); h = sizes(a.head) if a.head else None
    head_w = (h['exps'] + h['other']) if h else 0
    cards = [c.strip() for c in a.cards.split(',') if c.strip()]
    vram = [int(v) for v in a.vram_mib.split(',')] if a.vram_mib else [CARDS[c] for c in cards]
    assert len(vram) == len(cards), '--vram-mib needs one value per card'
    free = free_ram_mib()
    ram_budget = min(a.ram_mib, free) if a.ram_mib else max(0, free - 5000)
    out = {'lines': [], 'env': {}, 'args': ['-m', a.model, '-fa', 'on', '-c', str(a.ctx)], 'cards': cards[:1], 'ok': True}
    L = out['lines'].append
    L('model: %.0f MiB of experts, %.0f MiB of other weights%s%s' % (m['exps'], m['other'],
        ', %.0f MiB on-disk table' % m['lazy'] if m['lazy'] else '', '; draft head %.0f MiB' % head_w if h else ''))
    L('cards: %s; RAM free now %d MiB, budget %d MiB' % (', '.join('%s %d MiB' % cv for cv in zip(cards, vram)), free, ram_budget))
    if h: out['args'] += ['-md', a.head, '--spec-type', 'draft-mtp', '-ngld', '99', '-bs', '--spec-draft-n-max', '5']
    # Measured on a B70 with the Flash-Next Coder and its head (16 Nov): the card's used memory grows by 33.0 MiB per 1,024
    # tokens of context with the 16-bit attention cache and by 22.6 with the 8-bit one, from 30,430 MiB at none.
    # Tier loads need more than that (16 Nov: plans sized from the one-card figures failed to create the head's
    # context), so the tier sums keep the figures that the 262K tier loads were fitted to.
    ctx_mib = (21.6 if a.cache_bits == 8 else 32.0) + (4.0 if h else 0)
    if a.cache_bits == 8: out['args'] += ['-ctk', 'q8_0', '-ctv', 'q8_0']
    def fixed_base(ub): return m['other'] + 524 + a.ctx * ctx_mib / 1024 + (195 + 0.303 * ub) + ((head_w + 0.62 * ub) if h else 0)
    # what the one-card figures overstate when everything is on the card: 400 MiB and 3 MiB per 1,024 tokens with a head
    one_card_less = 400 + (a.ctx * 3.0 / 1024 if h else 0)
    # copy-drafting is on whenever the model and its wide checking passes fit one card; --copy-drafts / --no-copy-drafts decide otherwise
    copy = a.copy_drafts
    if copy is None:
        copy = m['exps'] + fixed_base(512) - one_card_less + WIDE_PASS_MIB + RESERVE_MIB <= vram[0]
        L('copy-drafting: %s' % ('on (fits this card with its %d MiB of working memory)' % WIDE_PASS_MIB if copy else
                                 'off (its %d MiB of working memory does not fit next to everything else; --copy-drafts forces it)' % WIDE_PASS_MIB))
    if copy:
        # copy-drafting picks its own length, from 8 up to 128 tokens (LLAMA_SPEC_NGRAM_AUTO)
        out['args'] += ['--spec-type', 'ngram-mod', '--spec-ngram-mod-n-min', '16', '--spec-ngram-mod-n-max', '128']
        out['env']['LLAMA_SPEC_NGRAM_AUTO'] = '1'
    def fixed(ub): return fixed_base(ub) + (WIDE_PASS_MIB if copy else 0)
    whole = m['exps'] + fixed(512)
    # 1. one card
    if whole - one_card_less + RESERVE_MIB <= vram[0]:
        L('plan: everything on the %s (%.0f of %d MiB)' % (cards[0], whole, vram[0]))
        out['args'] += ['-ngl', '99', '-lm', 'dio']
        return out
    # 2. all of it across the cards
    if len(cards) > 1 and whole + RESERVE_MIB * len(cards) + TWO_CARD_MARGIN_MIB <= sum(vram):
        L('plan: split over %s, all in VRAM (%.0f of %d MiB); expect it slower per token than a model that fits one card' % (
            ' + '.join(cards), whole, sum(vram)))
        out['cards'] = cards; out['args'] += ['-ngl', '99', '-lm', 'dio']
        return out
    # 3. tiers
    if m['exps'] < 0.5 * (m['exps'] + m['other']):
        out['ok'] = False
        L('REFUSED: the model needs %.0f MiB and only %.0f MiB of it is experts. The tiers hold experts only, so a mostly dense '
          'model this size cannot be placed. Use a smaller quant.' % (whole, m['exps']))
        return out
    use = cards if len(cards) > 1 else cards[:1]
    total = sum(vram[:len(use)])
    margin = RESERVE_MIB * len(use) + (TWO_CARD_MARGIN_MIB if len(use) > 1 else 0)
    choice = None
    for ub in ((a.prompt_batch,) if a.prompt_batch else (1024, 512, 256)):   # 1024 is the prompt batch the tier runs were measured with
        cache = total - fixed(ub) - margin
        if cache >= 3000:
            choice = (ub, cache); break
    if not choice:
        out['ok'] = False
        L('REFUSED: after the non-expert weights and working memory (%.0f MiB) there is no room for an expert cache on %s.' % (fixed(256), ' + '.join(use)))
        return out
    ub, cache = choice
    cache = int(min(cache, m['exps']) // 100 * 100)
    off_card = m['exps'] - cache
    ram_tier = int(max(0, min(ram_budget - SERVER_RAM_MIB, off_card)) // 100 * 100)
    disk = max(0, off_card - ram_tier)
    if off_card > 0 and ram_tier < min(2000, off_card):
        out['ok'] = False
        L('REFUSED: %.0f MiB of experts do not fit in VRAM and the RAM budget of %d MiB leaves a pinned tier of %d MiB (the server '
          'itself needs about %d). Free some RAM, raise --ram-mib, or add a card.' % (off_card, ram_budget, ram_tier, SERVER_RAM_MIB))
        return out
    L('plan: tiers on %s' % ' + '.join(use))
    L('  VRAM : %.0f MiB of non-expert weights and working memory + %d MiB expert cache' % (fixed(ub), cache))
    L('  RAM  : %d MiB of experts pinned, most recently used first (server total about %d MiB of the %d MiB budget)' % (ram_tier, ram_tier + SERVER_RAM_MIB, ram_budget))
    L('  disk : %.0f MiB of experts read from the file when needed' % disk)
    L('  prompt batch %d tokens' % ub)
    out['cards'] = use
    out['env'].update({'LLAMA_MOE_HOST_PIN_MIB': '0', 'LLAMA_MOE_RAM_CACHE_MIB': str(ram_tier)})
    if a.preload:
        out['env']['LLAMA_MOE_PRELOAD'] = '1'
        L('  start: the expert cache and the RAM tier are filled from the file before the first request (--no-preload fills them as experts are used)')
    out['args'] += ['-ngl', '99', '-lm', 'none', '-lzm', 'on', '-cmoe', '--moe-cache-mib', str(cache), '-b', str(ub), '-ub', str(ub)]
    return out

def card_selector(cards):
    """level_zero indices of the named cards, from sycl-ls (same rule as the job runner's card())."""
    txt = subprocess.run(['sycl-ls'], env=dict(os.environ, ONEAPI_DEVICE_SELECTOR='level_zero:*'), capture_output=True, text=True).stdout
    lines = [l for l in txt.splitlines() if l.startswith('[level_zero')]
    idx = []
    for c in cards:
        hit = [i for i, l in enumerate(lines) if c in l]
        if not hit: sys.exit('card %s not found by sycl-ls' % c)
        idx.append(str(hit[0]))
    return 'level_zero:' + ','.join(idx)

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['plan', 'run'])
ap.add_argument('--model', required=True); ap.add_argument('--head')
ap.add_argument('--cards', default='B70'); ap.add_argument('--vram-mib')
ap.add_argument('--ram-mib', type=int); ap.add_argument('--ctx', type=int, default=8192)
ap.add_argument('--bin', default=os.path.dirname(shutil.which('llama-server') or '') or '.')
ap.add_argument('--threads', type=int, default=12)
# usage memory, tiers only, off by default: the experts in VRAM and RAM at shutdown are loaded back at the next start
ap.add_argument('--remember', action='store_true'); ap.add_argument('--usage-file')
# copy-drafting for edits: long drafts copied from text already in the conversation, checked in wide passes
ap.add_argument('--copy-drafts', action=argparse.BooleanOptionalAction, default=None)
# attention cache precision: 8 keeps about 30% more context per MiB, the answers differ slightly from 16
ap.add_argument('--cache-bits', type=int, choices=[8, 16], default=16)
# tiers only: prompt batch (default: the largest of 1024 / 512 / 256 that leaves room for an expert cache)
ap.add_argument('--prompt-batch', type=int)
# tiers only: fill the expert cache and the RAM tier at start, so no expert is read from the file mid-answer
ap.add_argument('--preload', action=argparse.BooleanOptionalAction, default=True)
# everything after a bare -- goes to llama-server unchanged
argv = sys.argv[1:]
extra = []
if '--' in argv:
    cut = argv.index('--'); argv, extra = argv[:cut], argv[cut + 1:]
a = ap.parse_args(argv)
p = plan(a)
for l in p['lines']: print(l)
if not p['ok']: sys.exit(2)
args = p['args'] + ['-t', str(a.threads)] + extra
env = dict(p['env'], SYCL_UR_USE_LEVEL_ZERO_V2='0', GGML_SYCL_MMID_SCHED='3')
if (a.remember or a.usage_file) and 'LLAMA_MOE_RAM_CACHE_MIB' in p['env']:
    uf = a.usage_file or os.path.join(os.path.expanduser('~/.cache/arc-cascade'),
        '%s.%d.usage' % (os.path.basename(a.model), os.path.getsize(a.model)))
    os.makedirs(os.path.dirname(uf) or '.', exist_ok=True)
    env['LLAMA_MOE_USAGE_FILE'] = uf
    print('usage memory: ON, %s (%s). Not for speed measurements: a preloaded cache hides the cold start.' % (
        uf, 'found, experts from the last run are preloaded' if os.path.exists(uf) else 'not there yet, written at shutdown'))
elif a.remember or a.usage_file:
    print('usage memory: not used, the whole model is in VRAM')
print('cards used: ' + ' + '.join(p['cards']))
print('command: ' + ' '.join('%s=%s' % kv for kv in env.items()) + ' llama-server ' + ' '.join(args))
if a.mode == 'run':
    env['ONEAPI_DEVICE_SELECTOR'] = card_selector(p['cards'])
    exe = os.path.join(a.bin, 'llama-server')
    lib = a.bin + ((':' + os.environ['LD_LIBRARY_PATH']) if os.environ.get('LD_LIBRARY_PATH') else '')
    sys.stdout.flush()
    os.execve(exe, [exe] + args, dict(os.environ, LD_LIBRARY_PATH=lib, **env))

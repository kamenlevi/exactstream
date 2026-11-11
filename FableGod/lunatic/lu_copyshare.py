# LUNATIC round 2, Route C falsification: how much of an agent's written output re-emits text it has already seen?
# Workload proxy: a week of Kamen's Claude Code transcripts (aggregates only). Line-level: an output line counts as a
# copy when the same stripped line (>= 8 non-space chars) appeared earlier in the session (prompts, tool results,
# earlier outputs). A diff-anchored drafter (prompt lookup) could draft exactly those lines.
import json, os, re, sys, time

root = os.path.expanduser('~/.claude/projects')
cut = time.time() - 7 * 86400
NUM = re.compile(r'^\s*\d+[\t→]')

def lines_of(s):
    for ln in s.split('\n'):
        yield NUM.sub('', ln, count=1).strip()

def texts(content):
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for it in content:
            if isinstance(it, dict):
                if it.get('type') == 'text':
                    yield it.get('text', '')
                elif it.get('type') == 'tool_result':
                    yield from texts(it.get('content'))

tot = dict(files=0, out_all=0, out_file=0, cov_file=0, cov_file_long=0, n_write=0, n_edit=0,
           w_out=0, w_cov=0, e_out=0, e_cov=0)
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(dp, f)
        if not f.endswith('.jsonl') or os.path.getmtime(p) < cut or os.path.getsize(p) < 100_000:
            continue
        tot['files'] += 1
        seen = set()
        for raw in open(p, errors='replace'):
            try:
                r = json.loads(raw)
            except Exception:
                continue
            msg = r.get('message') or {}
            role = msg.get('role') or r.get('type')
            content = msg.get('content')
            if role == 'assistant' and isinstance(content, list):
                for it in content:
                    if not isinstance(it, dict):
                        continue
                    if it.get('type') == 'text':
                        t = it.get('text', ''); tot['out_all'] += len(t)
                        seen.update(l for l in lines_of(t) if len(l.replace(' ', '')) >= 8)
                    elif it.get('type') == 'tool_use':
                        inp = it.get('input') or {}
                        name = it.get('name')
                        outs = []
                        if name == 'Write':
                            outs = [('w', inp.get('content', ''))]
                        elif name == 'Edit':
                            outs = [('e', inp.get('new_string', ''))]
                        elif name == 'MultiEdit':
                            outs = [('e', e.get('new_string', '')) for e in inp.get('edits', [])]
                        else:
                            tot['out_all'] += len(json.dumps(inp))
                        for kind, s in outs:
                            if not isinstance(s, str):
                                continue
                            tot['n_write' if kind == 'w' else 'n_edit'] += 1
                            n = len(s); tot['out_all'] += n; tot['out_file'] += n
                            cov = 0
                            for ln in s.split('\n'):
                                st = ln.strip()
                                if len(st.replace(' ', '')) >= 8 and st in seen:
                                    cov += len(ln) + 1
                            cov = min(cov, n)
                            tot['cov_file'] += cov
                            tot[kind + '_out'] += n; tot[kind + '_cov'] += cov
                            seen.update(l for l in lines_of(s) if len(l.replace(' ', '')) >= 8)
            else:
                for t in texts(content):
                    seen.update(l for l in lines_of(t) if len(l.replace(' ', '')) >= 8)

t = tot
print(f"transcripts (7 days, > 100 KB): {t['files']}; Write calls {t['n_write']}, Edit/MultiEdit edits {t['n_edit']}")
print(f"assistant output chars: all {t['out_all'] / 1e6:.1f} M (prose + tool inputs), file-writing {t['out_file'] / 1e6:.1f} M "
      f"= {100 * t['out_file'] / max(1, t['out_all']):.1f} % of all output")
print(f"copied lines (seen earlier in the session) in file-writing output: {100 * t['cov_file'] / max(1, t['out_file']):.1f} %"
      f"  (Write {100 * t['w_cov'] / max(1, t['w_out']):.1f} %, Edit {100 * t['e_cov'] / max(1, t['e_out']):.1f} %)")
print(f"copied share of ALL assistant output: {100 * t['cov_file'] / max(1, t['out_all']):.1f} %")

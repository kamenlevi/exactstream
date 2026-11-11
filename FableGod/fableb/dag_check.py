"""dag_check.py (Fable B, mission B2): the 16-row check's graph as a DAG, from Leap's mx.export_to_dot of one 12-layer
check (R-11b, ~/Leap/data/r11b_check.dot; the Oct-27 engine: eager HC chain, MLX's quantized matmuls, pre-REF2 kernels).
Counts kernel-launching primitives (views excluded as in R-65), the critical path in kernels (the longest chain of true
data dependencies), the ASAP level widths (how many kernels could run at once if only data edges ordered them), and the
primitive mix on the critical path. CPU only, no numpy. Run: python3 dag_check.py ~/Leap/data/r11b_check.dot
The export names arrays with letters and reuses names across segments, so an edge counts only when the producer
appears earlier in the file than the consumer (the export is in creation order); later same-name producers are ignored."""
import re, sys
from collections import defaultdict, Counter, deque

VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Slice", "Squeeze", "Unflatten", "Transpose", "Flatten", "Split", "Depends"}

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Leap/data/r11b_check.dot"
label, pos = {}, {}             # primitive id -> label, file position
producers = defaultdict(list)   # array name -> [(pos, primitive)]
ins = defaultdict(list)         # primitive -> [(pos, array name)]
node_re = re.compile(r'^\{\s*(\d+)\s*\[label\s*=\s*"([^"]*)"')
edge_re = re.compile(r'^"?([^"\s]+)"?\s*->\s*"?([^"\s]+)"?\s*$')
n = 0
current, seen = {}, defaultdict(int)   # MLX's node ids are object addresses: a freed id can be reused later in the same export
for line in open(path):
    n += 1
    line = line.strip()
    m = node_re.match(line)
    if m:
        seen[m.group(1)] += 1
        key = "%s#%d" % (m.group(1), seen[m.group(1)])
        current[m.group(1)] = key
        label[key] = m.group(2); pos[key] = n
        continue
    m = edge_re.match(line)
    if m:
        a, b = m.group(1), m.group(2)
        if a.isdigit():
            producers[b].append((n, current[a]))
        elif b.isdigit():
            ins[current[b]].append((n, a))

prims = sorted(label, key=lambda p: pos[p])
is_kernel = {p: (label[p] not in VIEWS) for p in prims}
dup = sum(1 for v in producers.values() if len(v) > 1)
preds = {}
for p in prims:
    s = set()
    for epos, a in ins[p]:
        cands = [q for ppos, q in producers.get(a, []) if ppos < epos]
        if cands:
            s.add(cands[-1])              # the latest producer before this consumer
    preds[p] = sorted(s, key=lambda q: pos[q])
succ = defaultdict(list)
indeg = {p: 0 for p in prims}
for p in prims:
    for q in preds[p]:
        succ[q].append(p); indeg[p] += 1
order, dq = [], deque(p for p in prims if indeg[p] == 0)
while dq:
    p = dq.popleft(); order.append(p)
    for s in succ[p]:
        indeg[s] -= 1
        if indeg[s] == 0:
            dq.append(s)
assert len(order) == len(prims), (len(order), len(prims))
depth = {}
for p in order:
    depth[p] = max((depth[q] for q in preds[p]), default=0) + (1 if is_kernel[p] else 0)
tail = {}
for p in reversed(order):
    tail[p] = max((tail[s] for s in succ[p]), default=0) + (1 if is_kernel[p] else 0)

kernels = [p for p in prims if is_kernel[p]]
L = max(depth.values())
slack = {p: L - (depth[p] + tail[p] - 1) for p in kernels}
crit = [p for p in kernels if slack[p] == 0]
width = Counter(depth[p] for p in kernels)
w = sorted(width.values())
print("graph: %d primitives, %d kernel-launching (views excluded), %d array names (%d reused across segments)" % (len(prims), len(kernels), len(producers), dup))
print("critical path (longest chain of data-dependent kernels): %d kernels = %.1f%% of all kernels" % (L, 100.0 * L / len(kernels)))
print("kernels on some critical path (slack 0): %d (%.1f%%)" % (len(crit), 100.0 * len(crit) / len(kernels)))
print("ASAP level widths (mutually independent kernels per level): mean %.2f, median %d, max %d; levels of width 1: %d (%.0f%% of levels); width >= 4: %d levels"
      % (len(kernels) / L, w[len(w) // 2], w[-1], sum(1 for v in w if v == 1), 100.0 * sum(1 for v in w if v == 1) / L, sum(1 for v in w if v >= 4)))
print("slack (levels a kernel could move without lengthening the path): 0: %d; 1-3: %d; 4-20: %d; > 20: %d"
      % (sum(1 for p in kernels if slack[p] == 0), sum(1 for p in kernels if 1 <= slack[p] <= 3),
         sum(1 for p in kernels if 4 <= slack[p] <= 20), sum(1 for p in kernels if slack[p] > 20)))
print("primitive mix, all kernels:      " + ", ".join("%s %d" % kv for kv in Counter(label[p] for p in kernels).most_common(14)))
print("primitive mix, critical kernels: " + ", ".join("%s %d" % kv for kv in Counter(label[p] for p in crit).most_common(14)))
kpred = {p: [q for q in preds[p] if is_kernel[q]] for p in kernels}
print("kernels with exactly one kernel producer (pure chain links): %d (%.0f%%); with none (sources): %d; with >= 2 producers (joins): %d"
      % (sum(1 for p in kernels if len(kpred[p]) == 1), 100.0 * sum(1 for p in kernels if len(kpred[p]) == 1) / len(kernels),
         sum(1 for p in kernels if not kpred[p]), sum(1 for p in kernels if len(kpred[p]) >= 2)))
off = Counter(label[p] for p in kernels if slack[p] > 0)
print("off-critical kernels by primitive: " + ", ".join("%s %d" % kv for kv in off.most_common(12)))

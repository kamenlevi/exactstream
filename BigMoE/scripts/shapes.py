# Fetch safetensors headers (HTTP range) for every shard; write per-group param counts.
import json,re,struct,urllib.request,collections,sys,concurrent.futures as cf
repo=sys.argv[1]; out=sys.argv[2]
idx=json.load(open(sys.argv[3]))['weight_map']
base=f"https://huggingface.co/{repo}/resolve/main/"
def hdr(f):
    r=urllib.request.Request(base+f,headers={'Range':'bytes=0-7'})
    n=struct.unpack('<Q',urllib.request.urlopen(r).read())[0]
    r=urllib.request.Request(base+f,headers={'Range':f'bytes=8-{7+n}'})
    return json.loads(urllib.request.urlopen(r).read())
files=sorted(set(idx.values()))
shapes={}
with cf.ThreadPoolExecutor(16) as ex:
    for h in ex.map(hdr,files):
        for k,v in h.items():
            if k!='__metadata__': shapes[k]=(v['dtype'],v['shape'])
json.dump(shapes,open(out,'w'))
def grp(k):
    if k.startswith('model.visual'): return 'visual'
    if k.startswith('mtp'): return 'mtp'
    if 'ngram_embedding' in k: return 'ngram_tables'
    if '.ple.' in k: return 'ple_other'
    if 'mlp.experts' in k: return 'routed_experts'
    if 'shared_expert' in k: return 'shared_expert'
    if 'mlp.gate' in k: return 'router'
    if 'linear_attn' in k: return 'linear_attn'
    if 'self_attn' in k: return 'full_attn'
    if 'hyper_connection' in k: return 'hyper_conn'
    if 'embed_tokens' in k: return 'embed'
    if 'lm_head' in k: return 'lm_head'
    return 'other'
g=collections.Counter()
import math
for k,(dt,s) in shapes.items(): g[grp(k)]+=math.prod(s)
for k,v in g.most_common(): print(f"{k:16s} {v/1e9:8.3f} B")
print('total',sum(g.values())/1e9)
ex=[ (k,s) for k,(d,s) in shapes.items() if 'layers.0.' in k and ('experts' in k or 'ngram_embedding.shard_0' in k or 'ple' in k)]
for k,s in ex[:12]: print(k,s)

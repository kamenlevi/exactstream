# Where does a 31-deep chain end, and is that row flagged by the head's own p? (CPU, topk cells; teacher-forced rows)
# CORRECTED 08:5x: a topk cell's row is the TARGET position (anchor + depth); the anchor is row - depth.
import numpy as np
for s in ['heplus24','mbppbank']:
    z=np.load(f'/Users/kamen/BigMoE/data/deepdraft/ceiling/topk_t30b_{s}.npz')
    fi,row,d,r,tp,ws=z['fi'],z['row'],z['depth'],z['right'],z['top_p'][:,0],z['walk_start']
    an=row-d
    o=np.lexsort((d,an,fi)); fi,an,d,r,tp,ws=fi[o],an[o],d[o],r[o],tp[o],ws[o]
    key=fi*100000+an; starts=np.r_[0,np.flatnonzero(np.diff(key))+1]; ends=np.r_[starts[1:],len(key)]
    for name,sel in [('all anchors',lambda a:True),('walk starts',lambda a:ws[a])]:
        M=[];PM=[];P=[]
        for a,b in zip(starts,ends):
            if b-a<31 or not sel(a): continue
            assert (d[a:a+31]==np.arange(31)).all()
            rr=r[a:a+31]; pp=tp[a:a+31]
            miss=np.flatnonzero(~rr); m=miss[0] if len(miss) else 31
            M.append(m); PM.append(pp[m] if m<31 else 1.0); P.append(pp)
        M=np.array(M);PM=np.array(PM);P=np.array(P)
        print(f'{s} {name}: chains {len(M)}, first miss depth mean {M.mean():.2f}, full {np.mean(M==31):.3f}')
        for th in [0.5,0.8,0.9,0.95]:
            cov=((PM<th)|(M==31)).mean(); nch=(P<th).sum(1).mean()+1
            print(f'   p<{th}: end covered {cov:.3f} with {nch:.2f} chains/block')
        for k in [1,2,4]:
            c=sum(1 if m==31 else (m in np.argsort(pp)[:k]) for m,pp in zip(M,P))
            print(f'   lowest-{k} rows + end: covered {c/len(M):.3f} with {k+1} chains/block')

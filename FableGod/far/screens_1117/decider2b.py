"""BREAK98b (FAR): the frozen 2B / 4B as a DECIDER at the head's located misses, on the HE+ / MBPP topk cells.
phi = P(decider overrides at the chain's first miss and is right); psi = P(it overrides a right row). Walk as Lunatic's
floor_bar.py (chain-32 over windows, teacher-forced cells, no re-anchor). Price: one batched auditor pass per override
round (c ms), draft 3 ms per 31 levels, Leap's C6 + int8 bound 29.7 ms per check."""
import os, numpy as np
H=os.path.expanduser('~'); CE=f'{H}/BigMoE/data/deepdraft/ceiling'; MS=f'{H}/MoonShots/week98'
DEP=31; CHECK=7.6+16.4+1.2+1.5+3.0
def load(tag, rtag):
    z=np.load(f'{CE}/topk_t30b_{tag}.npz'); dep=z['depth'].astype(int); fi=z['fi']; row=z['row']; y=z['y']; right=z['right']; top=z['top_ids']; p1=z['top_p'][:,0]
    files=list(z['files'])
    aud={}
    for name in ('2b','4b'):
        try: a=np.load(f'{MS}/rows{name}_{rtag}.npz')
        except Exception as e: print('no',name,e); continue
        key={(f,int(r)):i for i,(f,r) in enumerate(zip(a['fname'],a['row']))}
        idx=np.array([key.get((str(files[f]),int(r)),-1) for f,r in zip(fi,row)])  # topk row = the target position (anchor + depth)
        ok=idx>=0; t1=np.full(len(dep),-1,np.int64); m=np.zeros(len(dep),np.float32); tk=np.full(len(dep),-2,np.int64)
        t1[ok]=a['top1'][idx[ok]]; m[ok]=a['margin2b'][idx[ok]]; tk[ok]=a['tk0'][idx[ok]]
        print(f'{tag} {name}: cells aligned {ok.mean():.3f}; tk0 == cell y on aligned {(tk[ok]==y[ok]).mean():.4f}')
        aud[name]=(t1,m,ok)
    return dep,right.astype(int),p1,y,top[:,0],top[:,1],aud
def windows(dep, *cols):
    st=np.flatnonzero((dep==0)&np.append(True,dep[:-1]!=0)); en=np.append(st[1:],len(dep)); out=[]
    for s,t in zip(st,en):
        dw=dep[s:t]; n0=int((dw==0).sum()); W=[]
        for c in cols:
            R=np.full((n0+1,DEP),-1,c.dtype if c.dtype!=bool else np.int8)
            for j in range(DEP):
                mm=dw==j; k=int(mm.sum()); R[:k,j]=c[s:t][mm]
            W.append(R)
        out.append(W)
    return out
def run(W, theta, tau, cms, rmax, use):
    """override at a row if head p1 < theta and auditor margin >= tau and auditor top1 != head top1. Returns kept, tok/s, phi, psi, rounds/check."""
    tok=0; nchk=0; ms=0.0; nfm=0; nphi=0; nright=0; npsi=0; rounds=0
    for R,P,DEC,AGR,OK in W:
        n=R.shape[0]; p=1
        while p<n:
            acc=0; k=0; used=0; extra=0.0
            while acc<31 and k<31 and R[p,k]>=0:
                flag = (P[p,k]<theta) and OK[p,k] and (not AGR[p,k]) and (use is None or use[p,k]>=tau)
                if R[p,k]==1:
                    nright+=1
                    if flag and used<rmax:
                        used+=1; extra+=cms+3.0*(31-acc)/31; npsi+=1; break   # false fix ends the block
                    acc+=1; k+=1; continue
                nfm+=1
                if flag and used<rmax:
                    used+=1; extra+=cms+3.0*(31-acc)/31
                    if DEC[p,k]==1: nphi+=1; acc+=1; k+=1; continue
                break
            rounds+=used; tok+=acc+1; nchk+=1; ms+=CHECK+extra; p+=acc+1
    return tok/nchk, 1000*tok/ms, nphi/max(nfm,1), npsi/max(nright,1), rounds/nchk
for tag,rtag,ref in (('heplus24','heplus',(462,566)),('mbppbank','mbpp',(232,321))):
    dep,right,p1,y,h1,h2,aud=load(tag,rtag)
    print(f'== {tag}: chain-32 + auditor decider; own chain {ref[0]} tok/s, tree-32 {ref[1]} at the bound')
    # information: at first misses (teacher-forced chain), who is right?
    for name,(t1,m,ok) in aud.items():
        W=windows(dep,right,p1,(t1==y).astype(np.int8),(t1==h1).astype(np.int8),ok.astype(np.int8),m)
        # first-miss stats
        fm=[];rt=[]
        for R,P,DEC,AGR,OK,M in W:
            for p in range(1,R.shape[0]):
                k=0
                while k<31 and R[p,k]==1: k+=1
                if k<31 and R[p,k]==0 and OK[p,k]: fm.append((P[p,k],DEC[p,k],AGR[p,k],M[p,k]))
        fm=np.array(fm,dtype=float)
        print(f'  {name}: first misses {len(fm)}: auditor right {fm[:,1].mean():.3f}; disagrees with head {1-fm[:,2].mean():.3f}; right & head p1<0.8 {np.mean(fm[:,1]*(fm[:,0]<0.8)):.3f}; right at margin>=2 {np.mean(fm[:,1]*(fm[:,3]>=2)):.3f}')
        Wr=[(R,P,DEC,AGR,OK) for R,P,DEC,AGR,OK,M in W]; Wm=[M for *_,M in W]
        for theta in (0.5,0.8,0.95):
            for tau in (0.0,1.0,2.0,4.0):
                for rmax in (2,99):
                    Wt=[(R,P,DEC,AGR,OK) for R,P,DEC,AGR,OK,M in W]
                    k,t,phi,psi,rr=run([(R,P,DEC,AGR,OK) for R,P,DEC,AGR,OK,M in W],theta,tau,4.0,rmax,None) if False else (None,)*5
                    # proper call with margin gate
                    res=run_m=None
                    def runm():
                        return run([(R,P,DEC,AGR,OK) for R,P,DEC,AGR,OK,M in W],theta,tau,4.0,rmax,None)
                    pass
                    Wg=[(R,P,DEC*(M>=tau),AGR|(M<tau),OK) for R,P,DEC,AGR,OK,M in W]
                    k,t,phi,psi,rr=run(Wg,theta,0,4.0,rmax,None)
                    print(f'   {name} theta {theta} tau {tau} rmax {rmax:2d}: kept {k:5.2f} ({t:3.0f} tok/s @ c 4 ms) phi {phi:.3f} psi {100*psi:.2f} % rounds/check {rr:.2f}')
    # the head's own #2 as the decider at flagged rows (= a tree sibling), for scale
    W=windows(dep,right,p1,(h2==y).astype(np.int8),np.zeros(len(dep),np.int8),np.ones(len(dep),np.int8))
    for theta in (0.5,0.8):
        for rmax in (2,99):
            k,t,phi,psi,rr=run(W,theta,0,0.3,rmax,None)
            print(f'   head #2 theta {theta} rmax {rmax:2d}: kept {k:5.2f} ({t:3.0f} @ c 0.3) phi {phi:.3f} psi {100*psi:.2f} % rounds {rr:.2f}')

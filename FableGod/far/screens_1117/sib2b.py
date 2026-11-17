"""BREAK98b (FAR): the 2B / 4B as a SIBLING PROPOSER (no override, no psi): at the chain's first miss, a sibling row holds
the auditor's token (if it differs from the head's #1) and / or the head's #2. Unbudgeted upper bounds (Lunatic's
'own + rank<=1 sibling 19.26' form): the chain continues teacher-forced after a caught miss. Also the catch rates."""
import os, numpy as np
H=os.path.expanduser('~'); CE=f'{H}/BigMoE/data/deepdraft/ceiling'; MS=f'{H}/MoonShots/week98'; DEP=31
def load(tag,rtag):
    z=np.load(f'{CE}/topk_t30b_{tag}.npz'); dep=z['depth'].astype(int); fi=z['fi']; row=z['row']; y=z['y']; right=z['right'].astype(int); top=z['top_ids']; p1=z['top_p'][:,0]
    files=list(z['files']); out={}
    for name in ('2b','4b'):
        a=np.load(f'{MS}/rows{name}_{rtag}.npz'); key={(str(f),int(r)):i for i,(f,r) in enumerate(zip(a['fname'],a['row']))}
        idx=np.array([key.get((str(files[f]),int(r)),-1) for f,r in zip(fi,row)]); assert (idx>=0).all()
        out[name]=(a['top1'][idx]==y).astype(np.int8), (a['top1'][idx]!=top[:,0]).astype(np.int8), a['margin2b'][idx]
    return dep,right,p1,(top[:,1]==y).astype(np.int8),(top[:,2]==y).astype(np.int8),out
def windows(dep,*cols):
    st=np.flatnonzero((dep==0)&np.append(True,dep[:-1]!=0)); en=np.append(st[1:],len(dep)); W=[]
    for s,t in zip(st,en):
        dw=dep[s:t]; n0=int((dw==0).sum()); Rs=[]
        for c in cols:
            R=np.full((n0+1,DEP),-1,np.int8 if c.dtype!=np.float32 else np.float32)
            for j in range(DEP):
                mm=dw==j; k=int(mm.sum()); R[:k,j]=c[s:t][mm]
            Rs.append(R)
        W.append(Rs)
    return W
def kept(W, catch, nsib_max=99):
    """walk; at a miss, if catch(row) the sibling holds y: continue (one sibling row spent); else the block ends."""
    tok=0;n=0;sib=0
    for Rs in W:
        R=Rs[0]; m=R.shape[0]; p=1
        while p<m:
            acc=0;k=0;used=0
            while acc<31 and k<31 and R[p,k]>=0:
                if R[p,k]==1: acc+=1;k+=1;continue
                if used<nsib_max and catch(Rs,p,k): used+=1; acc+=1; k+=1; continue
                break
            tok+=acc+1; n+=1; sib+=used; p+=acc+1
    return tok/n, sib/n
for tag,rtag in (('heplus24','heplus'),('mbppbank','mbpp')):
    dep,right,p1,h2,h3,aud=load(tag,rtag)
    d2,a2,m2=aud['2b']; d4,a4,m4=aud['4b']
    W=windows(dep,right,p1,h2,h3,d2,a2,m2,d4,a4,m4)
    # catch rates at first misses
    fm=[]
    for Rs in W:
        R=Rs[0]
        for p in range(1,R.shape[0]):
            k=0
            while k<31 and R[p,k]==1: k+=1
            if k<31 and R[p,k]==0: fm.append([Rs[i][p,k] for i in range(1,10)])
    fm=np.array(fm,dtype=float); P1,H2,H3,D2,A2,M2,D4,A4,M4=fm.T
    print(f'== {tag}: first misses {len(fm)}: head#2 {H2.mean():.3f} head#2|#3 {np.maximum(H2,H3).mean():.3f} 2B {D2.mean():.3f} 4B {D4.mean():.3f} | head#2 or 2B {np.maximum(H2,D2).mean():.3f} | #2|#3 or 2B {np.maximum(np.maximum(H2,H3),D2).mean():.3f} | #2 or 4B {np.maximum(H2,D4).mean():.3f} | 2B right where #2 wrong {np.mean(D2*(1-H2)):.3f}')
    for th in (0.8,0.95,1.01):
        f=P1<th; print(f'   flagged p1<{th}: share of first misses {f.mean():.3f}; within: head#2 {H2[f].mean():.3f} 2B {D2[f].mean():.3f} #2 or 2B {np.maximum(H2,D2)[f].mean():.3f}')
    base,_=kept(W,lambda Rs,p,k: False)
    print(f'   own chain {base:.2f}')
    for name,fn in (('+ head #2 sibling at every miss',lambda Rs,p,k: Rs[2][p,k]==1),
                    ('+ 2B sibling at every miss (unbudgeted)',lambda Rs,p,k: Rs[4][p,k]==1),
                    ('+ head #2 and 2B siblings at every miss',lambda Rs,p,k: Rs[2][p,k]==1 or Rs[4][p,k]==1),
                    ('+ head #2,#3 and 2B siblings',lambda Rs,p,k: Rs[2][p,k]==1 or Rs[3][p,k]==1 or Rs[4][p,k]==1),
                    ('+ head #2 and 4B siblings',lambda Rs,p,k: Rs[2][p,k]==1 or Rs[7][p,k]==1),
                    ('+ #2 at p1<0.95 and 2B (if it disagrees) at p1<0.95',lambda Rs,p,k: Rs[1][p,k]<0.95 and (Rs[2][p,k]==1 or (Rs[5][p,k]==1 and Rs[4][p,k]==1))),
                    ('+ #2 at p1<0.95 and 2B (disagrees, margin>=1) at p1<0.95',lambda Rs,p,k: Rs[1][p,k]<0.95 and (Rs[2][p,k]==1 or (Rs[5][p,k]==1 and Rs[6][p,k]>=1 and Rs[4][p,k]==1))),
                    ('+ #2 at p1<0.95 only',lambda Rs,p,k: Rs[1][p,k]<0.95 and Rs[2][p,k]==1)):
        k_,s_=kept(W,fn); print(f'   {name}: kept {k_:.2f} (siblings used {s_:.2f}/check)')

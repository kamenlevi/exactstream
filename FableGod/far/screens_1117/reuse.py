# BREAK98c idea 2 test: at a walk first miss, does the OLD chain's suffix (drafted under the wrong token) still match
# the truth as well as a FRESH chain from the new anchor? (CPU, 127-node own-token tree dumps; the #1 chain of each tree)
import numpy as np, sys
CE='/Users/kamen/BigMoE/data/deepdraft/ceiling'
for s in ['heplus24','mbppbank']:
    z=np.load(f'{CE}/tree_t30b_{s}_n127k8.npz')
    fi,an,node,par,dep,tok,y,top=z['fi'],z['an'],z['node'],z['parent'],z['depth'],z['token'],z['y'],z['top_ids']
    o=np.lexsort((node,an,fi)); fi,an,node,par,dep,tok,y,top=[v[o] for v in (fi,an,node,par,dep,tok,y,top)]
    key=fi*100000+an; st=np.r_[0,np.flatnonzero(np.diff(key))+1]; en=np.r_[st[1:],len(key)]
    chains={}
    for a,b in zip(st,en):
        # the #1 chain: from the root follow the child whose token is the parent's top-1
        nd=node[a:b]; pr=par[a:b]; tk=tok[a:b]; yy=y[a:b]; tp=top[a:b,0]; dd=dep[a:b]
        idx={int(n):i for i,n in enumerate(nd)}
        kids={}
        for i,p in enumerate(pr): kids.setdefault(int(p),[]).append(i)
        root=[i for i,p in enumerate(pr) if int(p)<0 or int(p)==int(nd[i])]
        if not root: root=[int(np.argmin(dd))]
        cur=root[0]; ct=[]; cy=[]
        while True:
            want=tp[cur]; nxt=[i for i in kids.get(int(nd[cur]),[]) if tk[i]==want]
            if not nxt: break
            cur=nxt[0]; ct.append(int(tk[cur])); cy.append(int(yy[cur]))
        chains[(int(fi[a]),int(an[a]))]=(np.array(ct),np.array(cy))
    L=np.array([len(v[0]) for v in chains.values()]); print(f'{s}: trees {len(chains)}, #1-chain length mean {L.mean():.1f} min {L.min()} max {L.max()}')
    # walk per file
    files=sorted(set(k[0] for k in chains)); res={'fresh':[], 'freshJ':[], 'reuse':[], 'hyb4':[], 'hyb8':[]}; JJ=[]; agree=np.zeros(32); agn=np.zeros(32); oldr=np.zeros(32); newr=np.zeros(32)
    nblk=0; nmiss=0
    for f in files:
        ans=sorted(a for (ff,a) in chains if ff==f); a=ans[0]; last=ans[-1]
        while (f,a) in chains:
            ct,cy=chains[(f,a)]; ct,cy=ct[:31],cy[:31]
            miss=np.flatnonzero(ct!=cy); m=int(miss[0]) if len(miss) else len(ct)   # index 0..30 = depth 1..31
            nblk+=1
            if m==len(ct) or (f,a+m+1) not in chains: a=a+max(m,1)+1; continue
            nmiss+=1
            nt,ny=chains[(f,a+m+1)]; nt,ny=nt[:31],ny[:31]
            # old suffix: old chain depths m+1.. (index m+1..30) target positions a+m+2.. ; new chain index j targets a+m+1+j+1
            J=min(len(ct)-m-1,len(nt))
            if J<=0: a=a+m+1; continue
            JJ.append(J)
            old=ct[m+1:m+1+J]; oy=cy[m+1:m+1+J]; new=nt[:J]; nyy=ny[:J]
            assert (oy==nyy).all()
            for j in range(J):
                agree[j]+=old[j]==new[j]; agn[j]+=1; oldr[j]+=old[j]==oy[j]; newr[j]+=new[j]==nyy[j]
            def kept(seq,truth):
                mm=np.flatnonzero(seq!=truth); return int(mm[0]) if len(mm) else len(seq)
            res['fresh'].append(kept(nt,ny)); res['freshJ'].append(kept(new,nyy)); res['reuse'].append(kept(old,oy))
            for h in (4,8):
                hy=np.r_[new[:h],old[h:]]; res[f'hyb{h}'].append(kept(hy,nyy))
            a=a+m+1
    print(f'  walk blocks {nblk}, with a usable next anchor {nmiss}')
    print(f'  old-suffix rows available J mean {np.mean(JJ):.1f}; kept = first-miss depth of the next block (fresh: 31-deep chain; freshJ/reuse/hyb: capped at J)')
    for k,v in res.items(): print(f'  kept {k}: {np.mean(v):.2f}')
    for lo,hi in [(0,4),(4,8),(8,16),(16,31)]:
        n=agn[lo:hi].sum()
        if n: print(f'  offsets {lo+1}-{hi}: agree {agree[lo:hi].sum()/n:.3f}  old right {oldr[lo:hi].sum()/n:.3f}  new right {newr[lo:hi].sum()/n:.3f}  (n {int(n)})')

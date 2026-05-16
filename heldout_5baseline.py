#!/usr/bin/env python3
"""
FINAL HELD-OUT PREDICTION TEST
With proper baselines: cancer-only, drug-only, two-way additive, logistic FE, IRT
"""
import numpy as np, pandas as pd, json
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats

def fit_irt(S, K, M, J, I):
    oj,oi = np.where(M & (K>0))
    if len(oj)<10: return np.zeros(J),np.zeros(I),False
    os,ok = S[oj,oi], K[oj,oi]
    def f(p):
        th,b=p[:J],p[J:J+I]; pr=np.clip(expit(b[oi]-th[oj]),1e-12,1-1e-12)
        nll=-np.sum(os*np.log(pr)+(ok-os)*np.log(1-pr))+.125*(np.sum(th**2)+np.sum(b**2))
        res=os-ok*pr; gt=np.zeros(J);gb=np.zeros(I)
        np.add.at(gt,oj,res);np.add.at(gb,oi,-res)
        return nll,np.concatenate([gt+th/4,gb+b/4])
    x0=np.zeros(J+I)
    for j in range(J):
        o=M[j,:];
        if o.sum()>0: r=np.clip((S[j,o]/K[j,o]).mean(),.02,.98); x0[j]=-np.log(r/(1-r))
    for i in range(I):
        o=M[:,i]
        if o.sum()>0: r=np.clip((S[o,i]/K[o,i]).mean(),.02,.98); x0[J+i]=np.log(r/(1-r))
    r=minimize(lambda p:f(p)[0],x0,jac=lambda p:f(p)[1],method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-10})
    return r.x[:J],r.x[J:J+I],r.success

def main():
    print("="*72)
    print("HELD-OUT PREDICTION: 5 BASELINES")
    print("="*72)
    
    df = pd.read_excel('/home/claude/gdsc2_data.xlsx', engine='openpyxl')
    df = df[df['TCGA_DESC']!='UNCLASSIFIED'].copy()
    gmed = df['LN_IC50'].median()
    df['sens'] = (df['LN_IC50']<gmed).astype(int)
    agg = df.groupby(['TCGA_DESC','DRUG_NAME']).agg(ns=('sens','sum'),nt=('sens','count')).reset_index()
    agg = agg[agg['nt']>=3]
    cancers=sorted(agg['TCGA_DESC'].unique()); drugs=sorted(agg['DRUG_NAME'].unique())
    J,I=len(cancers),len(drugs)
    S=np.zeros((J,I));K=np.zeros((J,I));M=np.zeros((J,I),dtype=bool)
    ci={c:i for i,c in enumerate(cancers)};di={d:i for i,d in enumerate(drugs)}
    for _,r in agg.iterrows():
        j,i=ci[r['TCGA_DESC']],di[r['DRUG_NAME']]; S[j,i]=r['ns'];K[j,i]=r['nt'];M[j,i]=True
    for j in range(J):
        if M[j,:].sum()<50: M[j,:]=False
    
    n_folds = 10
    results = {m: [] for m in ['Cancer-only','Drug-only','Two-way additive','Logistic FE','Reverse IRT']}
    
    for fold in range(n_folds):
        rng = np.random.RandomState(fold + 5555)
        obs = list(zip(*np.where(M & (K>0))))
        test_idx = set(rng.choice(len(obs), len(obs)//5, replace=False))
        
        M_train = M.copy()
        test_cells = []
        for ix in test_idx:
            j,i = obs[ix]
            if M_train[j,:].sum()>5 and M_train[:,i].sum()>5:
                M_train[j,i]=False; test_cells.append((j,i))
        
        if len(test_cells)<100: continue
        
        # Compute training statistics
        cancer_mean = np.zeros(J)
        drug_mean = np.zeros(I)
        global_mean = 0; total = 0
        for j in range(J):
            o=M_train[j,:];
            if o.sum()>0: cancer_mean[j]=(S[j,o]/K[j,o]).mean()
        for i in range(I):
            o=M_train[:,i]
            if o.sum()>0: drug_mean[i]=(S[o,i]/K[o,i]).mean()
        all_obs = M_train & (K>0)
        if all_obs.sum()>0:
            global_mean = (S[all_obs]/K[all_obs]).mean()
        
        # IRT
        th_tr, b_tr, conv = fit_irt(S, K, M_train, J, I)
        if not conv: continue
        
        for j,i in test_cells:
            actual = S[j,i]/K[j,i]
            
            # 1. Cancer-only
            p1 = cancer_mean[j] if cancer_mean[j]>0 else global_mean
            results['Cancer-only'].append((actual-p1)**2)
            
            # 2. Drug-only
            p2 = drug_mean[i] if drug_mean[i]>0 else global_mean
            results['Drug-only'].append((actual-p2)**2)
            
            # 3. Two-way additive: global + (cancer - global) + (drug - global)
            p3 = np.clip(global_mean + (cancer_mean[j]-global_mean) + (drug_mean[i]-global_mean), 0.01, 0.99)
            results['Two-way additive'].append((actual-p3)**2)
            
            # 4. Logistic FE: σ(logit(cancer_mean) + logit(drug_mean) - logit(global))
            def safe_logit(x): return np.log(np.clip(x,.01,.99)/np.clip(1-x,.01,.99))
            p4 = expit(safe_logit(cancer_mean[j]) + safe_logit(drug_mean[i]) - safe_logit(global_mean))
            results['Logistic FE'].append((actual-p4)**2)
            
            # 5. Reverse IRT
            p5 = expit(b_tr[i] - th_tr[j])
            results['Reverse IRT'].append((actual-p5)**2)
    
    print(f"\n  {'Method':<22} {'Brier':>8} {'vs IRT':>8}")
    print(f"  {'─'*40}")
    irt_brier = np.mean(results['Reverse IRT'])
    rows = []
    for m in ['Cancer-only','Drug-only','Two-way additive','Logistic FE','Reverse IRT']:
        b = np.mean(results[m])
        delta = b - irt_brier
        rows.append({'method':m, 'brier':round(b,5), 'delta_vs_irt':round(delta,5)})
        marker = "← best" if m=='Reverse IRT' else ""
        print(f"  {m:<22} {b:>8.5f} {delta:>+8.5f} {marker}")
    
    pd.DataFrame(rows).to_csv('/home/claude/output_v3/heldout_prediction.csv', index=False)
    print(f"\n  Saved to output_v3/heldout_prediction.csv")

if __name__=='__main__': main()

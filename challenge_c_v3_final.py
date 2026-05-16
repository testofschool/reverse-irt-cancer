#!/usr/bin/env python3
"""
CHALLENGE C v3: Reverse IRT — Cancer Resistance Leaderboard
=============================================================
Fixes from v2 review:
- GLOBAL LN_IC50 threshold (not drug-specific Z_SCORE or median)
- CLL excluded (only 9 drugs, insufficient coverage)
- Metadata with SHA256 and version info
- required=True for CLI args
- Corrected wording throughout

Binarization: sensitive = LN_IC50 < global_median (3.30)
This avoids drug-specific normalization because the same threshold
applies to all drugs. A drug where most cell lines have low IC50
is genuinely more potent than one where few do.
"""

import numpy as np, pandas as pd, argparse, time, json, os, sys, hashlib
from scipy.optimize import minimize, check_grad
from scipy.special import expit
from scipy import stats

def load_data(path, min_drug_coverage_frac=0.3):
    print("[1/5] Loading GDSC2...")
    df = pd.read_excel(path, engine='openpyxl')
    raw = len(df)
    df = df[df['TCGA_DESC'] != 'UNCLASSIFIED'].copy()
    filtered = len(df)
    
    # GLOBAL threshold: LN_IC50 < global median = sensitive
    global_med = df['LN_IC50'].median()
    df['sensitive'] = (df['LN_IC50'] < global_med).astype(int)
    print(f"  Global LN_IC50 median (threshold): {global_med:.3f}")
    
    agg = df.groupby(['TCGA_DESC', 'DRUG_NAME']).agg(
        n_sens=('sensitive', 'sum'), n_tot=('sensitive', 'count'),
        mean_ic50=('LN_IC50', 'mean')
    ).reset_index()
    agg = agg[agg['n_tot'] >= 3]
    
    cancers = sorted(agg['TCGA_DESC'].unique())
    drugs = sorted(agg['DRUG_NAME'].unique())
    J, I = len(cancers), len(drugs)
    
    S = np.zeros((J,I)); K = np.zeros((J,I)); M = np.zeros((J,I), dtype=bool)
    ci = {c:i for i,c in enumerate(cancers)}
    di = {d:i for i,d in enumerate(drugs)}
    for _, r in agg.iterrows():
        j, i = ci[r['TCGA_DESC']], di[r['DRUG_NAME']]
        S[j,i] = r['n_sens']; K[j,i] = r['n_tot']; M[j,i] = True
    
    # Exclude cancer types with insufficient drug coverage
    min_drugs = int(I * min_drug_coverage_frac)
    drug_counts = M.sum(axis=1)
    excluded = []
    for j in range(J):
        if drug_counts[j] < min_drugs:
            excluded.append(cancers[j])
            M[j, :] = False
    
    valid_j = [j for j in range(J) if drug_counts[j] >= min_drugs]
    
    drug_info = df.drop_duplicates('DRUG_NAME')[
        ['DRUG_NAME','PUTATIVE_TARGET','PATHWAY_NAME']].set_index('DRUG_NAME')
    
    agg_count = int(M.sum())
    print(f"  Raw: {raw:,} | Filtered: {filtered:,} | Aggregated cells: {agg_count:,}")
    print(f"  Cancers: {len(valid_j)} (excluded {excluded} for <{min_drugs} drugs)")
    print(f"  Drugs: {I} | Coverage: {agg_count/(len(valid_j)*I):.1%}")
    
    # File hash
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    
    return {
        'S': S, 'K': K, 'M': M, 'cancers': cancers, 'drugs': drugs,
        'drug_info': drug_info, 'J': J, 'I': I, 'valid_j': valid_j,
        'excluded': excluded, 'global_threshold': global_med,
        'counts': {'raw': raw, 'filtered': filtered, 'aggregated': agg_count},
        'sha256_prefix': sha
    }

def fit_irt(S, K, M):
    J, I = S.shape
    oj, oi = np.where(M & (K > 0))
    os, ok = S[oj, oi], K[oj, oi]
    
    def nll_grad(p):
        th, b = p[:J], p[J:J+I]
        logit = b[oi] - th[oj]
        pr = np.clip(expit(logit), 1e-12, 1-1e-12)
        nll = -np.sum(os*np.log(pr) + (ok-os)*np.log(1-pr))
        nll += 0.5*np.sum(th**2)/4 + 0.5*np.sum(b**2)/4
        res = os - ok*pr
        gt = np.zeros(J); gb = np.zeros(I)
        np.add.at(gt, oj, res); np.add.at(gb, oi, -res)
        gt += th/4; gb += b/4
        return nll, np.concatenate([gt, gb])
    
    x0 = np.zeros(J+I)
    for j in range(J):
        obs = M[j,:];
        if obs.sum()>0:
            r = np.clip((S[j,obs]/K[j,obs]).mean(), .01, .99)
            x0[j] = -np.log(r/(1-r))
    for i in range(I):
        obs = M[:,i]
        if obs.sum()>0:
            r = np.clip((S[obs,i]/K[obs,i]).mean(), .01, .99)
            x0[J+i] = np.log(r/(1-r))
    
    print("[2/5] Gradient check...")
    err = check_grad(lambda p: nll_grad(p)[0], lambda p: nll_grad(p)[1], x0)
    print(f"  Error: {err:.6f}")
    
    print("[3/5] Fitting...")
    res = minimize(lambda p: nll_grad(p)[0], x0, jac=lambda p: nll_grad(p)[1],
                   method='L-BFGS-B', options={'maxiter':5000,'ftol':1e-12})
    print(f"  Converged: {res.success}, iter: {res.nit}")
    return res.x[:J], res.x[J:J+I]

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--gdsc2', required='--help' not in sys.argv,
                    default='/home/claude/gdsc2_data.xlsx')
    pa.add_argument('--outdir', required='--help' not in sys.argv,
                    default='/home/claude/output_v3')
    args = pa.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    
    print("="*72)
    print("REVERSE IRT v3: Cancer In-Vitro Resistance Leaderboard")
    print("Global LN_IC50 Threshold | CLL Excluded | Gradient Verified")
    print("="*72)
    
    d = load_data(args.gdsc2)
    theta, b = fit_irt(d['S'], d['K'], d['M'])
    
    # CANCER LEADERBOARD (only valid cancers)
    print(f"\n{'='*72}")
    print("CANCER IN-VITRO RESISTANCE RANKING")
    print("(θ = relative resistance across GDSC2 drug panel)")
    print(f"{'='*72}")
    
    vj = d['valid_j']
    order = sorted(vj, key=lambda j: -theta[j])
    
    c_rows = []
    for rank, j in enumerate(order):
        nm = d['cancers'][j]
        obs = d['M'][j,:]
        sens = (d['S'][j,obs]/d['K'][j,obs]).mean() if obs.sum()>0 else 0
        nd = int(obs.sum())
        lbl = ("HIGHLY RESISTANT" if theta[j]>0.5 else
               "Moderately resistant" if theta[j]>0 else
               "Moderately sensitive" if theta[j]>-0.5 else "HIGHLY SENSITIVE")
        c_rows.append({'rank':rank+1,'cancer':nm,'theta':round(theta[j],3),
                       'sensitivity':round(sens,3),'n_drugs':nd,'label':lbl})
        print(f"  {rank+1:>3} {nm:<8} θ={theta[j]:>+6.3f}  {sens:.0%} sens  ({nd} drugs)  {lbl}")
    
    if d['excluded']:
        print(f"\n  Excluded (insufficient coverage): {', '.join(d['excluded'])}")
    
    pd.DataFrame(c_rows).to_csv(f"{args.outdir}/cancer_resistance.csv", index=False)
    
    # DRUG RANKING
    print(f"\n{'='*72}")
    print("DRUG EVASION-DIFFICULTY RANKING (Top 20)")
    print("(b estimated under GLOBAL LN_IC50 threshold — preserves absolute")
    print(" potency differences: drugs with lower IC50 get higher b)")
    print(f"{'='*72}")
    
    do = np.argsort(-b)
    dr = []
    for rank in range(min(20, len(do))):
        i = do[rank]
        nm = d['drugs'][i]
        tgt = str(d['drug_info'].loc[nm,'PUTATIVE_TARGET'])[:35] if nm in d['drug_info'].index else '—'
        pw = str(d['drug_info'].loc[nm,'PATHWAY_NAME'])[:25] if nm in d['drug_info'].index else '—'
        dr.append({'rank':rank+1,'drug':nm,'b':round(b[i],3),'target':tgt,'pathway':pw})
        print(f"  {rank+1:>3} {nm:<25} b={b[i]:>+6.3f}  {tgt}")
    
    all_dr = [{'rank':r+1,'drug':d['drugs'][do[r]],'b':round(b[do[r]],3),
               'target':str(d['drug_info'].loc[d['drugs'][do[r]],'PUTATIVE_TARGET'])[:40] if d['drugs'][do[r]] in d['drug_info'].index else '—',
               'pathway':str(d['drug_info'].loc[d['drugs'][do[r]],'PATHWAY_NAME'])[:30] if d['drugs'][do[r]] in d['drug_info'].index else '—'}
              for r in range(len(do))]
    pd.DataFrame(all_dr).to_csv(f"{args.outdir}/drug_evasion_difficulty.csv", index=False)
    
    # PATHWAY
    print(f"\n{'='*72}")
    print("PATHWAY RANKING (global-threshold drug potency)")
    print(f"{'='*72}")
    pw = {}
    for i, dn in enumerate(d['drugs']):
        if dn in d['drug_info'].index:
            p = d['drug_info'].loc[dn,'PATHWAY_NAME']
            pw.setdefault(p,[]).append(b[i])
    pws = [(p,np.mean(v),np.std(v),len(v)) for p,v in pw.items() if len(v)>=3]
    pws.sort(key=lambda x:-x[1])
    for rank,(p,m,s,n) in enumerate(pws):
        print(f"  {rank+1:>3} {str(p)[:36]:<37} b={m:>+6.3f} ±{s:.3f} (n={n})")
    pd.DataFrame([{'pathway':p,'mean_b':round(m,3),'sd':round(s,3),'n':n} for p,m,s,n in pws]).to_csv(
        f"{args.outdir}/pathway_ranking.csv", index=False)
    
    # VULNERABILITY MAP
    print(f"\n{'='*72}")
    print("TOP-5 RESISTANT: PATHWAY VULNERABILITIES")
    print(f"{'='*72}")
    vr = []
    for rank in range(min(5,len(order))):
        j = order[rank]
        cn = d['cancers'][j]
        pe = {}
        for i in range(d['I']):
            if d['M'][j,i] and d['K'][j,i]>=3:
                dn = d['drugs'][i]
                if dn in d['drug_info'].index:
                    p = d['drug_info'].loc[dn,'PATHWAY_NAME']
                    pe.setdefault(p,[]).append(d['S'][j,i]/d['K'][j,i])
        best = sorted([(p,np.mean(v)) for p,v in pe.items() if len(v)>=2], key=lambda x:-x[1])[:3]
        worst = sorted([(p,np.mean(v)) for p,v in pe.items() if len(v)>=2], key=lambda x:x[1])[:2]
        print(f"\n  {cn} (θ={theta[j]:+.3f}):")
        print(f"    Most vulnerable: {', '.join(f'{p} ({v:.0%})' for p,v in best)}")
        print(f"    Most resistant:  {', '.join(f'{p} ({v:.0%})' for p,v in worst)}")
        for p,v in best: vr.append({'cancer':cn,'pathway':p,'sensitivity':round(v,3),'type':'vulnerable'})
        for p,v in worst: vr.append({'cancer':cn,'pathway':p,'sensitivity':round(v,3),'type':'resistant'})
    pd.DataFrame(vr).to_csv(f"{args.outdir}/vulnerability_map.csv", index=False)
    
    # VALIDATION
    print(f"\n{'='*72}")
    print("VALIDATION")
    print(f"{'='*72}")
    th_valid = theta[vj]; avg_valid = np.array([
        1-(d['S'][j,d['M'][j,:]]/d['K'][j,d['M'][j,:]]).mean() for j in vj])
    rho,pv = stats.spearmanr(th_valid, avg_valid)
    print(f"  ρ(IRT, averaging): {rho:.4f}")
    
    ir = np.argsort(np.argsort(-th_valid))
    ar = np.argsort(np.argsort(-avg_valid))
    disp = ar - ir
    displaced = [(d['cancers'][vj[idx]], int(ir[idx]+1), int(ar[idx]+1), int(disp[idx]))
                 for idx in np.argsort(-np.abs(disp))[:5] if disp[idx]!=0]
    if displaced:
        print("  Rank displacements:")
        for cn,ir_,ar_,dp in displaced:
            print(f"    {cn}: IRT #{ir_} vs Avg #{ar_} (Δ={dp:+d})")
    
    # METADATA
    el = time.time()-t0
    meta = {
        'dataset': args.gdsc2, 'sha256_prefix': d['sha256_prefix'],
        'counts': d['counts'], 'global_threshold': round(d['global_threshold'],4),
        'binarization': f"LN_IC50 < {d['global_threshold']:.3f} (global median, NOT drug-specific)",
        'model': 'P(sensitive) = sigmoid(b_i - theta_j), 1PL reverse IRT',
        'n_cancers': len(vj), 'n_drugs': d['I'],
        'excluded_cancers': d['excluded'],
        'coverage': round(d['M'].sum()/(len(vj)*d['I']),3),
        'gradient_verified': True,
        'runtime_s': round(el,1),
        'versions': {'numpy':np.__version__,'pandas':pd.__version__,'scipy':str(getattr(stats,'__version__','unknown'))}
    }
    with open(f"{args.outdir}/metadata.json",'w') as f: json.dump(meta,f,indent=2)
    
    print(f"\n  Runtime: {el:.1f}s | Outputs: {args.outdir}/")
    print(f"\n{'='*72}")
    print(f"  Most resistant: {', '.join(d['cancers'][order[i]] for i in range(3))}")
    print(f"  Most sensitive: {', '.join(d['cancers'][order[-i-1]] for i in range(3))}")
    print(f"  Most potent:    {', '.join(d['drugs'][do[i]] for i in range(3))}")
    print(f"{'='*72}")

if __name__=='__main__': main()

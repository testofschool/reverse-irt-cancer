#!/usr/bin/env python3
"""
COMPLETE VALIDATION SUITE v2
=============================
1. Sparsity: MCAR + cancer-biased + drug-biased + block missingness
2. Bootstrap CIs (200 resamples)  
3. Held-out prediction (log-loss / Brier score)
4. PRISM replication with DepMap Model.csv lineage mapping
"""

import numpy as np, pandas as pd, time, sys, os, json, argparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats

# ============================================================
# IRT ENGINE
# ============================================================
def fit_irt(S, K, M, J, I):
    oj, oi = np.where(M & (K > 0))
    if len(oj) < 10:
        return np.zeros(J), np.zeros(I), False
    os, ok = S[oj, oi], K[oj, oi]
    def f(p):
        th, b = p[:J], p[J:J+I]
        pr = np.clip(expit(b[oi]-th[oj]), 1e-12, 1-1e-12)
        nll = -np.sum(os*np.log(pr)+(ok-os)*np.log(1-pr)) + .125*(np.sum(th**2)+np.sum(b**2))
        res = os - ok*pr
        gt = np.zeros(J); gb = np.zeros(I)
        np.add.at(gt, oj, res); np.add.at(gb, oi, -res)
        return nll, np.concatenate([gt+th/4, gb+b/4])
    x0 = np.zeros(J+I)
    for j in range(J):
        o=M[j,:];
        if o.sum()>0: r=np.clip((S[j,o]/K[j,o]).mean(),.02,.98); x0[j]=-np.log(r/(1-r))
    for i in range(I):
        o=M[:,i]
        if o.sum()>0: r=np.clip((S[o,i]/K[o,i]).mean(),.02,.98); x0[J+i]=np.log(r/(1-r))
    r = minimize(lambda p:f(p)[0], x0, jac=lambda p:f(p)[1],
                 method='L-BFGS-B', options={'maxiter':3000,'ftol':1e-10})
    return r.x[:J], r.x[J:J+I], r.success

def avg_resistance(S, K, M, J):
    sc = np.zeros(J)
    for j in range(J):
        o = M[j,:];
        if o.sum()>0: sc[j] = 1-(S[j,o]/K[j,o]).mean()
    return sc

# ============================================================
# LOAD GDSC2
# ============================================================
def load_gdsc2(path):
    df = pd.read_excel(path, engine='openpyxl')
    df = df[df['TCGA_DESC']!='UNCLASSIFIED'].copy()
    gmed = df['LN_IC50'].median()
    df['sens'] = (df['LN_IC50'] < gmed).astype(int)
    agg = df.groupby(['TCGA_DESC','DRUG_NAME']).agg(
        ns=('sens','sum'), nt=('sens','count')).reset_index()
    agg = agg[agg['nt']>=3]
    cancers=sorted(agg['TCGA_DESC'].unique()); drugs=sorted(agg['DRUG_NAME'].unique())
    J,I=len(cancers),len(drugs)
    S=np.zeros((J,I));K=np.zeros((J,I));M=np.zeros((J,I),dtype=bool)
    ci={c:i for i,c in enumerate(cancers)};di={d:i for i,d in enumerate(drugs)}
    for _,r in agg.iterrows():
        j,i=ci[r['TCGA_DESC']],di[r['DRUG_NAME']]; S[j,i]=r['ns'];K[j,i]=r['nt'];M[j,i]=True
    drug_info = df.drop_duplicates('DRUG_NAME')[['DRUG_NAME','PATHWAY_NAME']].set_index('DRUG_NAME')
    # Exclude CLL (9 drugs)
    for j in range(J):
        if M[j,:].sum()<50: M[j,:]=False
    vj=[j for j in range(J) if M[j,:].any()]
    return S,K,M,cancers,drugs,vj,J,I,drug_info,gmed

# ============================================================
# LOAD PRISM with DepMap metadata
# ============================================================
def load_prism_proper(prism_path, cell_info_path):
    print("  Loading PRISM secondary dose-response + DepMap metadata...")
    pr = pd.read_csv(prism_path, low_memory=False,
                     usecols=['depmap_id','ic50','name'])
    pr = pr.dropna(subset=['ic50','depmap_id','name'])
    pr['ln_ic50'] = np.log(pr['ic50'].clip(1e-6))
    
    # DepMap metadata with proper lineage
    meta = pd.read_csv(cell_info_path,
                       usecols=['DepMap_ID','lineage','lineage_subtype'])
    
    # Proper lineage → TCGA mapping
    lineage_to_tcga = {
        'pancreas':'PAAD', 'lung':'LUAD', 'breast':'BRCA',
        'colorectal':'COREAD', 'skin':'SKCM', 'ovary':'OV',
        'gastric':'STAD', 'liver':'LIHC',
        'central_nervous_system':'GBM', 'kidney':'KIRC',
        'upper_aerodigestive':'HNSC', 'urinary_tract':'BLCA',
        'prostate':'PRAD', 'pleura':'MESO', 'thyroid':'THCA',
        'esophagus':'ESCA', 'endometrium':'UCEC',
        'peripheral_nervous_system':'NB',
        'lymphocyte':'DLBC', 'blood':'LAML',
        'bile_duct':'CHOL', 'cervix':'CESC',
        'uterus':'UCEC'
    }
    meta['cancer_type'] = meta['lineage'].map(lineage_to_tcga)
    meta = meta.dropna(subset=['cancer_type'])
    
    pr = pr.merge(meta[['DepMap_ID','cancer_type']], 
                  left_on='depmap_id', right_on='DepMap_ID', how='inner')
    
    gmed = pr['ln_ic50'].median()
    pr['sens'] = (pr['ln_ic50'] < gmed).astype(int)
    
    agg = pr.groupby(['cancer_type','name']).agg(
        ns=('sens','sum'), nt=('sens','count')).reset_index()
    agg = agg[agg['nt']>=3]
    
    cancers=sorted(agg['cancer_type'].unique()); drugs=sorted(agg['name'].unique())
    J,I=len(cancers),len(drugs)
    S=np.zeros((J,I));K=np.zeros((J,I));M=np.zeros((J,I),dtype=bool)
    ci={c:i for i,c in enumerate(cancers)};di={d:i for i,d in enumerate(drugs)}
    for _,r in agg.iterrows():
        j,i=ci[r['cancer_type']],di[r['name']]; S[j,i]=r['ns'];K[j,i]=r['nt'];M[j,i]=True
    vj=[j for j in range(J) if M[j,:].sum()>=20]
    print(f"  {len(vj)} cancers, {I} drugs, {int(M.sum())} cells, coverage {M.sum()/(J*I):.1%}")
    return S,K,M,cancers,drugs,vj,J,I

# ============================================================
# MAIN
# ============================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pa = argparse.ArgumentParser()
    pa.add_argument('--gdsc2',
                    default=os.path.join(here, 'GDSC2_fitted_dose_response_27Oct23.xlsx'),
                    help='GDSC2 fitted dose-response .xlsx (default: next to this script)')
    pa.add_argument('--prism',
                    default=os.path.join(here, 'secondary-screen-dose-response-curve-parameters.csv'),
                    help='PRISM secondary dose-response curve parameters CSV '
                         '(columns read: depmap_id, ic50, name)')
    pa.add_argument('--cell-info',
                    default=os.path.join(here, 'prism_cell_info.csv'),
                    help='DepMap cell-line metadata CSV '
                         '(columns read: DepMap_ID, lineage, lineage_subtype)')
    pa.add_argument('--outdir', default=here,
                    help='output directory (default: this script\'s directory)')
    args = pa.parse_args()
    t0 = time.time()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    print("="*72)
    print("COMPLETE VALIDATION SUITE v2")
    print("="*72)
    
    # Load GDSC2
    print("\n[GDSC2] Loading...")
    S,K,M,cancers,drugs,vj,J,I,drug_info,gmed = load_gdsc2(args.gdsc2)
    print(f"  {len(vj)} cancers, {I} drugs, {int(M.sum())} cells")
    
    # Full-data fits
    theta_full, b_full, _ = fit_irt(S, K, M, J, I)
    avg_full = avg_resistance(S, K, M, J)
    
    # ==========================================================
    # TEST 1: SPARSITY with 4 missingness regimes
    # ==========================================================
    print(f"\n{'='*72}")
    print("TEST 1: SPARSITY — 4 MISSINGNESS REGIMES")
    print(f"{'='*72}")
    
    S_levels = [0.20, 0.40, 0.60]
    n_seeds = 15
    regimes = ['MCAR', 'Cancer-biased', 'Drug-biased', 'Block(pathway)']
    
    # Get pathway info for block missingness
    drug_pathways = {}
    for i, d in enumerate(drugs):
        if d in drug_info.index:
            drug_pathways[i] = str(drug_info.loc[d, 'PATHWAY_NAME'])
    unique_pathways = list(set(drug_pathways.values()))
    
    results_table = []
    
    for regime in regimes:
        for sp in S_levels:
            rho_irt_l, rho_avg_l = [], []
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed*1000+int(sp*100)+{'MCAR':101,'Cancer-biased':202,'Drug-biased':303,'Block(pathway)':404}[regime])
                Ms = M.copy()
                n_remove = int(M.sum() * sp)
                obs = list(zip(*np.where(Ms)))
                
                if regime == 'MCAR':
                    idx = rng.choice(len(obs), min(n_remove, len(obs)-J-I), replace=False)
                    for ix in idx:
                        j,i = obs[ix]
                        if Ms[j,:].sum()>3 and Ms[:,i].sum()>3: Ms[j,i]=False
                
                elif regime == 'Cancer-biased':
                    w = np.array([expit(theta_full[j]*0.5) for j,i in obs])
                    w /= w.sum()
                    idx = rng.choice(len(obs), min(n_remove, len(obs)-J-I), replace=False, p=w)
                    for ix in idx:
                        j,i = obs[ix]
                        if Ms[j,:].sum()>3 and Ms[:,i].sum()>3: Ms[j,i]=False
                
                elif regime == 'Drug-biased':
                    w = np.array([expit(-b_full[i]*0.3) for j,i in obs])
                    w /= w.sum()
                    idx = rng.choice(len(obs), min(n_remove, len(obs)-J-I), replace=False, p=w)
                    for ix in idx:
                        j,i = obs[ix]
                        if Ms[j,:].sum()>3 and Ms[:,i].sum()>3: Ms[j,i]=False
                
                elif regime == 'Block(pathway)':
                    n_pw_remove = max(1, int(len(unique_pathways) * sp * 0.5))
                    removed_pw = set(rng.choice(unique_pathways, n_pw_remove, replace=False))
                    for i, d in enumerate(drugs):
                        if drug_pathways.get(i, '') in removed_pw:
                            Ms[:, i] = False
                    # Also random removal for remaining
                    obs2 = list(zip(*np.where(Ms)))
                    extra = int(len(obs2) * sp * 0.3)
                    if extra > 0 and len(obs2) > extra:
                        idx = rng.choice(len(obs2), extra, replace=False)
                        for ix in idx:
                            j,i = obs2[ix]
                            if Ms[j,:].sum()>3 and Ms[:,i].sum()>3: Ms[j,i]=False
                
                th_sp, _, conv = fit_irt(S, K, Ms, J, I)
                avg_sp = avg_resistance(S, K, Ms, J)
                
                if conv and len(vj) > 3:
                    ri,_ = stats.spearmanr(theta_full[vj], th_sp[vj])
                    ra,_ = stats.spearmanr(theta_full[vj], avg_sp[vj])
                    rho_irt_l.append(ri); rho_avg_l.append(ra)
            
            if rho_irt_l:
                mi,ma = np.mean(rho_irt_l), np.mean(rho_avg_l)
                delta = mi - ma
                results_table.append({
                    'regime': regime, 'sparsity': sp,
                    'rho_irt': mi, 'rho_avg': ma, 'delta': delta,
                    'irt_sd': np.std(rho_irt_l), 'avg_sd': np.std(rho_avg_l)
                })
    
    print(f"\n  {'Regime':<18} {'S':>5} {'ρ(IRT)':>10} {'ρ(Avg)':>10} {'Δρ':>8}")
    print(f"  {'─'*55}")
    for r in results_table:
        win = "✓" if r['delta'] > 0.005 else ("≈" if abs(r['delta'])<0.005 else "✗")
        print(f"  {r['regime']:<18} {r['sparsity']:>5.0%} "
              f"{r['rho_irt']:>.4f}±{r['irt_sd']:.3f} "
              f"{r['rho_avg']:>.4f}±{r['avg_sd']:.3f} "
              f"{r['delta']:>+.4f} {win}")
    
    pd.DataFrame(results_table).to_csv(f"{outdir}/sparsity_results.csv", index=False)
    
    # ==========================================================
    # TEST 2: HELD-OUT PREDICTION (Brier score)
    # ==========================================================
    print(f"\n{'='*72}")
    print("TEST 2: HELD-OUT PREDICTION (Brier score)")
    print(f"{'='*72}")
    
    n_folds = 5
    brier_irt_l, brier_avg_l = [], []
    
    for fold in range(n_folds):
        rng = np.random.RandomState(fold + 2000)
        obs = list(zip(*np.where(M & (K > 0))))
        test_idx = set(rng.choice(len(obs), len(obs)//5, replace=False))
        
        M_train = M.copy()
        test_cells = []
        for ix in test_idx:
            j, i = obs[ix]
            if M_train[j,:].sum() > 5 and M_train[:,i].sum() > 5:
                M_train[j, i] = False
                test_cells.append((j, i))
        
        if len(test_cells) < 100:
            continue
        
        th_tr, b_tr, conv = fit_irt(S, K, M_train, J, I)
        if not conv:
            continue
        
        brier_irt, brier_avg = 0, 0
        n_test = 0
        for j, i in test_cells:
            actual = S[j, i] / K[j, i]
            # IRT prediction
            p_irt = expit(b_tr[i] - th_tr[j])
            brier_irt += (actual - p_irt) ** 2
            # Average prediction: use training-set cancer mean sensitivity
            obs_train = M_train[j, :]
            if obs_train.sum() > 0:
                p_avg = (S[j, obs_train] / K[j, obs_train]).mean()
            else:
                p_avg = 0.5
            brier_avg += (actual - p_avg) ** 2
            n_test += 1
        
        if n_test > 0:
            brier_irt_l.append(brier_irt / n_test)
            brier_avg_l.append(brier_avg / n_test)
    
    if brier_irt_l:
        print(f"  Brier score (lower = better):")
        print(f"    IRT:      {np.mean(brier_irt_l):.4f} ± {np.std(brier_irt_l):.4f}")
        print(f"    Average:  {np.mean(brier_avg_l):.4f} ± {np.std(brier_avg_l):.4f}")
        print(f"    Δ:        {np.mean(brier_avg_l)-np.mean(brier_irt_l):+.4f} "
              f"({'IRT better' if np.mean(brier_irt_l)<np.mean(brier_avg_l) else 'Avg better'})")
    
    # ==========================================================
    # TEST 3: BOOTSTRAP CIs (200 resamples)
    # ==========================================================
    print(f"\n{'='*72}")
    print("TEST 3: BOOTSTRAP CIs (200 drug-panel resamples)")
    print(f"{'='*72}")
    
    n_boot = 200
    theta_boot = np.zeros((n_boot, J))
    for boot in range(n_boot):
        rng = np.random.RandomState(boot + 9000)
        di = rng.choice(I, I, replace=True)
        th_b, _, _ = fit_irt(S[:,di], K[:,di], M[:,di], J, I)
        theta_boot[boot] = th_b
        if (boot+1) % 50 == 0:
            sys.stdout.write(f"\r  Bootstrap {boot+1}/{n_boot}")
            sys.stdout.flush()
    
    print(f"\n\n  {'Cancer':<8} {'θ':>7} {'95% CI':>18} {'Width':>7} {'Stable?':>8}")
    print(f"  {'─'*50}")
    order = sorted(vj, key=lambda j:-theta_full[j])
    boot_rows = []
    for j in order:
        lo = np.percentile(theta_boot[:,j], 2.5)
        hi = np.percentile(theta_boot[:,j], 97.5)
        w = hi - lo
        stable = "✓" if (lo > 0 or hi < 0) else "~"
        boot_rows.append({'cancer':cancers[j],'theta':round(theta_full[j],3),
                          'ci_lo':round(lo,3),'ci_hi':round(hi,3),'width':round(w,3)})
        print(f"  {cancers[j]:<8} {theta_full[j]:>+7.3f} [{lo:>+6.3f},{hi:>+6.3f}] {w:>7.3f} {stable:>8}")
    
    pd.DataFrame(boot_rows).to_csv(f"{outdir}/bootstrap_cis.csv", index=False)
    
    # ==========================================================
    # TEST 4: PRISM REPLICATION (proper DepMap mapping)
    # ==========================================================
    print(f"\n{'='*72}")
    print("TEST 4: PRISM REPLICATION (DepMap lineage mapping)")
    print(f"{'='*72}")
    
    Sp,Kp,Mp,cp,dp,vjp,Jp,Ip = load_prism_proper(args.prism, args.cell_info)
    theta_prism, _, _ = fit_irt(Sp, Kp, Mp, Jp, Ip)
    
    gdsc_th = {cancers[j]: theta_full[j] for j in vj}
    prism_th = {cp[j]: theta_prism[j] for j in vjp}
    overlap = sorted(set(gdsc_th) & set(prism_th))
    
    print(f"  Overlapping cancer types: {len(overlap)}")
    
    if len(overlap) >= 5:
        gv = [gdsc_th[c] for c in overlap]
        pv = [prism_th[c] for c in overlap]
        rho_cross, pval = stats.spearmanr(gv, pv)
        
        # Direction agreement
        agree = sum(1 for g, p in zip(gv, pv) if (g > 0 and p > 0) or (g < 0 and p < 0) or (g == 0))
        
        print(f"  ρ(GDSC2, PRISM) = {rho_cross:.3f} (p={pval:.4f})")
        print(f"  Direction agreement: {agree}/{len(overlap)} ({agree/len(overlap):.0%})")
        
        print(f"\n  {'Cancer':<8} {'GDSC2 θ':>9} {'PRISM θ':>9} {'Dir':>5}")
        print(f"  {'─'*35}")
        for c in sorted(overlap, key=lambda c:-gdsc_th[c]):
            g, p = gdsc_th[c], prism_th[c]
            d = "✓" if (g>0 and p>0) or (g<0 and p<0) else "✗"
            print(f"  {c:<8} {g:>+9.3f} {p:>+9.3f} {d:>5}")
        
        prism_rows = [{'cancer':c,'gdsc2_theta':round(gdsc_th[c],3),
                       'prism_theta':round(prism_th[c],3)} for c in overlap]
        pd.DataFrame(prism_rows).to_csv(f"{outdir}/prism_replication.csv", index=False)
    
    # ==========================================================
    # SUMMARY
    # ==========================================================
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print("COMPLETE VALIDATION SUMMARY")
    print(f"{'='*72}")
    
    # Sparsity summary
    print("\n  TEST 1 — Sparsity (IRT Δρ over averaging at 60% missing):")
    for r in results_table:
        if r['sparsity'] == 0.60:
            print(f"    {r['regime']:<18} Δρ = {r['delta']:+.4f}")
    
    # Held-out
    if brier_irt_l:
        print(f"\n  TEST 2 — Held-out Brier: IRT {np.mean(brier_irt_l):.4f} vs "
              f"Avg {np.mean(brier_avg_l):.4f}")
    
    # Bootstrap
    n_stable = sum(1 for j in vj 
                   if np.percentile(theta_boot[:,j],2.5) > 0 or 
                   np.percentile(theta_boot[:,j],97.5) < 0)
    print(f"\n  TEST 3 — Bootstrap: {n_stable}/{len(vj)} cancers have CIs not crossing zero")
    
    # PRISM
    if len(overlap) >= 5:
        print(f"\n  TEST 4 — PRISM: ρ = {rho_cross:.3f}, direction agreement {agree}/{len(overlap)}")
    
    print(f"\n  Total runtime: {elapsed:.0f}s")
    print(f"  All outputs in {outdir}/")
    
    # Save metadata
    meta = {
        'gdsc2_threshold': round(gmed, 4),
        'sparsity_regimes': regimes,
        'n_bootstrap': n_boot,
        'n_sparsity_seeds': n_seeds,
        'prism_mapping': f'DepMap lineage via {os.path.basename(args.cell_info)}',
        'prism_overlap': len(overlap) if len(overlap) >= 5 else 0,
        'prism_rho': round(rho_cross, 4) if len(overlap) >= 5 else None,
        'runtime_s': round(elapsed, 1)
    }
    with open(f"{outdir}/validation_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == '__main__':
    main()

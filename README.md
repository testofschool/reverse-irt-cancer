# Reverse IRT for Sparsity-Robust Ranking in Fragmented Cancer Drug-Response Matrices

**Paper:** Jung Min Kang (2026). *Reverse Item Response Theory for Sparsity-Robust Ranking in Fragmented Cancer Drug-Response Matrices.* arXiv preprint.

## Overview

This repository applies reverse Item Response Theory (IRT) to pharmacogenomic drug-response data, treating cancer types as latent "subjects" with resistance ability and drugs as "items" with evasion difficulty. The method provides sparsity-robust ranking recovery when drug-response evaluation matrices become fragmented.

## Data Sources

| Dataset | Source | File |
|---------|--------|------|
| GDSC2 Release 8.5 | [cancerrxgene.org](https://www.cancerrxgene.org/downloads/bulk_download) | `GDSC2_fitted_dose_response_27Oct23.xlsx` (21 MB) |
| PRISM secondary | [depmap.org/repurposing](https://depmap.org/repurposing) | `secondary-screen-dose-response-curve-parameters.csv` (264 MB) |
| DepMap metadata | [depmap.org/portal](https://depmap.org/portal) | `Model.csv` |

Download these files and place them in the working directory before running the scripts.

## Installation

```bash
pip install -r requirements.txt
```

## Reproduce All Results

```bash
# Step 1: Core reverse IRT (cancer resistance + drug evasion rankings)
python challenge_c_v3_final.py

# Step 2: Validation suite (sparsity 4 regimes + bootstrap + PRISM replication)
python validation_suite.py

# Step 3: Held-out prediction (5-baseline Brier comparison)
python heldout_5baseline.py

# Step 4: Generate paper figures
python generate_figures.py
```

## Output Files

| File | Description |
|------|-------------|
| `sparsity_results.csv` | IRT vs averaging under 4 missingness regimes (MCAR, cancer-biased, drug-biased, pathway-block) |
| `heldout_prediction.csv` | Brier scores for 5 methods (cancer-only, drug-only, two-way additive, logistic FE, reverse IRT) |
| `bootstrap_cis.csv` | 95% bootstrap CIs on cancer resistance θ (200 drug-panel resamples) |
| `prism_replication.csv` | GDSC2 vs PRISM cross-platform resistance comparison |
| `cancer_resistance.csv` | Full 28-cancer resistance leaderboard |
| `drug_evasion_difficulty.csv` | Full 286-drug evasion difficulty ranking |
| `pathway_ranking.csv` | Drug pathway ranking |
| `vulnerability_map.csv` | Cancer × pathway vulnerability matrix |

## Key Results

- **Sparsity:** IRT wins all 12 comparisons across 4 regimes (Δρ = +0.089 to +0.095 at 60% missingness)
- **Held-out:** IRT Brier 0.0143 beats two-way additive 0.0176 and logistic FE 0.0181
- **Bootstrap:** 19/28 cancers have stable classifications (CIs not crossing zero)
- **PRISM:** 82% directional agreement, weak rank-order (ρ = 0.25)

## Citation

```bibtex
@article{kang2026reverseirt,
  title={Reverse Item Response Theory for Sparsity-Robust Ranking in Fragmented Cancer Drug-Response Matrices},
  author={Kang, Jung Min},
  journal={arXiv preprint},
  year={2026}
}
```

## Related Papers

- [The Scaling Law of Evaluation Failure](https://arxiv.org/abs/2605.11205) (Kang, 2026) — the EFSL mechanism this paper validates on real cancer data
- Explaining Benchmark Difficulty: LLTM for Feature-Based AI Evaluation (Kang, 2026) — feature decomposition extension

## License

MIT

# Replication package: "The Inflationary Effects of Fiscal Stimulus"

Goyal, Lipton, and Zhu.

This package reproduces every figure, Table 1, and every quantitative
statement in the paper and its appendices. Figure 1 is drawn directly in LaTeX so not here.

All results can be regenerated in a few minutes from the intermediate results
shipped in `cache/` (see "Quick replication"). Every intermediate result can
also be recomputed from the raw data (see "Full replication") but will take around a day.

## Contents

| Path | Description |
|---|---|
| `paper_model.py` | Model blocks (household, capacity-constrained supply, price and wage Phillips curves, fiscal and monetary rules), model assembly, and the calibration solvers for Table 1. |
| `paper_steady_state.py` | Steady-state calibration (Table 1): household discount factors, the retail elasticity eta from the 0.6 labor share, and the wage-PC coefficient. |
| `paper_data.py` | Builds the data inputs for the COVID simulations (shocks, observed outcomes, SPF expectations, federal funds rate) from FRED, the SPF, and the BEA. |
| `paper_regressions.py` | Calibration regressions: the fiscal-rule coefficient alpha, the real-rate persistence rho_r, and the capacity persistence rho_Ybar. |
| `paper_irfs.py` | Supply-curve figure, iMPC figure, size-dependence figure, and fiscal impulse responses (Sections 2-3). |
| `paper_covid_simulations.py` | COVID-period estimation and simulations, all Section 4 figures, the ARMA(1,1) appendix figure, and the numbers quoted in the text. |
| `paper_robustness.py` | Appendix robustness figures (varying the fiscal-rule coefficient alpha and the wage-PC slope kappa_w). |
| `Data/` | Raw input files that are not on FRED (see "Data sources"). |
| `ymarkov_combined.txt`, `ygrid_combined.txt` | Idiosyncratic income process (transition matrix generator and grid). |
| `lorenz_nw_scf_2019.raw` | Net-worth Lorenz curve, 2019 Survey of Consumer Finances. |
| `cache/` | Intermediate results used in the paper: data caches, steady states, estimated shock processes, and simulated paths (about 10 MB). |

Figures are written to `Figures/`. The package ships the full set of 37 figure files produced by the commands below.

## Computational environment

- Python 3.13 (tested with 3.13.2)
- Packages (tested versions): `sequence-jacobian` 1.0.0, `numba` 0.64,
  `numpy` 2.4, `scipy` 1.17, `pandas` 2.2, `matplotlib` 3.10, `statsmodels`
  0.14, `fredapi` 0.5, `openpyxl` 3.1.
- Only `paper_data.py` and `paper_regressions.py` need internet access and a
  free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html), set
  in the environment variable `FRED_API_KEY`. Everything else runs offline.
- Run every script **from the package root**. All paths are relative to it.
- Other environments (like a lower Python3 version) also probably work.

## Quick replication (minutes, from `cache/`)

```
python paper_model.py                        # Table 1: re-solve gamma, sigma_v, theta, kappa_w, eta, k_w
python paper_regressions.py                  # alpha, rho_r, rho_Ybar (needs FRED_API_KEY)
python paper_irfs.py                         # Figures 2-4 and the iMPC figure (~3 min)
python paper_covid_simulations.py figures    # Figures 5-9 and the ARMA(1,1) figure (< 1 min)
python paper_covid_simulations.py numbers    # numbers quoted in the text (seconds)
python paper_robustness.py plots             # robustness figures (seconds)
```

`paper_irfs.py` solves the model directly and does not use stored
results. Its first run builds and caches the steady-state Jacobians in
`cache/`.

## Full replication (from raw data)

Every stage reads the outputs of the previous ones from `cache/` and
overwrites them. Run the stages one at a time, since the solver is already
multithreaded. To start completely from scratch, delete `cache/` first.

1. **Data** (minutes):
   `python paper_data.py --force`
   rebuilds `cache/revision_covid_cache.pkl` and `cache/ffr_covid_cache.pkl`.
2. **Steady states** (minutes): built automatically on first use. You can
   also run `python paper_steady_state.py`.
3. **COVID estimation and simulations** (each estimation stage takes several
   hours):
   ```
   python paper_covid_simulations.py estimate HANK          # (beta_t, rho_t), HANK and HANK CC
   python paper_covid_simulations.py estimate RANK          # (beta_t, rho_t), RANK and RANK CC
   python paper_covid_simulations.py estimate-arma HANK     # ARMA(1,1) appendix variant
   python paper_covid_simulations.py estimate-arma RANK
   python paper_covid_simulations.py replay                 # simulated paths, 4 models
   python paper_covid_simulations.py replay arma
   python paper_covid_simulations.py ablation               # shock decomposition
   python paper_covid_simulations.py causal                 # fiscal effect, real-rate rule
   python paper_covid_simulations.py estimate-taylor HANK   # Taylor-rule estimation
   python paper_covid_simulations.py estimate-taylor HANK_CC
   python paper_covid_simulations.py taylor-nogt            # fiscal effect, Taylor rule
   python paper_covid_simulations.py figures
   python paper_covid_simulations.py numbers
   ```
4. **Robustness** (several hours per parameter value and model):
   `python paper_robustness.py alpha` and `python paper_robustness.py kappa`.
   These stages skip any (model, parameter) pair whose result is already in
   `cache/paper_robust_*.pkl`. Delete those files to recompute them.

**Starting values.** Each quarter's discount-factor shock and its persistence
are found by numerical search (an inner Newton solve for beta_t and a
bounded scalar search for rho_t). `estimate` and `estimate-taylor` start this
search from the estimates stored in `cache/`, which are the values reported
in the paper. If those files are absent, the search starts from
(beta, rho) = (0.05, 0.85). The ARMA estimation starts from the AR(1)
estimates. The robustness estimations always start from (0.05, 0.85).

## Mapping from the paper to the code

| Paper item | Script / stage | Output (`Figures/`) |
|---|---|---|
| Table 1 (calibration) | `paper_model.py`, `paper_steady_state.py`; alpha from `paper_regressions.py` | printed |
| Fig. 2: Partial equilibrium goods prices and labor demand | `paper_irfs.py supply` | `supply_curve.pdf` |
| Fig. 3: Size dependence of fiscal stimulus | `paper_irfs.py sizedep` | `size_dependence_real_fiscal_{GDP,pi,w,Pi_int,Pi_tot,lab_inc}.pdf` |
| Fig. 4: Impulse responses to fiscal stimulus | `paper_irfs.py irf` | `IRF_{GDP,pi,w}.pdf`, `IRF_{GDP,pi,w}_cc.pdf` |
| Fig. 5: Covid simulations | `paper_covid_simulations.py figures` | `finalmodel_real_4model_{GDP,pi,w}_hankwbeta_ss.pdf`, `finalmodel_shock_series.pdf` |
| Fig. 6: HANK vs. RANK inflation decomposition | `paper_covid_simulations.py figures` | `finalmodel_real_tailsplice_pi_hankref_hankwbeta_ss.pdf` |
| Fig. 7: Effect of fiscal policy during Covid | `paper_covid_simulations.py figures` | top row `fiscal_effect_4model_{GDP,pi,w}_hankwbeta_ss.pdf`; bottom row `hank_real_vs_taylor_gt_effect_{GDP,pi,w}.pdf` |
| Fig. 8: Simulation shock decomposition | `paper_covid_simulations.py figures` | `finalmodel_real_ablation_{HANK,HANK_CC}_{GDP,pi,w}.pdf` |
| Fig. 9: Profit decomposition | `paper_covid_simulations.py figures` | `finalmodel_profits_{total,int,ret}_hankonly.pdf` |
| Appendix: Intertemporal marginal propensities to consume | `paper_irfs.py mpc` | `MPC.pdf` |
| Appendix: Role of the fiscal rule | `paper_robustness.py alpha` / `plots` | `covid_fiscal_effects_varyalpha_estrho_20q2.pdf` |
| Appendix: Role of the wage Phillips curve | `paper_robustness.py kappa` / `plots` | `covid_vary_wage_PC.pdf` |
| Appendix: HANK vs. RANK inflation decomposition (ARMA(1,1) beta) | `paper_covid_simulations.py figures` | `finalmodel_arma11_tailsplice_pi_hankref_hankwbeta_ss.pdf` |
| alpha = 0.03, rho_r = 0.82, rho_Ybar = 0.96 (calibration and data appendices) | `paper_regressions.py` | printed |
| Numbers in the introduction, Section 4, and the conclusion | `paper_covid_simulations.py numbers` | printed |

How the printed output of `numbers` maps to the text:

- **Fiscal effect (real-rate rule):** fiscal inflation in 2021Q2 (2.4 vs. 1.8
  p.p.; 0.9 p.p. in RANK CC), cumulative fiscal price growth (5.8% vs. 5.0%),
  the cumulative GDP effect (15.2%), and the peak real-wage effect (0.2% vs.
  0.7%).
- **Simulation levels (all shocks):** excess inflation in 2021Q2 (4.4 vs. 2.4
  p.p.) and cumulative excess price growth over 2020Q2-2023Q4 (3.7% vs. 1.9%;
  -1.0% and -1.7% in RANK CC and RANK). The data's excess price growth (6.6%)
  is printed in the fiscal-effect block.
- **Real wage max/min (all shocks):** real wages rise by up to 0.1%
  (conventional) and fall by up to 0.4% (capacity constraint).
- **Shares attributable to fiscal policy:** the "about 55%" of the 2021Q2
  inflation spike in the conclusion is the fiscal effect's share of the
  capacity-constraint HANK model's 2021Q2 inflation.
- **Taylor rule:** fiscal inflation in 2021Q2 (1.7 vs. 1.3 p.p.) and cumulative
  fiscal price growth (4.2% vs. 3.8%).

Model labels in the code are as follows: `HANK` / `RANK` are the conventional
models and `HANK_CC` / `RANK_CC` the capacity-constraint models. "hankwbeta_ss"
refers to the RANK convention described in the Section 4 footnote: the wage
Phillips curve discounts at HANK's steady-state beta of 0.954.

## Data sources

| Data | Source | Used in |
|---|---|---|
| Nominal GDP (`GDP`), PCE price index (`PCEPI`), core PCE price index (`PCEPILFE`), effective federal funds rate (`DFF`), industrial capacity (`CAPB50001SQ`) | FRED, Federal Reserve Bank of St. Louis | `paper_data.py`, `paper_regressions.py` |
| Federal current tax receipts (`NA000327Q`), total federal debt (`GFDEBTN`) | FRED | `paper_regressions.py` |
| Mean SPF forecasts of the 3-month T-bill rate, headline PCE inflation, and nominal GDP (`Data/Mean_TBILL_Level.xlsx`, `Data/Mean_PCE_Level.xlsx`, `Data/Mean_NGDP_Level.xlsx`) | Survey of Professional Forecasters, Federal Reserve Bank of Philadelphia | `paper_data.py` |
| Effects of Selected Federal Pandemic Response Programs (`Data/NEA-CU23-Pandemic-Response-Q-and-A-Federal.xlsx`) | Bureau of Economic Analysis | `paper_data.py` |
| Net-worth Lorenz curve (`lorenz_nw_scf_2019.raw`) | 2019 Survey of Consumer Finances | `paper_steady_state.py` |
| Income process (`ymarkov_combined.txt`, `ygrid_combined.txt`) | Kaplan, Moll, and Violante (2018) | `paper_model.py` |


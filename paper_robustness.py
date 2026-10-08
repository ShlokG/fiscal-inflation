"""paper_robustness.py — Appendix robustness figures: the effect of the COVID
fiscal stimulus varying the fiscal-rule coefficient alpha and the wage-PC
slope kappa_w.

Produces, in Figures/:
  covid_fiscal_effects_varyalpha_estrho_20q2.pdf   (paper Fig 'vary_alpha', appendix)
  covid_vary_wage_PC.pdf                           (paper Fig 'vary_kappa', appendix)

For each parameter value:
  1. rebuild the steady states (paper_steady_state.build(alpha_Tau, kappa_w));
  2. re-estimate (beta_t, rho_t) each quarter of 2020Q2-2023Q4 for HANK
     conventional and HANK CC (beta matches detrended GDP exactly; rho
     minimizes the SPF 4-quarter expected-GDP loss) — same method as
     paper_covid_simulations.estimate_model, one-period G,T;
  3. run the FULL and no-G,T trajectories (independent-trajectory fiscal
     counterfactual) and store the impact paths;
  4. plot the fiscal inflation effect (annualized pp) and fiscal real-wage
     effect (percent, per-model SS normalization) — capacity-constraint solid,
     conventional dashed, one color per parameter value.

Grids:
  alpha   in {0.03, 0.035, 0.04}           (kappa_w at baseline 0.0092703)
  kappa_w in {0.005, 0.0092703, 0.011}     (alpha at baseline 0.03)

WARNING: EACH (parameter value, model) pair requires a full per-quarter
re-estimation — expect several hours per pair; run ONE stage at a time:
    python paper_robustness.py alpha      # runs + caches + plots vary-alpha
    python paper_robustness.py kappa      # runs + caches + plots vary-kappa_w
    python paper_robustness.py plots      # plots only, from cached pickles
Intermediate results are cached per (model, alpha, kappa) in
cache/paper_robust_{model}_a{alpha}_k{kappa}.pkl; an existing pickle is
loaded instead of recomputed, so interrupted runs resume.
"""
import os
import pickle
import sys
import time
from copy import deepcopy

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict

import paper_model as PM
import paper_steady_state as PSS
from paper_model import hank_real, hank_cc_real
from paper_covid_simulations import (load_data, quiet, set_block_options,
                                     propagate_state, newton_on_beta,
                                     RHO_R, RHO_YBAR, RHO_G, RHO_T,
                                     RHO_BOUNDS, RHO_XATOL, RHO_MAXITER)

FIGDIR = 'Figures'
KAPPA_BASE = PSS.KAPPA_W_SLOPE          # 0.0092703
ALPHA_BASE = 0.03
ALPHA_GRID = [0.03, 0.035, 0.04]
KAPPA_GRID = [0.005, KAPPA_BASE, 0.011]

# figure style
title_fontsize, label_fontsize, legend_fontsize = 22, 18, 16
tick_fontsize, line_width = 16, 3
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _pkl(model, alpha, kappa):
    return f'cache/paper_robust_{model}_a{alpha}_k{kappa:.6f}.pkl'


def _build_shocks(D, t, beta_t, rho_t, is_cc, gt_on=True):
    sh = {'beta_hi': beta_t * rho_t ** np.arange(D['T']),
          'G': np.zeros(D['T']), 'T': np.zeros(D['T'])}
    if gt_on:
        sh['G'] = float(D['shocks']['G'].iloc[t]) * RHO_G ** np.arange(D['T'])
        sh['T'] = float(D['shocks']['T'].iloc[t]) * RHO_T ** np.arange(D['T'])
    dr = np.zeros(D['T'])
    dr[0] = float(D['shocks']['r_ante'].iloc[t])
    dr[1:D['r_expect'].shape[1]] = D['r_expect'][t, :(D['T'] - 1)]
    sh['r_ante'] = dr
    if is_cc:
        sh['Y_bar'] = float(D['shocks']['Y_bar'].iloc[t]) * RHO_YBAR ** np.arange(D['T'])
    return sh


def run_one(model_key, alpha, kappa):
    """Estimate (beta, rho) and compute FULL/noGT paths for one parameter
    combination. Cached; returns the result dict."""
    path = _pkl(model_key, alpha, kappa)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    set_block_options(maxit=300, tol=1e-8)
    is_cc = model_key == 'HANK_CC'
    model = hank_cc_real if is_cc else hank_real
    ssd = PSS.load_or_build(alpha_Tau=alpha, kappa_w=kappa, verbose=False)
    ss = deepcopy(ssd['ss_cc' if is_cc else 'ss_conv'])
    D = load_data()

    inputs = ['r_ante', 'beta_hi', 'G', 'T'] + (['Y_bar'] if is_cc else [])
    jac_path = f'cache/cached_jac_paperls_{model_key}_a{alpha}_k{kappa:.6f}.pkl'
    if os.path.exists(jac_path):
        Js, H_U = pickle.load(open(jac_path, 'rb'))
    else:
        print(f"  building jacobians for {model_key} a={alpha} k={kappa}...", flush=True)
        Js = quiet(model.partial_jacobians, ss, inputs=set(inputs) | set(['Y', 'w', 'pi']),
                   outputs={'asset_mkt', 'res_wpc', 'res_pc'}, T=D['T'])
        H_U = quiet(model.jacobian, ss, ['Y', 'w', 'pi'],
                    ['asset_mkt', 'res_wpc', 'res_pc'], T=D['T'], Js=Js)
        try:
            pickle.dump((Js, H_U), open(jac_path, 'wb'))
        except Exception:
            pass
    H_U_f = FactoredJacobianDict(H_U, D['T'])

    def make_solver(gt_on):
        def solve(t, b, r, ss_in, eps0=None):
            sh = _build_shocks(D, t, float(b), float(r), is_cc, gt_on=gt_on)
            return quiet(model.solve_impulse_nonlinear, ss, ss_initial=ss_in,
                         unknowns=['Y', 'w', 'pi'],
                         targets=['asset_mkt', 'res_wpc', 'res_pc'],
                         inputs=sh, H_U_factored=H_U_f, Js=Js,
                         maxit=300, tol=1e-8, verbose=False, internals=['hh'])
        return solve

    solve_full = make_solver(gt_on=True)

    # ---- estimation loop (same nested SPF-rho method as the baseline) -------
    n = D['shock_len']
    beta_arr, rho_arr = np.zeros(n), np.zeros(n)
    full = {k: np.full(n, np.nan) for k in ('GDP', 'pi', 'w')}
    ss_in = deepcopy(ss)
    beta0, rho0 = 0.05, 0.85
    for t in range(n):
        t0 = time.time()
        target_Y = float(D['data_Y'][t])
        ngdp_t = D['ngdp_expect'][t, :].astype(float)
        last_beta, last_slope, cache = [beta0], [None], {}

        def evaluate_rho(r_):
            k = round(float(r_), 4)
            if k in cache:
                return cache[k]
            b, irf, resid, slope = newton_on_beta(solve_full, t, last_beta[0],
                                                  float(r_), ss_in, target_Y,
                                                  slope0=last_slope[0])
            if (resid is None or irf is None
                    or not np.all(np.isfinite(irf['GDP'][:5])) or abs(resid) > 1e-4):
                cache[k] = (None, None, np.inf)
                return cache[k]
            loss = float(np.mean((np.asarray(irf['GDP'][1:5]) - ngdp_t) ** 2))
            last_beta[0] = b
            if slope is not None and np.isfinite(slope):
                last_slope[0] = slope
            cache[k] = (b, irf, loss)
            return cache[k]

        _ = evaluate_rho(rho0)
        res = opt.minimize_scalar(lambda r_: evaluate_rho(r_)[2],
                                  bounds=RHO_BOUNDS, method='bounded',
                                  options={'xatol': RHO_XATOL, 'maxiter': RHO_MAXITER})
        rho_star = float(res.x)
        b_s, irf_s, loss_s = evaluate_rho(rho_star)
        if round(rho0, 4) in cache and cache[round(rho0, 4)][2] < loss_s:
            rho_star = rho0
            b_s, irf_s, loss_s = cache[round(rho0, 4)]
        beta_arr[t], rho_arr[t] = b_s, rho_star
        for k in full:
            full[k][t] = float(irf_s[k][0])
        propagate_state(ss_in, ss, irf_s, het=True)
        beta0, rho0 = b_s, rho_star
        print(f"  [{model_key} a={alpha} k={kappa:.4f}] t={t:2d} "
              f"beta={b_s:+.4f} rho={rho_star:+.3f} "
              f"pi={full['pi'][t]*400:+.2f}pp [{time.time()-t0:.0f}s]", flush=True)

    # ---- no-G,T counterfactual trajectory -----------------------------------
    solve_nogt = make_solver(gt_on=False)
    nogt = {k: np.full(n, np.nan) for k in ('GDP', 'pi', 'w')}
    ss_in = deepcopy(ss)
    for t in range(n):
        irf = solve_nogt(t, float(beta_arr[t]), float(rho_arr[t]), ss_in)
        for k in nogt:
            nogt[k][t] = float(irf[k][0])
        propagate_state(ss_in, ss, irf, het=True)
        print(f"  [{model_key} a={alpha} k={kappa:.4f}] noGT t={t:2d}", flush=True)

    out = {'beta': beta_arr, 'rho': rho_arr, 'full': full, 'nogt': nogt,
           'w_ss': float(ss['w']), 'alpha': alpha, 'kappa': kappa}
    pickle.dump(out, open(path, 'wb'))
    print(f"Saved {path}")
    return out


# ---------------------------------------------------------------------------
# Plotting (two-panel layout)
# ---------------------------------------------------------------------------
def _plot_vary(results, labels_fmt, out_pdf):
    """results: list of (label_value, res_cc, res_conv). Two panels: fiscal
    inflation effect (annualized pp) and fiscal real-wage effect (%)."""
    n = len(results[0][1]['full']['pi'])
    q = pd.period_range('2020Q2', periods=n, freq='Q')
    dates = np.arange(n)
    labels = [str(q[i]) if i % 2 == 0 else '' for i in range(n)]

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    for j, (val, cc, conv) in enumerate(results):
        col = default_colors[j]
        plt.plot(dates, 400*(cc['full']['pi'] - cc['nogt']['pi']), '-',
                 label=f"Capacity Constraint, {labels_fmt(val)}", color=col,
                 linewidth=line_width)
        plt.plot(dates, 400*(conv['full']['pi'] - conv['nogt']['pi']), '--',
                 label=f"Conventional, {labels_fmt(val)}", color=col,
                 linewidth=line_width)
    plt.axhline(y=0, linestyle=':', color='black', linewidth=line_width-1)
    plt.xticks(dates, labels, rotation=45, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xlabel("Quarter", fontsize=label_fontsize)
    plt.ylabel("Percentage Points", fontsize=label_fontsize)
    plt.title("Annualized Inflation", fontsize=title_fontsize)
    plt.legend(framealpha=0, loc='upper right', fontsize=legend_fontsize-4)

    plt.subplot(1, 2, 2)
    for j, (val, cc, conv) in enumerate(results):
        col = default_colors[j]
        plt.plot(dates, 100*(cc['full']['w'] - cc['nogt']['w'])/cc['w_ss'], '-',
                 color=col, linewidth=line_width)
        plt.plot(dates, 100*(conv['full']['w'] - conv['nogt']['w'])/conv['w_ss'], '--',
                 color=col, linewidth=line_width)
    plt.axhline(y=0, linestyle=':', color='black', linewidth=line_width-1)
    plt.xticks(dates, labels, rotation=45, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xlabel("Quarter", fontsize=label_fontsize)
    plt.ylabel("Percent", fontsize=label_fontsize)
    plt.title("Real Wage", fontsize=title_fontsize)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(f"{FIGDIR}/{out_pdf}")
    plt.close()
    print(f"Wrote {FIGDIR}/{out_pdf}")


def stage_alpha(run=True):
    results = []
    for a in ALPHA_GRID:
        if run:
            cc = run_one('HANK_CC', a, KAPPA_BASE)
            conv = run_one('HANK', a, KAPPA_BASE)
        else:
            cc = pickle.load(open(_pkl('HANK_CC', a, KAPPA_BASE), 'rb'))
            conv = pickle.load(open(_pkl('HANK', a, KAPPA_BASE), 'rb'))
        results.append((a, cc, conv))
    _plot_vary(results, lambda a: rf"$\alpha = {a}$",
               "covid_fiscal_effects_varyalpha_estrho_20q2.pdf")


def stage_kappa(run=True):
    results = []
    for k in KAPPA_GRID:
        if run:
            cc = run_one('HANK_CC', ALPHA_BASE, k)
            conv = run_one('HANK', ALPHA_BASE, k)
        else:
            cc = pickle.load(open(_pkl('HANK_CC', ALPHA_BASE, k), 'rb'))
            conv = pickle.load(open(_pkl('HANK', ALPHA_BASE, k), 'rb'))
        results.append((k, cc, conv))
    _plot_vary(results, lambda k: rf"$\kappa_w = {k:.3f}$",
               "covid_vary_wage_PC.pdf")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'plots'
    if stage == 'alpha':
        stage_alpha(run=True)
    elif stage == 'kappa':
        stage_kappa(run=True)
    elif stage == 'plots':
        stage_alpha(run=False)
        stage_kappa(run=False)
    else:
        raise SystemExit("stage must be alpha | kappa | plots")

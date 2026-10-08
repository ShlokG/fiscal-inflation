"""paper_covid_simulations.py — COVID-era simulations (paper Section 4 and
the appendices).

Produces, in Figures/:

  finalmodel_real_4model_{GDP,pi,w}_hankwbeta_ss.pdf      (Fig: Covid simulations)
  finalmodel_shock_series.pdf                             (Fig: Covid simulations, shocks)
  finalmodel_real_tailsplice_pi_hankref_hankwbeta_ss.pdf  (Fig: HANK vs. RANK decomposition)
  finalmodel_arma11_tailsplice_pi_hankref_hankwbeta_ss.pdf (App: ARMA(1,1) version)
  fiscal_effect_4model_{GDP,pi,w}_hankwbeta_ss.pdf        (Fig: fiscal effect, top row)
  hank_real_vs_taylor_gt_effect_{GDP,pi,w}.pdf            (Fig: fiscal effect, bottom row)
  finalmodel_real_ablation_{HANK,HANK_CC}_{GDP,pi,w}.pdf  (Fig: shock decomposition)
  finalmodel_profits_{total,int,ret}_hankonly.pdf         (Fig: profit decomposition)

and prints every quantitative claim in the text (stage `numbers`).

METHOD (paper Sec 4 + appendix 'Data Inputs to Quantitative Exercise'):
  * Repeated-MIT-shock nonlinear perfect-foresight simulation in the sequence
    space: each quarter t in 2020Q2..2023Q4 the model is hit by that quarter's
    surprises, solved as a nonlinear impulse (unknowns {Y, w, pi}, targets
    {asset_mkt, res_wpc, res_pc}, T=300, quasi-Newton with the SS Jacobian),
    and the state is propagated into t+1.
  * Shocks/expectations per quarter t:
      G, T     : BEA pandemic-response data, fully unanticipated (expected 0
                 after impact).
      r_ante   : data value at h=0; SPF-derived real-rate expectations at
                 h>=1, extended beyond the SPF horizon by AR(1) at RHO_R=0.82.
      Y_bar    : industrial capacity (CC models only), expected AR(1) at
                 RHO_YBAR=0.96.
      beta_hi  : estimated (beta_t, rho_t): beta_t matches detrended GDP
                 exactly; rho_t minimizes the SPF 4-quarter expected-GDP loss.
  * "hankwbeta_ss" convention (paper footnote, Sec 4): RANK/RANK_CC use a wage
    PC whose discount is HANK's steady-state beta_bar = 0.954 plus RANK's OWN
    estimated beta shock path; RANK demand keeps its own beta_hi.
  * Fiscal effect: the FULL simulation minus a separately propagated
    simulation with G=T=0 (all other shocks on).
  * Taylor-rule variant (HANK and HANK CC only): static rule
    i = rss + 1.5 (pi-1) + (1/12) (Y-1) + eps_r; per quarter (beta, eps_r[0])
    are solved jointly so GDP[0] = data and i[0] = observed FFR (in deviations
    from its 1990-2019 mean); eps_r decays AR(1) at 0.82 in expectation. The
    Taylor fiscal effect holds (beta, rho, eps_r[0]) fixed and sets G=T=0.

INPUTS: the data caches cache/revision_covid_cache.pkl and
cache/ffr_covid_cache.pkl (built from raw data by paper_data.py) and the
steady states from paper_steady_state.py. All intermediate results are
stored in cache/ (file names below), so each stage can be re-run on its own.

USAGE (run from the replication-package root)
    python paper_covid_simulations.py figures        # all figures (minutes,
                                                     #   from the shipped results)
    python paper_covid_simulations.py numbers        # numbers quoted in the text
    python paper_covid_simulations.py estimate [HANK|RANK]       # (beta,rho), AR(1)
    python paper_covid_simulations.py estimate-arma [HANK|RANK]  # ARMA(1,1) variant
    python paper_covid_simulations.py estimate-taylor [HANK|HANK_CC]
    python paper_covid_simulations.py replay [arma]  # simulated paths
    python paper_covid_simulations.py ablation       # shock decomposition
    python paper_covid_simulations.py causal         # fiscal effect (real-rate rule)
    python paper_covid_simulations.py taylor-nogt    # fiscal effect (Taylor rule)
Run heavy stages ONE AT A TIME (the solver is already multithreaded).

ESTIMATION STRUCTURE: under the real-rate rule GDP is fully demand-determined
and the demand block is identical across the conventional and CC variants, so
(beta_t, rho_t) is estimated on the demand (IKC) block ONCE per household
type — `estimate HANK` covers HANK and HANK_CC; `estimate RANK` covers RANK and
RANK_CC. Under the Taylor rule inflation feeds back into demand, so
`estimate-taylor` runs per full model. Each estimation starts its search from
the shipped estimates in cache/ when they exist.

FULL RE-RUN ORDER (each stage sequentially):
    estimate HANK; estimate RANK; estimate-arma HANK; estimate-arma RANK;
    replay; replay arma; ablation; causal; estimate-taylor HANK;
    estimate-taylor HANK_CC; taylor-nogt; figures; numbers
"""
import io
import os
import pickle
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import sequence_jacobian as sj
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict

import paper_model as PM
import paper_steady_state as PSS
from paper_model import (
    hank_real, hank_cc_real,
    rank_real_wexo, rank_cc_real_wexo,
    hank_ikc, rank_ikc,
    hank_taylor, hank_cc_taylor,
    capitalization, fiscal, ra, pc,
    mc, labor, mc_convex, util, labor_convex, after_tax_income,
    wpc, wpc_wexo,
    ALPHA_PI, ALPHA_Y, ALPHA_TAU,
)

# ---------------------------------------------------------------------------
# Constants (paper Sec 4 / data appendix)
# ---------------------------------------------------------------------------
RHO_R    = 0.82     # AR(1) real-rate (and Taylor eps_r) expectation persistence
RHO_YBAR = 0.96     # AR(1) capacity expectation persistence
RHO_G    = 0.0      # fiscal shocks fully unanticipated (paper baseline)
RHO_T    = 0.0
T_SIM    = 300
HMAX     = 10       # anticipation horizon stored for the tailsplice figure

# Data caches, built from raw sources (FRED + SPF/BEA files in Data/) by
# paper_data.py. If missing: `python paper_data.py` (the FRED pull is pinned
# to a fixed vintage, so a rebuild reproduces them exactly).
DATA_CACHE = 'cache/revision_covid_cache.pkl'
FFR_CACHE  = 'cache/ffr_covid_cache.pkl'

# Intermediate results ('_ls' = the paper's labor-share calibration).
REPLAY_HWB_PKL      = 'cache/replay_finalmodel_real_paths_1per_hankwbeta_ss_ls.pkl'
REPLAY_ARMA_HWB_PKL = 'cache/replay_finalmodel_arma11_paths_hankwbeta_ss_ls.pkl'
ABLATION_PKL        = 'cache/finalmodel_real_ablation_ls.pkl'
TAYLOR_NOGT_PKL     = 'cache/taylor_ratematch_nogt_replay_ls.pkl'

# The (beta_t, rho_t) estimates are per HOUSEHOLD TYPE, estimated on the IKC
# demand block (identical across conv/CC): HANK and HANK_CC share one estimate
# pickle, RANK and RANK_CC share another. Taylor-rule estimates are per model
# (inflation feeds back into demand through the rule). RANK rows always use
# the hankwbeta_ss wage-PC convention (the `wexo` models). Jacobian caches
# (jac*) are built on first use.
MODELS = {
    'HANK':    dict(model=hank_real,    taylor=hank_taylor,    is_cc=False, het=True,
                    ikc=hank_ikc, ss_name='ss_conv',
                    jac='cache/cached_jac_paperls_full_HANK.pkl',
                    jac_ikc='cache/cached_jac_paperls_ikc_HANK.pkl',
                    jac_taylor='cache/cached_jac_paperls_taylor_HANK.pkl',
                    est='cache/ikc_est_HANK_ls.pkl',
                    est_taylor='cache/hank_finalmodel_taylor_nl_est_ratematch_ls.pkl',
                    est_arma='cache/ikc_arma_HANK_ls.pkl'),
    'HANK_CC': dict(model=hank_cc_real, taylor=hank_cc_taylor, is_cc=True,  het=True,
                    ikc=hank_ikc, ss_name='ss_cc',
                    jac='cache/cached_jac_paperls_full_HANK_CC.pkl',
                    jac_taylor='cache/cached_jac_paperls_taylor_HANK_CC.pkl',
                    est='cache/ikc_est_HANK_ls.pkl',          # shared with HANK
                    est_taylor='cache/hank_cc_finalmodel_taylor_nl_est_ratematch_ls.pkl',
                    est_arma='cache/ikc_arma_HANK_ls.pkl'),
    'RANK':    dict(wexo=rank_real_wexo, is_cc=False, het=False,
                    ikc=rank_ikc, ss_name='ss_rank',
                    jac_ikc='cache/cached_jac_paperls_ikc_RANK.pkl',
                    jac_wexo='cache/cached_jac_paperls_full_RANK_wexo.pkl',
                    est='cache/ikc_est_RANK_ls.pkl',
                    est_arma='cache/ikc_arma_RANK_ls.pkl'),
    'RANK_CC': dict(wexo=rank_cc_real_wexo, is_cc=True, het=False,
                    ikc=rank_ikc, ss_name='ss_rank_cc',
                    jac_wexo='cache/cached_jac_paperls_full_RANK_CC_wexo.pkl',
                    est='cache/ikc_est_RANK_ls.pkl',          # shared with RANK
                    est_arma='cache/ikc_arma_RANK_ls.pkl'),
}
TAYLOR_MODELS = ['HANK', 'HANK_CC']
CAUSAL_PKL = {  # fiscal-effect (FULL vs noGT) output, hankwbeta_ss for RANK
    'HANK':    'cache/causal_gt_nl_finalmodel_real_HANK_1per_ls.pkl',
    'HANK_CC': 'cache/causal_gt_nl_finalmodel_real_HANK_CC_1per_ls.pkl',
    'RANK':    'cache/causal_gt_nl_finalmodel_real_RANK_hankwbeta_ss_1per_ls.pkl',
    'RANK_CC': 'cache/causal_gt_nl_finalmodel_real_RANK_CC_hankwbeta_ss_1per_ls.pkl',
}

UNKS    = ['Y', 'w', 'pi']
TARGETS = ['asset_mkt', 'res_wpc', 'res_pc']
SAVE_PATHS = ['Y', 'C', 'Tau', 'w', 'pi', 'Delta_p', 'N', 'GDP', 'mc']

FIGDIR = 'Figures'

PLT_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.format": "pdf",
    "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "font.size": 16, "axes.labelsize": 19,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "axes.linewidth": 1.2,
}


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def set_block_options(maxit=600, tol=1e-8):
    """Iteration caps for the internally-solved blocks."""
    for blk in [capitalization, fiscal, ra, pc]:
        blk.solve_impulse_nonlinear_options['maxit'] = maxit
        blk.solve_impulse_nonlinear_options['tol'] = tol
        blk.impulse_nonlinear_options = blk.solve_impulse_nonlinear_options


# ---------------------------------------------------------------------------
# Data / expectations
# ---------------------------------------------------------------------------
def extend_r_expect_ar1(r_expect_raw, rho_r, T):
    """Continue each quarter's SPF real-rate expectation row with AR(1) decay
    at rho_r from its last nonzero (genuinely observed) entry."""
    r_expect_raw = np.asarray(r_expect_raw, dtype=float)
    if r_expect_raw.shape[1] >= T:
        return r_expect_raw[:, :T]
    out = np.zeros((r_expect_raw.shape[0], T))
    for i in range(r_expect_raw.shape[0]):
        row = r_expect_raw[i, :]
        nz = np.where(row != 0)[0]
        k, v = (nz[-1], row[nz[-1]]) if len(nz) else (-1, 0.0)
        out[i, :] = np.concatenate((row[:(k + 1)], v * rho_r ** np.arange(1, T - k)))
    return out


def load_data():
    """COVID shocks, data outcomes and expectations from the data cache.
    Returns dict with: shocks (DataFrame-like), data_Y, data_core_pi, ngdp_expect,
    r_expect (AR(1)-extended to T), T, shock_len."""
    if not os.path.exists(DATA_CACHE):
        raise FileNotFoundError(
            f"{DATA_CACHE} missing — build it with `python paper_data.py` "
            f"(exactly reproducible: the FRED pull is vintage-pinned)")
    cache = pickle.load(open(DATA_CACHE, 'rb'))
    shocks = cache['shocks']
    # The simulation horizon T_SIM may exceed the horizon the data cache was
    # built with (cache['T'] = 200): the SPF r-expectation rows are only ~14
    # quarters of genuine data anyway and are AR(1)-extended to T_SIM below;
    # all shock paths are constructed at T_SIM directly.
    T = T_SIM
    ngdp = np.asarray(cache['ngdp_expect'])
    shock_len = len(shocks['G'])
    if ngdp.shape[0] == shock_len + 1:
        ngdp = ngdp[1:]
    assert ngdp.shape == (shock_len, 4)
    return dict(shocks=shocks,
                data_Y=np.asarray(cache['out_vals']['Y'], float),
                data_pi=np.asarray(cache['out_vals']['pi'], float),
                data_core_pi=np.asarray(cache['out_vals_core_pi'], float),
                ngdp_expect=ngdp,
                r_expect=extend_r_expect_ar1(cache['r_expect'], RHO_R, T),
                T=T, shock_len=shock_len)


def load_ss_all():
    d = PSS.load_or_build(verbose=False)
    ss = {k: d[k] for k in ('ss_cc', 'ss_conv', 'ss_rank', 'ss_rank_cc')}
    for s in ss.values():   # profits reporting values, as in the drivers
        if 'Pi' not in s.toplevel:
            dp = float(s['Delta_p']) if 'Delta_p' in s.toplevel else 1.0
            s['Pi'] = float(s['Y']) / dp - float(s['w']) * float(s['N'])
        if 'Pi_int' not in s.toplevel:
            s['Pi_int'] = float(s['mc']) * float(s['Y']) - float(s['w']) * float(s['N'])
    return ss


def load_est(path):
    """(beta, rho) arrays from an estimation pickle (first dict entry with 'beta')."""
    d = pickle.load(open(path, 'rb'))
    key = [k for k in d if isinstance(d[k], dict) and 'beta' in d[k]][0]
    R = d[key]
    return (np.asarray(R['beta'], float),
            np.asarray(R['rho'] if 'rho' in R else R['p1'], float), R)


def get_jacobians(model, ss, inputs, path, T, unknowns=None, targets=None):
    """SS Jacobians for the quasi-Newton solver, cached at `path`."""
    unknowns = unknowns or UNKS
    targets = targets or TARGETS
    if os.path.exists(path):
        Js, H_U = pickle.load(open(path, 'rb'))
    else:
        print(f"  building SS jacobians -> {path}", flush=True)
        Js = quiet(model.partial_jacobians, ss,
                   inputs=set(inputs) | set(unknowns), outputs=set(targets), T=T)
        H_U = quiet(model.jacobian, ss, unknowns, targets, T=T, Js=Js)
        try:
            pickle.dump((Js, H_U), open(path, 'wb'))
        except Exception:
            pass
    return Js, FactoredJacobianDict(H_U, T)


# ---------------------------------------------------------------------------
# Engine: per-quarter shocks, nonlinear solve, state propagation
# ---------------------------------------------------------------------------
def arma11_path(beta0, phi, theta, T):
    """Expected beta path under an ARMA(1,1): b[1] = b0 (phi + theta), then AR."""
    b = np.zeros(T)
    b[0] = beta0
    if T > 1:
        b[1] = beta0 * (phi + theta)
    for h in range(2, T):
        b[h] = phi * b[h - 1]
    return b


def build_shocks(key, D, t, beta_t, rho_t, *, gt_on=True, r_on=True,
                 ybar_on=True, beta_w=None, arma_theta=None,
                 taylor_eps0=None):
    """Anticipated shock paths fed to the model at quarter t.

    gt_on/r_on/ybar_on switch shock groups (for the ablation / counterfactuals;
    switched-off groups are fed zero paths). beta_w=(level path) adds the
    wpc_wexo discount input. arma_theta activates the ARMA(1,1) beta path.
    taylor_eps0 (not None) replaces the real-rate input with the Taylor eps_r
    path eps0 * RHO_R**h (used with the *_taylor models)."""
    cf = MODELS[key]
    sh = {'G': np.zeros(D['T']), 'T': np.zeros(D['T'])}
    if arma_theta is None:
        sh['beta_hi'] = beta_t * rho_t ** np.arange(D['T'])
    else:
        sh['beta_hi'] = arma11_path(beta_t, rho_t, arma_theta, D['T'])
    if gt_on:
        sh['G'] = float(D['shocks']['G'].iloc[t]) * RHO_G ** np.arange(D['T'])
        sh['T'] = float(D['shocks']['T'].iloc[t]) * RHO_T ** np.arange(D['T'])
    if taylor_eps0 is not None:
        sh['eps_r'] = float(taylor_eps0) * RHO_R ** np.arange(D['T'])
    else:
        dr = np.zeros(D['T'])
        if r_on:
            dr[0] = float(D['shocks']['r_ante'].iloc[t])
            dr[1:D['r_expect'].shape[1]] = D['r_expect'][t, :(D['T'] - 1)]
        sh['r_ante'] = dr
    if cf['is_cc']:
        sh['Y_bar'] = (float(D['shocks']['Y_bar'].iloc[t]) * RHO_YBAR ** np.arange(D['T'])
                       if ybar_on else np.zeros(D['T']))
    if beta_w is not None:
        sh['beta_w'] = beta_w
    return sh


def propagate_state(ss_in, ss, irf, het):
    """Carry the end-of-quarter state into the next quarter's initial condition."""
    ss_in.update({k: float(ss[k]) + float(irf[k][0]) for k in irf
                  if hasattr(irf[k], '__len__') and not isinstance(irf[k], dict)})
    if het and 'hh' in getattr(irf, 'internals', {}):
        for k, v in irf.internals['hh'].items():
            if k in ss_in.internals.get('hh', {}):
                try:
                    ss_in.internals['hh'][k] = ss.internals['hh'][k] + v[1, ...]
                except Exception:
                    pass


def run_trajectory(key, ss, D, beta, rho, *, use_wexo=False, taylor_eps=None,
                   arma_thetas=None, record_paths=False, label='',
                   maxit=600, tol=1e-8, skip_fail=False, **shock_flags):
    """State-propagated per-quarter simulation of model `key`.

    use_wexo: RANK wpc_wexo variant with the hankwbeta_ss discount (level =
      HANK beta_bar, shock = this model's own beta path).
    taylor_eps: array of impact eps_r values -> use the Taylor-rule model
      (HANK and HANK_CC only).
    arma_thetas: array of MA parameters -> ARMA(1,1) beta paths.
    record_paths: additionally store the full anticipated supply-block I/O and
      propagated states (needed by the tailsplice figure).
    Returns dict of impact arrays (+ paths/ss_in/GDP_fore if record_paths).
    """
    cf = MODELS[key]
    if taylor_eps is not None:
        assert key in TAYLOR_MODELS, f"no Taylor-rule variant for {key}"
        model = cf['taylor']
        ss = deepcopy(ss)
        ss.update({'rss': float(ss['r_ante']), 'alpha_pi': ALPHA_PI,
                   'alpha_y': ALPHA_Y, 'eps_r': 0.0, 'i': float(ss['r_ante'])})
        jac_path = cf['jac_taylor']
        inputs = ['eps_r', 'beta_hi', 'G', 'T'] + (['Y_bar'] if cf['is_cc'] else [])
    elif use_wexo:
        model = cf['wexo']
        ss = deepcopy(ss)
        ss['beta_w'] = PM.hank_wpc_beta_bar(load_ss_all()['ss_conv'])
        jac_path = cf['jac_wexo']
        inputs = ['r_ante', 'beta_hi', 'beta_w', 'G', 'T'] + (['Y_bar'] if cf['is_cc'] else [])
    else:
        assert key.startswith('HANK'), "RANK rows use the wexo wage PC (use_wexo=True)"
        model = cf['model']
        jac_path = cf['jac']
        inputs = ['r_ante', 'beta_hi', 'G', 'T'] + (['Y_bar'] if cf['is_cc'] else [])

    Js, H_U_f = get_jacobians(model, ss, inputs, jac_path, D['T'])
    n = D['shock_len']
    rec_keys = ['GDP', 'pi', 'w', 'B', 'Pi', 'Pi_int', 'N', 'mc', 'r_ante', 'i']
    out = {k: np.full(n, np.nan) for k in rec_keys}
    out['beta'], out['rho'] = np.asarray(beta, float).copy(), np.asarray(rho, float).copy()
    out['ss_ref'] = {k: float(ss[k]) for k in ss.keys()
                     if np.ndim(ss[k]) == 0 and not isinstance(ss[k], dict)}
    if record_paths:
        out.update({'GDP_fore': np.full((n, HMAX + 1), np.nan),
                    'paths': [None] * n, 'ss_in': [None] * n,
                    'beta_hi_path': np.full((n, D['T']), np.nan),
                    'r_ante_path': np.full((n, D['T']), np.nan),
                    'Y_bar_path': (np.full((n, D['T']), np.nan) if cf['is_cc'] else None)})

    ss_in = deepcopy(ss)
    for t in range(n):
        bw = None
        if use_wexo:   # hankwbeta_ss: own beta shocks on top of HANK's level
            if arma_thetas is not None:
                bw = arma11_path(float(beta[t]), float(rho[t]),
                                 float(arma_thetas[t]), D['T'])
            else:
                bw = float(beta[t]) * float(rho[t]) ** np.arange(D['T'])
        sh = build_shocks(key, D, t, float(beta[t]), float(rho[t]), beta_w=bw,
                          arma_theta=(None if arma_thetas is None else float(arma_thetas[t])),
                          taylor_eps0=(None if taylor_eps is None else taylor_eps[t]),
                          **shock_flags)
        if record_paths:
            out['ss_in'][t] = {k: float(ss_in[k]) for k in ss_in.keys()
                               if np.ndim(ss_in[k]) == 0 and not isinstance(ss_in[k], dict)}
        kw = dict(ss_initial=ss_in, unknowns=UNKS, targets=TARGETS, inputs=sh,
                  H_U_factored=H_U_f, Js=Js, maxit=maxit, tol=tol, verbose=False)
        if cf['het']:
            kw['internals'] = ['hh']
        try:
            irf = quiet(model.solve_impulse_nonlinear, ss, **kw)
        except Exception as e:
            if not skip_fail:
                raise
            print(f"  [{label or key}] t={t:2d} NO CONVERGENCE ({type(e).__name__}) "
                  f"-> NaN, subset halted", flush=True)
            break
        for k in rec_keys:
            if k in irf.toplevel:
                out[k][t] = float(irf[k][0])
        if record_paths:
            out['GDP_fore'][t] = np.asarray(irf['GDP'][:HMAX + 1])
            out['beta_hi_path'][t] = sh['beta_hi'] if bw is None else sh['beta_hi']
            out['r_ante_path'][t] = sh.get('r_ante', np.zeros(D['T']))
            if cf['is_cc']:
                out['Y_bar_path'][t] = sh['Y_bar']
            out['paths'][t] = {k: np.asarray(irf[k], float)
                               for k in SAVE_PATHS if k in irf.toplevel}
        propagate_state(ss_in, ss, irf, cf['het'])
        print(f"  [{label or key}] t={t:2d}  GDP={out['GDP'][t]*100:+7.3f}%  "
              f"pi={out['pi'][t]*400:+6.2f}pp", flush=True)
    return out


# ---------------------------------------------------------------------------
# Estimation of (beta_t, rho_t)  [stage: estimate]
# ---------------------------------------------------------------------------
TOL_GDP, MAX_NEWTON_BETA = 1e-7, 25
RHO_BOUNDS, RHO_XATOL, RHO_MAXITER = (-0.999, 0.999), 1e-3, 25


def make_quarter_solver(key, model, ss, Js, H_U_f, D, het, taylor=False,
                        unknowns=None, targets=None, ikc=False,
                        arma_theta_cell=None):
    """Returns solve(t, beta, rho, ss_in, eps0) -> full irf for one quarter.
    ikc=True drops the Y_bar shock (no supply side). arma_theta_cell, if
    given, is a 1-element list holding the current ARMA(1,1) MA parameter."""
    unknowns = unknowns or UNKS
    targets = targets or TARGETS

    def solve(t, b, r, ss_in, eps0=None):
        theta = None if arma_theta_cell is None else float(arma_theta_cell[0])
        sh = build_shocks(key, D, t, float(b), float(r),
                          taylor_eps0=(eps0 if taylor else None),
                          arma_theta=theta)
        if ikc:
            sh.pop('Y_bar', None)
        kw = dict(ss_initial=ss_in, unknowns=unknowns, targets=targets, inputs=sh,
                  H_U_factored=H_U_f, Js=Js, maxit=300, tol=1e-8, verbose=False)
        if het:
            kw['internals'] = ['hh']
        return quiet(model.solve_impulse_nonlinear, ss, **kw)
    return solve


def newton_on_beta(solve, t, beta0, rho_t, ss_in, target_Y, slope0=None, eps0=None):
    """Secant Newton on beta_t so GDP[0] = target_Y."""
    cache = {}

    def f(b):
        k = round(float(b), 8)
        if k in cache:
            return cache[k]
        try:
            irf = solve(t, float(b), rho_t, ss_in, eps0)
            cache[k] = (target_Y - float(irf['GDP'][0]), irf)
        except Exception:
            cache[k] = (None, None)
        return cache[k]

    r0, irf0 = f(beta0)
    if r0 is None:
        for db in [0.01, -0.01, 0.05, -0.05]:
            r0, irf0 = f(beta0 + db)
            if r0 is not None:
                beta0 = beta0 + db
                break
        if r0 is None:
            return None, None, None, slope0
    best_b, best_irf, best_r = beta0, irf0, r0
    slope = slope0
    b_prev = r_prev = None
    for _ in range(MAX_NEWTON_BETA):
        if abs(best_r) < TOL_GDP:
            break
        if b_prev is not None and abs(best_b - b_prev) > 1e-12:
            slope = (best_r - r_prev) / (best_b - b_prev)
        elif slope is None:
            h = 0.05 * max(abs(best_b), 1e-3)
            r_h, _ = f(best_b + h)
            if r_h is None:
                break
            slope = (r_h - best_r) / h
        if not np.isfinite(slope) or abs(slope) < 1e-12:
            break
        step = -best_r / slope
        alpha, improved = 1.0, False
        for _ in range(6):
            tb = best_b + alpha * step
            tr, tirf = f(tb)
            if tr is not None and abs(tr) < abs(best_r):
                b_prev, r_prev = best_b, best_r
                best_b, best_irf, best_r = tb, tirf, tr
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    return best_b, best_irf, best_r, slope


def estimate_model(key, taylor=False, out_pkl=None):
    """Nested SPF-rho estimation of (beta_t, rho_t).

    Real-rate rule (taylor=False): estimated on the IKC (demand-block-only)
    model — GDP is fully demand-determined, so the supply side is irrelevant
    for the fit and one estimation per HOUSEHOLD TYPE covers both the
    conventional and CC variants. Pass key in {'HANK', 'RANK'}.

    Taylor rule (taylor=True): inflation feeds back into demand through the
    rule, so (beta_t, rho_t, eps_r0_t) are estimated per FULL model; eps_r[0]
    is solved so the model's nominal rate matches the observed FFR on impact.
    Pass key in {'HANK', 'HANK_CC', 'RANK', 'RANK_CC'}.

    Outer: bounded scalar search on rho_t minimizing the SPF expected-GDP loss
           mean((GDP[1:5] - E_t^SPF[GDP])^2).
    Inner: Newton on beta_t s.t. GDP[0] = data (Taylor: + damped fixed point
           on eps_r[0] s.t. i[0] = di_target).
    """
    set_block_options(maxit=300, tol=1e-8)
    D = load_data()
    cf = MODELS[key]
    ss_all = load_ss_all()
    ss = deepcopy(ss_all[cf['ss_name']])
    if taylor:
        assert key in TAYLOR_MODELS, "Taylor-rule estimation: HANK or HANK_CC"
        model = cf['taylor']
        ss.update({'rss': float(ss['r_ante']), 'alpha_pi': ALPHA_PI,
                   'alpha_y': ALPHA_Y, 'eps_r': 0.0, 'i': float(ss['r_ante'])})
        inputs = ['eps_r', 'beta_hi', 'G', 'T'] + (['Y_bar'] if cf['is_cc'] else [])
        unks, tgts = UNKS, TARGETS
        Js, H_U_f = get_jacobians(model, ss, inputs, cf['jac_taylor'], D['T'],
                                  unknowns=unks, targets=tgts)
        di_target = np.asarray(pickle.load(open(FFR_CACHE, 'rb'))['di_target'], float)
    else:
        assert key in ('HANK', 'RANK'), \
            "real-rate estimation runs on the IKC block per household type"
        model = cf['ikc']
        inputs = ['r_ante', 'beta_hi', 'G', 'T']       # no Y_bar: no supply side
        unks, tgts = ['Y'], ['asset_mkt']
        Js, H_U_f = get_jacobians(model, ss, inputs, cf['jac_ikc'], D['T'],
                                  unknowns=unks, targets=tgts)
        di_target = None
    solve = make_quarter_solver(key, model, ss, Js, H_U_f, D, cf['het'], taylor,
                                unknowns=unks, targets=tgts,
                                ikc=(not taylor))

    # Warm start: the search for each quarter's (beta_t, rho_t) starts from the
    # shipped estimates (the values reported in the paper) when they exist,
    # otherwise from (beta, rho) = (0.05, 0.85).
    warm = out_pkl or (cf['est_taylor'] if taylor else cf['est'])
    if os.path.exists(warm):
        beta_init, rho_init, _ = load_est(warm)
        print(f"  warm start from {warm}", flush=True)
    else:
        print(f"  {warm} not found: cold start", flush=True)
        beta_init, rho_init = (np.full(D['shock_len'], 0.05),
                               np.full(D['shock_len'], 0.85))

    def newton_beta_eps(t, beta0, e0_0, rho_t, ss_in, target_Y, target_i, slope0=None):
        """Taylor rate-matched inner solve: damped fixed point on eps0 around
        the beta-Newton (di[0]/deps0 ~ 1)."""
        e0, beta_w, slope = float(e0_0), float(beta0), slope0
        best = (None, None, None, slope, e0, None)
        for _ in range(12):
            b, irf, r, slope = newton_on_beta(solve, t, beta_w, rho_t, ss_in,
                                              target_Y, slope0=slope, eps0=e0)
            if r is None or irf is None:
                return best if best[1] is not None else (None, None, None, slope, e0, None)
            beta_w = b
            i_resid = target_i - float(irf['i'][0])
            best = (b, irf, r, slope, e0, i_resid)
            if abs(i_resid) < 1e-7:
                break
            e0 = e0 + 0.9 * i_resid
        return best

    def estimate_quarter(t, beta0, rho0, ss_in, e0_warm=None):
        target_Y = float(D['data_Y'][t])
        ngdp_t = D['ngdp_expect'][t, :].astype(float)
        last_beta, last_slope = [float(beta0)], [None]
        last_e0 = [float(e0_warm) if e0_warm is not None else 0.0]
        cache = {}

        def evaluate_rho(r_):
            k = round(float(r_), 4)
            if k in cache:
                return cache[k]
            if taylor:
                b, irf, resid, slope, e0, i_resid = newton_beta_eps(
                    t, last_beta[0], last_e0[0], float(r_), ss_in,
                    target_Y, float(di_target[t]), slope0=last_slope[0])
            else:
                b, irf, resid, slope = newton_on_beta(
                    solve, t, last_beta[0], float(r_), ss_in, target_Y,
                    slope0=last_slope[0])
                e0, i_resid = None, None
            bad = (resid is None or irf is None
                   or not np.all(np.isfinite(irf['GDP'][:5])) or abs(resid) > 1e-4
                   or (taylor and (i_resid is None or abs(i_resid) > 1e-4)))
            if bad:
                cache[k] = (None, None, None, np.inf, None)
                return cache[k]
            loss = float(np.mean((np.asarray(irf['GDP'][1:5]) - ngdp_t) ** 2))
            last_beta[0] = b
            if taylor:
                last_e0[0] = e0
            if slope is not None and np.isfinite(slope):
                last_slope[0] = slope
            cache[k] = (b, irf, resid, loss, e0)
            return cache[k]

        _ = evaluate_rho(rho0)
        res = opt.minimize_scalar(lambda r_: evaluate_rho(r_)[3],
                                  bounds=RHO_BOUNDS, method='bounded',
                                  options={'xatol': RHO_XATOL, 'maxiter': RHO_MAXITER})
        rho_star = float(res.x)
        b_s, irf_s, r_s, loss_s, e0_s = evaluate_rho(rho_star)
        if round(float(rho0), 4) not in cache:
            _ = evaluate_rho(float(rho0))
        if cache[round(float(rho0), 4)][3] < loss_s:
            rho_star = float(rho0)
            b_s, irf_s, r_s, loss_s, e0_s = cache[round(rho_star, 4)]
        return dict(beta=b_s, rho=rho_star, irf=irf_s, gdp_resid=r_s,
                    spf_loss=loss_s, eps0=e0_s)

    n = D['shock_len']
    R = {k: np.zeros(n) for k in ['beta', 'rho', 'GDP_NL', 'pi_NL', 'w_NL',
                                  'gdp_resid', 'spf_loss']}
    R['expected_GDP'] = np.zeros((n, 4))
    if taylor:
        R['eps_r_solved'] = np.zeros(n)
        R['i_NL'] = np.zeros(n)
    ss_in = deepcopy(ss)
    # per-quarter checkpoint: an interrupted estimation resumes at the first
    # unfinished quarter (state ss_in is part of the checkpoint).
    out_pkl = out_pkl or (cf['est_taylor'] if taylor else cf['est'])
    ckpt = out_pkl.replace('.pkl', '_ckpt.pkl')
    start_t = 0
    if os.path.exists(ckpt):
        try:
            saved = pickle.load(open(ckpt, 'rb'))
            R, ss_in, start_t = saved['R'], saved['ss_in'], int(saved['n_done'])
            print(f"  resuming from checkpoint at t={start_t}", flush=True)
        except Exception as e:
            print(f"  checkpoint unreadable ({e}); starting at t=0", flush=True)
    for t in range(start_t, n):
        t0 = time.time()
        o = estimate_quarter(t, float(beta_init[t]), float(rho_init[t]), ss_in,
                             e0_warm=(0.0 if taylor else None))
        irf = o['irf']
        R['beta'][t], R['rho'][t] = o['beta'], o['rho']
        R['GDP_NL'][t] = float(irf['GDP'][0])
        R['pi_NL'][t] = float(irf['pi'][0]) if 'pi' in irf.toplevel else np.nan
        R['w_NL'][t] = float(irf['w'][0]) if 'w' in irf.toplevel else np.nan
        R['expected_GDP'][t] = np.asarray(irf['GDP'][1:5])
        R['gdp_resid'][t], R['spf_loss'][t] = o['gdp_resid'], o['spf_loss']
        if taylor:
            R['eps_r_solved'][t] = o['eps0']
            R['i_NL'][t] = float(irf['i'][0])
        propagate_state(ss_in, ss, irf, MODELS[key]['het'])
        print(f"  t={t:2d}  beta={o['beta']:+.4f} rho={o['rho']:+.3f}  "
              f"GDP data {D['data_Y'][t]:+.4f} | NL {R['GDP_NL'][t]:+.4f}  "
              f"pi={R['pi_NL'][t]*400:+.2f}pp  [{time.time()-t0:.0f}s]", flush=True)
        try:
            pickle.dump({'R': R, 'ss_in': ss_in, 'n_done': t + 1}, open(ckpt, 'wb'))
        except Exception:
            pass
    tag = f"{key}_finalmodel_taylor" if taylor else f"{key}_ikc"
    pickle.dump({tag: R, 'KAPPA_W': PSS.KAPPA_W_SLOPE, 'ALPHA_TAU': ALPHA_TAU},
                open(out_pkl, 'wb'))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"Saved {out_pkl}")
    return R


def estimate_model_arma(key):
    """ARMA(1,1)-beta variant of the IKC estimation (for the appendix
    tail-splice figure): per quarter, beta_t matches GDP exactly (inner
    Newton) and (phi_t, theta_t) minimize the SPF expected-GDP loss
    (Nelder-Mead over the 2 ARMA parameters). key in {'HANK', 'RANK'}."""
    assert key in ('HANK', 'RANK')
    set_block_options(maxit=300, tol=1e-8)
    D = load_data()
    cf = MODELS[key]
    ss = deepcopy(load_ss_all()[cf['ss_name']])
    inputs = ['r_ante', 'beta_hi', 'G', 'T']
    unks, tgts = ['Y'], ['asset_mkt']
    Js, H_U_f = get_jacobians(cf['ikc'], ss, inputs, cf['jac_ikc'], D['T'],
                              unknowns=unks, targets=tgts)
    theta_cell = [0.0]
    solve = make_quarter_solver(key, cf['ikc'], ss, Js, H_U_f, D, cf['het'],
                                unknowns=unks, targets=tgts, ikc=True,
                                arma_theta_cell=theta_cell)
    try:        # warm start (phi from the AR(1) IKC estimates)
        beta_init, rho_init, _ = load_est(cf['est'])
    except Exception:
        beta_init, rho_init = (np.full(D['shock_len'], 0.05),
                               np.full(D['shock_len'], 0.85))

    n = D['shock_len']
    R = {k: np.zeros(n) for k in ['beta', 'rho', 'theta', 'GDP_NL',
                                  'gdp_resid', 'spf_loss']}
    R['expected_GDP'] = np.zeros((n, 4))
    ss_in = deepcopy(ss)
    ckpt = cf['est_arma'].replace('.pkl', '_ckpt.pkl')
    start_t = 0
    if os.path.exists(ckpt):
        try:
            saved = pickle.load(open(ckpt, 'rb'))
            R, ss_in, start_t = saved['R'], saved['ss_in'], int(saved['n_done'])
            print(f"  resuming from checkpoint at t={start_t}", flush=True)
        except Exception as e:
            print(f"  checkpoint unreadable ({e}); starting at t=0", flush=True)
    for t in range(start_t, n):
        t0 = time.time()
        target_Y = float(D['data_Y'][t])
        ngdp_t = D['ngdp_expect'][t, :].astype(float)
        last_beta, last_slope, best = [float(beta_init[t])], [None], [None]

        def loss(x):
            phi, th = float(x[0]), float(x[1])
            if abs(phi) >= 0.999 or abs(th) >= 5.0:
                return 1e6
            theta_cell[0] = th
            b, irf, resid, slope = newton_on_beta(solve, t, last_beta[0], phi,
                                                  ss_in, target_Y,
                                                  slope0=last_slope[0])
            if (resid is None or irf is None
                    or not np.all(np.isfinite(irf['GDP'][:5])) or abs(resid) > 1e-4):
                return 1e6
            last_beta[0] = b
            if slope is not None and np.isfinite(slope):
                last_slope[0] = slope
            L = float(np.mean((np.asarray(irf['GDP'][1:5]) - ngdp_t) ** 2))
            if best[0] is None or L < best[0][4]:
                best[0] = (b, phi, th, irf, L, resid)
            return L

        # clip the AR(1) warm start into the interior: loss() rejects
        # |phi| >= 0.999, and quarters where the AR(1) fit hit its rho
        # boundary (e.g. rho=0.999) would otherwise start NM on an all-1e6
        # simplex, terminate immediately, and leave best[0] = None.
        phi0 = float(np.clip(rho_init[t], -0.99, 0.99))
        starts = [(phi0, 0.0), (0.9, 0.0), (0.85, -0.3), (0.5, 0.0)]
        if t > 0 and R['rho'][t-1] != 0.0:
            starts.insert(1, (float(np.clip(R['rho'][t-1], -0.99, 0.99)),
                              float(R['theta'][t-1])))
        for x0 in starts:
            opt.minimize(loss, x0=np.array(x0), method='Nelder-Mead',
                         options={'xatol': 1e-3, 'fatol': 1e-10, 'maxiter': 60})
            if best[0] is not None:
                break
            print(f"  t={t:2d}  no valid point from start {x0}; retrying",
                  flush=True)
        if best[0] is None:
            raise RuntimeError(f"ARMA estimation failed at t={t}: Newton on "
                               f"beta failed for every (phi, theta) tried")
        b_s, phi_s, th_s, irf_s, loss_s, resid_s = best[0]
        R['beta'][t], R['rho'][t], R['theta'][t] = b_s, phi_s, th_s
        R['GDP_NL'][t] = float(irf_s['GDP'][0])
        R['expected_GDP'][t] = np.asarray(irf_s['GDP'][1:5])
        R['gdp_resid'][t], R['spf_loss'][t] = resid_s, loss_s
        propagate_state(ss_in, ss, irf_s, cf['het'])
        print(f"  t={t:2d}  beta={b_s:+.4f} phi={phi_s:+.3f} theta={th_s:+.3f} "
              f"SPF={loss_s:.2e}  [{time.time()-t0:.0f}s]", flush=True)
        try:
            pickle.dump({'R': R, 'ss_in': ss_in, 'n_done': t + 1}, open(ckpt, 'wb'))
        except Exception:
            pass
    pickle.dump({f'{key}_ikc_arma': R, 'ALPHA_TAU': ALPHA_TAU},
                open(cf['est_arma'], 'wb'))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"Saved {cf['est_arma']}")
    return R


# ---------------------------------------------------------------------------
# Replay products  [stages: replay / causal / taylor-nogt]
# ---------------------------------------------------------------------------
def stage_replay(arma=False):
    """Rebuild the 4-model path replays from the stored per-model
    (beta_t, rho_t) estimates. RANK/RANK_CC rows use the hankwbeta_ss wage PC
    (wpc_wexo)."""
    set_block_options(maxit=600, tol=1e-8)
    D = load_data()
    ss_all = load_ss_all()
    meta = {'HMAX': HMAX, 'T': D['T'], 'shock_len': D['shock_len'],
            'ALPHA_TAU': ALPHA_TAU, 'RHO_R': RHO_R, 'RHO_YBAR': RHO_YBAR,
            'SAVE_PATHS': SAVE_PATHS}

    def one(key, use_wexo, arma_pkl=None):
        cf = MODELS[key]
        est_path = cf['est_arma'] if arma_pkl else cf['est']
        beta, rho, R = load_est(est_path)
        thetas = None
        if arma_pkl:   # ARMA(1,1): R carries the MA parameter per quarter
            for tk in ('theta', 'q1', 'p2'):
                if tk in R:
                    thetas = np.asarray(R[tk], float)
                    break
        out = run_trajectory(key, ss_all[cf['ss_name']], D, beta, rho,
                             use_wexo=use_wexo, arma_thetas=thetas,
                             record_paths=True, label=key)
        out['GDP_NL'], out['pi_NL'], out['w_NL'] = out.pop('GDP'), out.pop('pi'), out.pop('w')
        return out

    out_pkl = REPLAY_ARMA_HWB_PKL if arma else REPLAY_HWB_PKL
    results = {}
    for key in ['RANK', 'RANK_CC', 'HANK', 'HANK_CC']:
        use_wexo = key.startswith('RANK')
        print(f"=== replay {key}{' (hankwbeta_ss)' if use_wexo else ''} -> {out_pkl}",
              flush=True)
        results[key] = one(key, use_wexo, arma_pkl=arma)
    pickle.dump({'results': results, 'meta': meta}, open(out_pkl, 'wb'))
    print(f"Saved {out_pkl}")


def stage_ablation():
    """Cumulative shock subsets (beta | beta+r | beta+r+G,T | all(+Ybar) | noGT)
    for HANK and HANK CC -> ABLATION_PKL (paper Fig 'Simulation Shock
    Decomposition'; the real-rate rows of the Taylor-rule comparison also use
    'all' - 'nogt')."""
    set_block_options(maxit=3000, tol=1e-8)
    D = load_data()
    ss_all = load_ss_all()
    SUBSETS = {'b':    dict(gt_on=False, r_on=False, ybar_on=False),
               'br':   dict(gt_on=False, r_on=True,  ybar_on=False),
               'brgt': dict(gt_on=True,  r_on=True,  ybar_on=False),
               'all':  dict(gt_on=True,  r_on=True,  ybar_on=True),
               'nogt': dict(gt_on=False, r_on=True,  ybar_on=True)}
    results = {}
    for key in ['HANK', 'HANK_CC']:
        cf = MODELS[key]
        beta, rho, _ = load_est(cf['est'])
        out = {'beta': beta.copy(), 'rho': rho.copy()}
        subsets = ['b', 'br', 'brgt'] + (['all', 'nogt'] if cf['is_cc'] else [])
        for sub in subsets:
            print(f"=== ablation {key} [{sub}]", flush=True)
            r = run_trajectory(key, ss_all[cf['ss_name']], D, beta, rho,
                               label=f"{key}:{sub}",
                               skip_fail=True, **SUBSETS[sub])
            out[sub] = {k: r[k] for k in ['GDP', 'pi', 'w', 'Pi']}
        if not cf['is_cc']:      # conv: no Y_bar input -> all == brgt, nogt == br
            out['all'], out['nogt'] = out['brgt'], out['br']
        results[key] = out
    pickle.dump(results, open(ABLATION_PKL, 'wb'))
    print(f"Saved {ABLATION_PKL}")


def stage_causal():
    """Fiscal effect under the real-rate rule: FULL and noGT as two independent
    state-propagated trajectories per model (paper's counterfactual). RANK rows
    use the hankwbeta_ss wage-PC convention (paper baseline)."""
    set_block_options(maxit=2000, tol=1e-6)
    D = load_data()
    ss_all = load_ss_all()
    for key, out_pkl in CAUSAL_PKL.items():
        cf = MODELS[key]
        use_wexo = key.startswith('RANK')
        beta, rho, _ = load_est(cf['est'])
        print(f"=== causal {key}{' (hankwbeta_ss)' if use_wexo else ''}", flush=True)
        full = run_trajectory(key, ss_all[cf['ss_name']], D, beta, rho,
                              use_wexo=use_wexo, label=f"{key}:FULL",
                              maxit=300, tol=1e-7)
        nogt = run_trajectory(key, ss_all[cf['ss_name']], D, beta, rho,
                              use_wexo=use_wexo, gt_on=False, label=f"{key}:noGT",
                              maxit=300, tol=1e-7)
        out = {'beta': beta, 'rho': rho, 'name': key}
        for k in ['GDP', 'pi', 'w', 'N', 'mc', 'Pi', 'Pi_int', 'r_ante']:
            out[f'{k}_full'], out[f'{k}_noGT'] = full[k], nogt[k]
        pickle.dump(out, open(out_pkl, 'wb'))
        print(f"Saved {out_pkl}")


def stage_taylor_nogt():
    """No-fiscal counterfactual for the Taylor rate-matched estimates: replay
    at the STORED (beta, rho, eps_r_solved) with G = T = 0."""
    set_block_options(maxit=600, tol=1e-8)
    D = load_data()
    ss_all = load_ss_all()
    res = {}
    if os.path.exists(TAYLOR_NOGT_PKL):
        try:
            res = pickle.load(open(TAYLOR_NOGT_PKL, 'rb'))
        except Exception:
            res = {}
    for key in TAYLOR_MODELS:
        cf = MODELS[key]
        if not os.path.exists(cf['est_taylor']):
            print(f"  taylor-noGT {key}: no estimate pkl yet, skipped", flush=True)
            continue
        d = pickle.load(open(cf['est_taylor'], 'rb'))[f'{key}_finalmodel_taylor']
        beta = np.asarray(d['beta'], float)
        rho = np.asarray(d['rho'], float)
        eps0 = np.asarray(d['eps_r_solved'], float)
        print(f"=== taylor-noGT {key}", flush=True)
        r = run_trajectory(key, ss_all[cf['ss_name']], D, beta, rho,
                           taylor_eps=eps0, gt_on=False,
                           label=f"{key}:tay-noGT")
        res[key] = {k: r[k] for k in ('GDP', 'pi', 'w')}
        pickle.dump(res, open(TAYLOR_NOGT_PKL, 'wb'))
    print(f"Saved {TAYLOR_NOGT_PKL}")


# ---------------------------------------------------------------------------
# Tailsplice machinery (supply block re-solve; paper Fig "gdp_expect")
# ---------------------------------------------------------------------------
@sj.simple
def gdp_target(Y, Delta_p, GDP):
    """GDP (=Y/Delta_p) is the exogenous input; gross output Y becomes an
    unknown pinned by Y/Delta_p = GDP."""
    res_gdp = Y / Delta_p - GDP
    return res_gdp


supply_conv_gdp = sj.create_model([mc, labor, after_tax_income, pc, wpc, gdp_target],
                                  name='supply_conv_gdp')
supply_cc_gdp = sj.create_model([util, mc_convex, labor_convex, after_tax_income,
                                 pc, wpc, gdp_target], name='supply_cc_gdp')
SUP_UNKS, SUP_TARGETS = ['w', 'pi', 'Y'], ['res_wpc', 'res_pc', 'res_gdp']


class SupplySolver:
    """Self-contained supply block {mc(-convex), labor, taxes, price PC, wage PC}
    solved for {w, pi, Y} given anticipated paths of {GDP, C, Tau, beta_hi,
    r_ante (, Y_bar)}. Used to re-solve inflation under counterfactual GDP
    expectations (the tail splice)."""

    state_vars = ('w', 'Delta_p')

    def __init__(self, name, ss_ref_scalars, T):
        self.is_cc = name.endswith('_CC')
        self.model = supply_cc_gdp if self.is_cc else supply_conv_gdp
        self.T = T
        scal = dict(ss_ref_scalars)
        if 'GDP' not in scal:
            scal['GDP'] = float(scal['Y']) / float(scal.get('Delta_p', 1.0))
        scal['res_gdp'] = 0.0
        self.ss = sj.classes.SteadyStateDict(scal)
        self.inputs_exog = ['GDP', 'C', 'Tau', 'beta_hi', 'r_ante'] + \
                           (['Y_bar'] if self.is_cc else [])
        Js = quiet(self.model.partial_jacobians, self.ss,
                   inputs=set(self.inputs_exog) | set(SUP_UNKS),
                   outputs=set(SUP_TARGETS), T=T)
        H_U = quiet(self.model.jacobian, self.ss, SUP_UNKS, SUP_TARGETS, T=T, Js=Js)
        self.Js, self.H_U_factored = Js, FactoredJacobianDict(H_U, T)

    def ss_initial(self, ss_in_scalars):
        s = deepcopy(self.ss)
        # Only these variables enter the supply block with a lag. In
        # particular, do not let the full-model initial state overwrite the
        # wage-PC steady-state discount embedded in self.ss.
        for k in self.state_vars:
            if k in ss_in_scalars and k in s:
                s[k] = float(ss_in_scalars[k])
        return s

    def solve(self, inputs, ss_initial):
        irf = quiet(self.model.solve_impulse_nonlinear, self.ss,
                    ss_initial=ss_initial, unknowns=SUP_UNKS,
                    targets=SUP_TARGETS, inputs=inputs,
                    H_U_factored=self.H_U_factored, Js=self.Js,
                    maxit=300, tol=1e-11, verbose=False)
        return irf

    def solve_pi(self, inputs, ss_in_scalars):
        irf = self.solve(inputs, self.ss_initial(ss_in_scalars))
        return np.asarray(irf['pi'], float)

    def propagate_state(self, ss_initial, irf):
        """Advance the supply block's predetermined state by one quarter."""
        for k in self.state_vars:
            if k in irf.toplevel:
                ss_initial[k] = float(self.ss[k]) + float(irf[k][0])


def _supply_inputs(res, name, t):
    R = res[name]
    p = R['paths'][t]
    d = {'GDP': p['GDP'], 'C': p['C'], 'Tau': p['Tau'],
         'beta_hi': R['beta_hi_path'][t], 'r_ante': R['r_ante_path'][t]}
    if name.endswith('_CC'):
        d['Y_bar'] = R['Y_bar_path'][t]
    return d


def fig_tailsplice(replay_pkl=REPLAY_HWB_PKL,
                   out_pdf=f'{FIGDIR}/finalmodel_real_tailsplice_pi_hankref_hankwbeta_ss.pdf'):
    """RANK-vs-HANK inflation decomposition figure (paper Fig 'gdp_expect'):
    col 1 GDP expectations (2020Q2 top / 2021Q2 bottom), col 2 simulated pi,
    col 3 state-propagated pi with RANK given HANK's anticipated {GDP, C} for
    h>=5 in every quarter. RANK keeps its own beta shock path everywhere.
    The supply solver carries HANK's wage-PC steady-state discount scalars,
    matching the wpc_wexo replay."""
    style = dict(PLT_STYLE)
    style.update({"font.size": 13, "axes.titlesize": 16.5, "axes.labelsize": 17,
                  "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                  "xtick.major.size": 5, "ytick.major.size": 5,
                  "xtick.major.width": 1.1, "ytick.major.width": 1.1,
                  "axes.linewidth": 1.1})
    plt.rcParams.update(style)
    COL = {'HANK': '#E69F00', 'RANK': '#56B4E9',
           'HANK_CC': '#E69F00', 'RANK_CC': '#56B4E9'}
    LAB = {m: m.replace('_CC', ' CC') for m in COL}

    rp = pickle.load(open(replay_pkl, 'rb'))
    res, meta = rp['results'], rp['meta']
    T, shock_len = meta['T'], meta['shock_len']
    D = load_data()
    spf, data_Y = D['ngdp_expect'], D['data_Y']

    PAIRS = [('Conventional', ['RANK', 'HANK'], 'HANK'),
             ('Capacity Constraint', ['RANK_CC', 'HANK_CC'], 'HANK_CC')]
    print("building supply solvers...", flush=True)
    # hankwbeta_ss: solver carries HANK's wage-PC discount scalars for RANK
    solvers = {}
    for _, models, ref in PAIRS:
        ss_ref_ref = res[ref]['ss_ref']
        for m in models:
            scal = dict(res[m]['ss_ref'])
            if m != ref:
                for k in ('omega', 'dbeta', 'beta_hi'):
                    scal[k] = ss_ref_ref[k]
            solvers[m] = SupplySolver(m, scal, T)

    def pi_cf_trajectory(m, ref, swap_tail=False):
        """Sequential supply-block counterfactual for model m.

        Demand-side paths remain those from m's replay. At every calendar
        quarter, HANK's anticipated GDP and C paths replace m's paths from
        horizon 5 onward, and the resulting {w, Delta_p} state is carried into
        the next calendar quarter. The beta_hi input always remains m's own.
        """
        pi = np.full(shock_len, np.nan)
        ss_cf = solvers[m].ss_initial(res[m]['ss_in'][0])
        for t in range(shock_len):
            inp = {k: np.asarray(v, float).copy()
                   for k, v in _supply_inputs(res, m, t).items()}
            rin = _supply_inputs(res, ref, t)
            if swap_tail:
                inp['GDP'][5:] = np.asarray(rin['GDP'], float)[5:]
                inp['C'][5:] = np.asarray(rin['C'], float)[5:]
            try:
                irf = solvers[m].solve(inp, ss_cf)
            except Exception as e:
                print(f"  [{m}] t={t} tail: no equilibrium "
                      f"({type(e).__name__}) -> remaining path is NaN",
                      flush=True)
                break
            pi[t] = float(irf['pi'][0])
            solvers[m].propagate_state(ss_cf, irf)
        return pi

    results = {}
    for pair_name, models, ref in PAIRS:
        out = {m: {'act': np.asarray(res[m]['pi_NL'], float),
                   'tail': np.full(shock_len, np.nan)} for m in models}
        for m in models:
            if m == ref:
                out[m]['tail'] = out[m]['act'].copy()
                continue
            out[m]['tail'] = pi_cf_trajectory(m, ref, swap_tail=True)
        results[pair_name] = (models, ref, out)

    quarters = pd.period_range('2020Q2', periods=shock_len, freq='Q')
    xt = np.arange(shock_len)
    tick = np.arange(0, shock_len, 3)
    tlab = [str(quarters[i]) for i in tick]
    PANELS = [('act', 'Actual (Own Expectations)'),
              ('tail', 'HANK h≥5 Tail Expectations')]
    NPI = len(PANELS)
    fig = plt.figure(figsize=(4.9 * (1 + NPI) + 0.9, 9.2))
    # col 0 widened (was 1.0) to eat the white space up to the divider line;
    # gap column shrunk to match so the pi columns and right-of-line space keep
    # their sizes.
    gs = fig.add_gridspec(2, 2 + NPI, width_ratios=[1.07, 0.11] + [1.0] * NPI)
    axes_rows, pi_base = [], None
    for r in (0, 1):
        row = [fig.add_subplot(gs[r, 0])]
        for c in range(2, 2 + NPI):
            ax = fig.add_subplot(gs[r, c]) if pi_base is None \
                else fig.add_subplot(gs[r, c], sharey=pi_base)
            if pi_base is None:
                pi_base = ax
            row.append(ax)
        axes_rows.append(row)
    axes = np.array(axes_rows)
    TSTAR_ROW = [0, 4]     # 2020Q2 top, 2021Q2 bottom

    for row, (pair_name, models, ref) in enumerate(PAIRS):
        models, ref, out = results[pair_name]
        ax = axes[row, 0]
        tstar = TSTAR_ROW[row]
        for m in models:
            fore = np.asarray(res[m]['GDP_fore'], float)[tstar] * 100.0
            ax.plot(np.arange(HMAX + 1), fore, color=COL[m], lw=2.6,
                    marker='o', ms=4.5, label=LAB[m])
        ax.plot(np.arange(5), np.concatenate([[data_Y[tstar] * 100.0], spf[tstar] * 100.0]),
                color='black', lw=2.2, marker='s', ms=4.5, label='SPF (4-q)')
        ax.axvspan(4.5, HMAX, color='0.85', alpha=0.55, zorder=0)
        ax.set_xlabel('Anticipation Horizon (Quarters)')
        ax.set_ylabel('Expected GDP (% Dev. from SS)')
        ax.set_title(f'GDP Expectations ({quarters[tstar]})')
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        if row == 0:
            ax.legend(frameon=False, fontsize=14.5, loc='upper left')
        for c, (key_, title) in enumerate(PANELS, start=1):
            ax = axes[row, c]
            for m in models:
                ax.plot(xt, out[m][key_] * 400.0, color=COL[m], lw=2.6,
                        marker='o', ms=4, label=LAB[m])
            ax.set_xticks(tick)
            ax.set_xticklabels(tlab, rotation=45, ha='right')
            ax.axhline(0, color='black', lw=0.5, alpha=0.4)
            if row == 0:
                ax.set_title(title)
            if c == 1:
                ax.set_ylabel('Inflation (Annualized p.p.)')
            else:
                ax.tick_params(labelleft=False)
    plt.tight_layout()
    from matplotlib.lines import Line2D
    gap_l, gap_r = axes[0, 0].get_position().x1, axes[0, 1].get_position().x0
    # line sits just off col 0's (now wider) right edge; label centered in the
    # remaining gap to its right.
    x_rule, x_lab = gap_l + 0.05 * (gap_r - gap_l), gap_l + 0.52 * (gap_r - gap_l)
    y_lo = min(axes[1, 0].get_position().y0, axes[1, 1].get_position().y0)
    y_hi = axes[0, 0].get_position().y1
    fig.add_artist(Line2D([x_rule, x_rule], [y_lo - 0.04, y_hi + 0.02],
                          transform=fig.transFigure, color='0.65', lw=1.0))
    for row in range(2):
        pos = axes[row, 1].get_position()
        fig.text(x_lab, 0.5 * (pos.y0 + pos.y1), PAIRS[row][0], rotation=90,
                 va='center', ha='center', fontsize=18, fontweight='bold')
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ---------------------------------------------------------------------------
# Figures (default stage)
# ---------------------------------------------------------------------------
COL4 = {'RANK': '#56B4E9', 'RANK_CC': '#56B4E9', 'HANK': '#E69F00', 'HANK_CC': '#E69F00'}
STY4 = {m: ('--' if m.endswith('_CC') else '-') for m in COL4}
LAB4 = {m: m.replace('_CC', ' CC') for m in COL4}
ORDER4 = ['RANK', 'RANK_CC', 'HANK', 'HANK_CC']


def _qaxis(n, step=2):
    q = pd.period_range('2020Q2', periods=n, freq='Q')
    tick = np.arange(0, n, step)
    return np.arange(n), tick, [str(q[i]) for i in tick]


def fig_4model_paths(out_sfx='_hankwbeta_ss', src=REPLAY_HWB_PKL):
    """Simulated GDP / inflation / real-wage paths, 4 models + data."""
    plt.rcParams.update(PLT_STYLE)
    res = pickle.load(open(src, 'rb'))['results']
    D = load_data()
    n = len(D['data_Y'])
    xt, tick, tlab = _qaxis(n)
    SPECS = [
        ('GDP', 'GDP_NL', lambda v, m: np.asarray(v, float) * 100.0,
         'Real GDP (% Dev. from SS)', D['data_Y'] * 100.0),
        ('pi', 'pi_NL', lambda v, m: np.asarray(v, float) * 400.0,
         'Inflation (Annualized p.p.)', D['data_core_pi'] * 400.0),   # core PCE
        ('w', 'w_NL', lambda v, m: np.asarray(v, float) / float(res[m]['ss_ref']['w']) * 100.0,
         'Real Wage (% Dev. from SS)', None),
    ]
    os.makedirs(FIGDIR, exist_ok=True)
    for tag, key, fn, ylab, data in SPECS:
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        handles = {}
        for m in ORDER4:
            v = fn(res[m][key], m)
            emph = m.startswith('HANK')
            (h,) = ax.plot(xt[:len(v)], v, color=COL4[m], ls=STY4[m],
                           lw=3.0 if emph else 2.8, alpha=1.0 if emph else 0.8,
                           marker='o', ms=5, label=LAB4[m], zorder=4 if emph else 2)
            handles[m] = h
        h_data = None
        if data is not None:
            (h_data,) = ax.plot(xt[:len(data)], data, color='black', lw=2.6,
                                marker='s', ms=5.5, label='Data', zorder=5)
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_xticks(tick)
        ax.set_xticklabels(tlab, rotation=45, ha='right')
        ax.set_ylabel(ylab)
        col1 = [handles[m] for m in ['RANK', 'HANK']]
        col2 = [handles[m] for m in ['RANK_CC', 'HANK_CC']]
        if h_data is not None:
            col1.append(h_data)
        hs = col1 + col2
        if tag != 'w':      # 'w' panel omits the legend (shares the GDP/pi one)
            ax.legend(handles=hs, labels=[h.get_label() for h in hs], ncol=2,
                      frameon=False, fontsize=17, handlelength=2.6, columnspacing=1.2)
        plt.tight_layout()
        out = f'{FIGDIR}/finalmodel_real_4model_{tag}{out_sfx}.pdf'
        plt.savefig(out)
        plt.close(fig)
        print(f'Wrote {out}')


def fig_shock_series():
    """Panel (d) of the simulation figure: impact values of all shocks."""
    plt.rcParams.update(PLT_STYLE)
    D = load_data()
    sh = D['shocks']
    ss_cc = load_ss_all()['ss_cc']
    beta_hank, _, _ = load_est(MODELS['HANK']['est'])
    beta_rank, _, _ = load_est(MODELS['RANK']['est'])
    n = len(sh['G'])
    xt, tick, tlab = _qaxis(n)
    SERIES = [
        (np.asarray(sh['r_ante'], float) * 400.0, r'$r$', '#0072B2', '-', 'o'),
        (np.asarray(sh['G'], float) * 100.0, r'$G$', '#009E73', '-', '^'),
        (np.asarray(sh['T'], float) * 100.0, r'$T$', '#D55E00', '-', 'v'),
        (np.asarray(sh['Y_bar'], float) / float(ss_cc['Y_bar']) * 100.0,
         r'$\bar{Y}$', '#CC79A7', '-', 's'),
        # beta recolored purple: orange (#E69F00) collides with HANK in the
        # companion 4-model figures; purple is distinct from those sky-blue/
        # orange/black series.
        (beta_hank * 100.0, r'$\beta$ (HANK)', '#9467BD', '-', 'D'),
        (beta_rank * 100.0, r'$\beta$ (RANK)', '#9467BD', '--', 'D'),
    ]
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for y, lab, col, ls, mk in SERIES:
        ax.plot(xt[:len(y)], y, color=col, ls=ls, marker=mk, lw=2.8, ms=5, label=lab)
    ax.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax.set_xticks(tick)
    ax.set_xticklabels(tlab, rotation=45, ha='right')
    ax.set_ylabel('Shock size')
    ax.legend(frameon=False, fontsize=17, handlelength=2.6, ncol=2, columnspacing=1.2)
    plt.tight_layout()
    out = f'{FIGDIR}/finalmodel_shock_series.pdf'
    plt.savefig(out)
    plt.close(fig)
    print(f'Wrote {out}')


def _fiscal_effects():
    """FULL - noGT per model from the causal pickles (paper conventions)."""
    ss_w = pickle.load(open(REPLAY_HWB_PKL, 'rb'))['results']
    Dw = {}
    for m in ORDER4:
        d = pickle.load(open(CAUSAL_PKL[m], 'rb'))
        Dw[m] = dict(
            pi=(np.asarray(d['pi_full'], float) - np.asarray(d['pi_noGT'], float)) * 400.0,
            GDP=(np.asarray(d['GDP_full'], float) - np.asarray(d['GDP_noGT'], float)) * 100.0,
            w=(np.asarray(d['w_full'], float) - np.asarray(d['w_noGT'], float))
              / float(ss_w[m]['ss_ref']['w']) * 100.0)
    return Dw


def fig_fiscal_effect_4model(out_sfx='_hankwbeta_ss'):
    plt.rcParams.update(PLT_STYLE)
    plt.rcParams.update({"axes.labelsize": 26, "xtick.labelsize": 20,
                         "ytick.labelsize": 20})   # much larger labels & ticks
    Dw = _fiscal_effects()
    n = len(Dw['RANK']['pi'])
    xt, tick, tlab = _qaxis(n)
    SPECS = [('pi', 'Annualized p.p.'),
             ('w', '% Dev. from SS'),
             ('GDP', '% Dev. from SS')]
    os.makedirs(FIGDIR, exist_ok=True)
    for tag, ylab in SPECS:
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        handles = {}
        for m in ORDER4:
            emph = m.startswith('HANK')
            (h,) = ax.plot(xt, Dw[m][tag], color=COL4[m], ls=STY4[m],
                           lw=3.4 if emph else 3.2, alpha=1.0 if emph else 0.8,
                           marker='o', ms=5, label=LAB4[m], zorder=4 if emph else 2)
            handles[m] = h
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_xticks(tick)
        ax.set_xticklabels(tlab, rotation=45, ha='right')
        ax.set_ylabel(ylab)
        if tag == 'GDP':      # pi and w panels share the GDP panel's legend
            hs = [handles[m] for m in ['RANK', 'HANK', 'RANK_CC', 'HANK_CC']]
            ax.legend(handles=hs, labels=[h.get_label() for h in hs], ncol=1,
                      frameon=False, fontsize=22, handlelength=2.6, columnspacing=1.2)
        plt.tight_layout()
        out = f'{FIGDIR}/fiscal_effect_4model_{tag}{out_sfx}.pdf'
        plt.savefig(out)
        plt.close(fig)
        print(f'Wrote {out}')


def fig_real_vs_taylor():
    """Fiscal effect: real-rate rule vs (rate-matched, static) Taylor rule,
    HANK and HANK CC."""
    plt.rcParams.update(PLT_STYLE)
    plt.rcParams.update({"axes.labelsize": 26, "xtick.labelsize": 20,
                         "ytick.labelsize": 20})   # much larger labels & ticks
    abl = pickle.load(open(ABLATION_PKL, 'rb'))
    ss_w = pickle.load(open(REPLAY_HWB_PKL, 'rb'))['results']
    tay_nogt = pickle.load(open(TAYLOR_NOGT_PKL, 'rb'))
    tay_full = {m: pickle.load(open(MODELS[m]['est_taylor'], 'rb'))[f'{m}_finalmodel_taylor']
                for m in ['HANK', 'HANK_CC']}
    KEYMAP = {'GDP': ('GDP', 'GDP_NL'), 'pi': ('pi', 'pi_NL'), 'w': ('w', 'w_NL')}

    def effect(m, rule, key):
        krep, kest = KEYMAP[key]
        if rule == 'real':
            nogt = abl[m]['nogt'] if 'nogt' in abl[m] else abl[m]['br']
            return np.asarray(abl[m]['all'][krep], float) - np.asarray(nogt[krep], float)
        return np.asarray(tay_full[m][kest], float) - np.asarray(tay_nogt[m][krep], float)

    def scaled(m, rule, key, sc):
        e = effect(m, rule, key) * sc
        if key == 'w':
            e = e / float(ss_w[m]['ss_ref']['w'])
        return e

    VARS = [('GDP', 100.0, '% Dev. from SS'),
            ('pi', 400.0, 'Annualized p.p.'),
            ('w', 100.0, '% Dev. from SS')]
    SER = [('HANK', 'real', '#E69F00', '-', 'o', 'HANK, real-rate'),
           ('HANK_CC', 'real', '#E69F00', '--', 's', 'HANK CC, real-rate'),
           ('HANK', 'taylor', '#0072B2', '-', 'o', 'HANK, Taylor'),
           ('HANK_CC', 'taylor', '#0072B2', '--', 's', 'HANK CC, Taylor')]
    n = len(abl['HANK']['all']['GDP'])
    x, tick, tlab = _qaxis(n)
    os.makedirs(FIGDIR, exist_ok=True)
    for key, sc, ylab in VARS:
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        for m, rule, col, ls, mk, lab in SER:
            ax.plot(x, scaled(m, rule, key, sc), color=col, ls=ls, lw=3.4,
                    marker=mk, ms=6, label=lab)
        ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_ylabel(ylab)
        ax.set_xticks(tick)
        ax.set_xticklabels(tlab, rotation=45, ha='right')
        if key == 'GDP':      # pi and w panels share the GDP legend
            ax.legend(frameon=False, fontsize=18, loc='best')
        plt.tight_layout()
        out = f'{FIGDIR}/hank_real_vs_taylor_gt_effect_{key}.pdf'
        plt.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def fig_ablation_singles(models=('HANK', 'HANK_CC')):
    """Per-panel ablation figures (cumulative shock sets)."""
    plt.rcParams.update(PLT_STYLE)
    RES = pickle.load(open(ABLATION_PKL, 'rb'))
    SUBSETS = [('b', r'$\beta$ only', '#999999', ':', 'o'),
               ('br', r'$\beta$, $r$', '#56B4E9', '-.', '^'),
               ('brgt', r'$\beta$, $r$, $G$, $T$', '#0072B2', '--', 's'),
               ('all', r'all ($+\bar Y$)', '#D55E00', '-', 'D')]
    PANELS = [('GDP', 100.0, 'Real GDP (% dev from SS)'),
              ('pi', 400.0, 'Inflation (annualized pp)'),
              ('w', 100.0, 'Real wage (% dev from SS)')]
    n = len(next(iter(RES.values()))['b']['GDP'])
    xt, tick, tlab = _qaxis(n)
    # shared y-limits: CC and non-CC versions of each variable get the same axis
    ylims = {}
    for fam in {m.replace('_CC', '') for m in models}:
        for key, scale, _ in PANELS:
            vals = []
            for m in (fam, fam + '_CC'):
                for sub, *_ in SUBSETS:
                    if sub == 'all' and not m.endswith('_CC'):
                        continue
                    vals.append(np.asarray(RES[m][sub][key], float) * scale)
            arr = np.concatenate(vals)
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            pad = 0.05 * (hi - lo)
            ylims[(fam, key)] = (lo - pad, hi + pad)
    os.makedirs(FIGDIR, exist_ok=True)
    for m in models:
        is_cc = m.endswith('_CC')
        for key, scale, ylab in PANELS:
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            for sub, lab, colr, ls, mk in SUBSETS:
                if sub == 'all' and not is_cc:
                    continue
                if sub == 'brgt' and not is_cc:
                    lab = r'$\beta$, $r$, $G$, $T$ (= all)'
                ax.plot(xt, np.asarray(RES[m][sub][key], float) * scale,
                        color=colr, ls=ls, lw=3.0, marker=mk, ms=6.5, label=lab)
            ax.set_ylabel(ylab)
            ax.set_ylim(*ylims[(m.replace('_CC', ''), key)])
            ax.set_xticks(tick)
            ax.set_xticklabels(tlab, rotation=45, ha='right')
            ax.axhline(0, color='black', lw=0.7, alpha=0.4)
            if key == 'GDP':      # w and pi panels share the GDP legend
                ax.legend(frameon=False, fontsize=16)
            plt.tight_layout()
            out = f'{FIGDIR}/finalmodel_real_ablation_{m}_{key}.pdf'
            plt.savefig(out)
            plt.close(fig)
            print(f'Wrote {out}')


def fig_profits_hankonly(src=REPLAY_HWB_PKL, out_sfx='_hankonly'):
    """Total / intermediary / retailer profits, HANK conventional vs CC
    ('greedflation' figure). Profits are a reporting block: evaluated from the
    replay's stored supply-block paths at each quarter's impact values."""
    plt.rcParams.update(PLT_STYLE)
    plt.rcParams.update({"axes.labelsize": 26, "xtick.labelsize": 20,
                         "ytick.labelsize": 20})   # much larger labels & ticks
    MODELS_P = ['HANK', 'HANK_CC']
    LABP = {m: ('Capacity Constraint' if m.endswith('_CC') else 'Conventional')
            for m in MODELS_P}
    COLP = {m: ('#D55E00' if m.endswith('_CC') else '#0072B2') for m in MODELS_P}
    LSP = {'HANK': '-', 'HANK_CC': '--'}
    MKP = {'HANK': 'o', 'HANK_CC': 's'}
    res = pickle.load(open(src, 'rb'))['results']

    def profit_devs(r):
        s = r['ss_ref']
        Yss, wss, Nss, dpss, mcss = (float(s[k]) for k in ['Y', 'w', 'N', 'Delta_p', 'mc'])
        Pi_ss = Yss / dpss - wss * Nss
        Piint_ss = mcss * Yss - wss * Nss
        paths = r['paths']
        n = len(paths)
        Pi, Piint = np.full(n, np.nan), np.full(n, np.nan)
        for t in range(n):
            p = paths[t]
            if p is None or 'Y' not in p:
                continue
            Y = Yss + float(np.asarray(p['Y'])[0])
            w = wss + float(np.asarray(p['w'])[0])
            N = Nss + float(np.asarray(p['N'])[0])
            dp = dpss + float(np.asarray(p['Delta_p'])[0])
            mc_ = mcss + float(np.asarray(p['mc'])[0])
            Pi[t] = (Y / dp - w * N) - Pi_ss
            Piint[t] = (mc_ * Y - w * N) - Piint_ss
        return {'Pi': Pi, 'Pi_int': Piint, 'Pi_ret': Pi - Piint,
                'ss': {'Pi': Pi_ss, 'Pi_int': Piint_ss, 'Pi_ret': Pi_ss - Piint_ss}}

    Dp = {m: profit_devs(res[m]) for m in MODELS_P}
    n = len(res['HANK']['paths'])
    xt, tick, tlab = _qaxis(n)
    SPECS = [('total', 'Pi', 'Total profits'),
             ('int', 'Pi_int', 'Intermediary profits'),
             ('ret', 'Pi_ret', 'Retailer profits')]
    os.makedirs(FIGDIR, exist_ok=True)
    for tag, key, base in SPECS:
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        any_rel_total = False
        for m in MODELS_P:
            s = Dp[m]['ss']
            dev = np.asarray(Dp[m][key], float)
            lab = LABP[m]
            if abs(s[key]) > 1e-6:
                y = dev / s[key] * 100.0
            else:      # conv Pi_int_ss = 0 -> % of SS total profits
                y = dev / s['Pi'] * 100.0
                lab += r' (% of SS total $\Pi$)'
                any_rel_total = True
            ax.plot(xt, y, color=COLP[m], ls=LSP[m], lw=4.0, marker=MKP[m],
                    ms=6.5, label=lab)
        ax.set_ylabel('% Dev. from SS')
        ax.set_xticks(tick)
        ax.set_xticklabels(tlab, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        if tag == 'total':     # legend on the total panel only
            ax.legend(frameon=False, fontsize=20 if any_rel_total else 26)
        plt.tight_layout()
        out = f'{FIGDIR}/finalmodel_profits_{tag}{out_sfx}.pdf'
        plt.savefig(out)
        plt.close(fig)
        print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# Quantitative claims used in the paper text  [stage: numbers]
# ---------------------------------------------------------------------------
def stage_numbers():
    """Print every headline number quoted in the paper, computed from the same
    cached pickles the figures use."""
    D = load_data()
    n = D['shock_len']
    q = pd.period_range('2020Q2', periods=n, freq='Q')
    i_21q2 = list(q).index(pd.Period('2021Q2'))
    eff = _fiscal_effects()
    res = pickle.load(open(REPLAY_HWB_PKL, 'rb'))['results']
    data_pi = D['data_core_pi']

    print("=== Fiscal effect (real-rate rule, FULL - noGT) ===")
    for m in ['HANK_CC', 'HANK', 'RANK_CC', 'RANK']:
        pi_e = eff[m]['pi']
        w_e = eff[m]['w']
        print(f"  {m:8s}: pi 2021Q2 = {pi_e[i_21q2]:+.2f}pp ann. "
              f"(peak {np.nanmax(pi_e):+.2f} at {q[int(np.nanargmax(pi_e))]}) | "
              f"cum price growth = {np.nansum(pi_e)/4:.2f}pp | "
              f"real w max = {np.nanmax(w_e):+.2f}% at {q[int(np.nanargmax(w_e))]} | "
              f"real w min = {np.nanmin(w_e):+.2f}% at {q[int(np.nanargmin(w_e))]} | "
              f"GDP cum = {np.nansum(eff[m]['GDP'])/4:.1f} %-yrs")
    print(f"  Data excess price growth (core PCE - 2%): {np.nansum(data_pi)*100:.2f}pp; "
          f"2021Q2 excess = {data_pi[i_21q2]*400:.2f}pp ann.")

    # All demand shocks = every shock except Y_bar (beta + r + G,T): the 'brgt'
    # ablation subset (for conv models Y_bar is absent anyway, so 'brgt' == 'all').
    # Real wage w is normalized to % dev from SS.
    print("=== All demand shocks (all shocks except Ybar; 'brgt' ablation) ===")
    abl = pickle.load(open(ABLATION_PKL, 'rb'))
    ss_w = pickle.load(open(REPLAY_HWB_PKL, 'rb'))['results']
    for m in ['HANK', 'HANK_CC']:
        pi_b = np.asarray(abl[m]['brgt']['pi'], float)
        w_b = np.asarray(abl[m]['brgt']['w'], float) / float(ss_w[m]['ss_ref']['w']) * 100.0
        i_hi, i_lo = int(np.nanargmax(w_b)), int(np.nanargmin(w_b))
        print(f"  {m:8s}: pi 2021Q2 = {pi_b[i_21q2]*400:+.2f}pp ann. | "
              f"cum price growth (2020Q2-2023Q4) = {np.nansum(pi_b)*100:+.2f}pp | "
              f"real w 2021Q2 = {w_b[i_21q2]:+.2f}% | "
              f"real w cum (2023Q4) = {w_b[-1]:+.2f}% | "
              f"real w peak = {w_b[i_hi]:+.2f}% at {q[i_hi]} | "
              f"real w trough = {w_b[i_lo]:+.2f}% at {q[i_lo]}")

    print("=== Simulation levels (all shocks) ===")
    for m in ['HANK_CC', 'HANK', 'RANK_CC', 'RANK']:
        pi_m = np.asarray(res[m]['pi_NL'], float)
        x, y = pi_m, np.asarray(data_pi, float)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"  {m:8s}: pi 2021Q2 = {pi_m[i_21q2]*400:+.2f}pp ann. | "
              f"cum price growth (2020Q2-2023Q4) = {np.nansum(pi_m)*100:+.2f}pp "
              f"(peak {np.nanmax(pi_m)*400:.2f}pp at {q[int(np.nanargmax(pi_m))]}); "
              f"R2(data on model) = {r2:.3f}")

    # Real wage extremes (all shocks): max and min of the real-wage level,
    # normalized to % dev from SS (w_NL / ss_w * 100), with their quarters.
    print("=== Real wage max/min (all shocks) ===")
    for m in ['HANK_CC', 'HANK', 'RANK_CC', 'RANK']:
        w_m = np.asarray(res[m]['w_NL'], float) / float(res[m]['ss_ref']['w']) * 100.0
        i_hi, i_lo = int(np.nanargmax(w_m)), int(np.nanargmin(w_m))
        print(f"  {m:8s}: max = {w_m[i_hi]:+.2f}% at {q[i_hi]} | "
              f"min = {w_m[i_lo]:+.2f}% at {q[i_lo]}")
    print("=== Shares attributable to fiscal policy ===")
    for m in ['HANK_CC', 'HANK']:
        pi_model = np.asarray(res[m]['pi_NL'], float) * 400.0
        print(f"  {m:8s}: share of the model's 2021Q2 inflation spike "
              f"(conclusion) = {eff[m]['pi'][i_21q2]/pi_model[i_21q2]*100:.1f}% | "
              f"share of the data's 2021Q2 excess inflation = "
              f"{eff[m]['pi'][i_21q2]/(data_pi[i_21q2]*400)*100:.0f}% | "
              f"share of the data's cumulative excess price growth = "
              f"{(np.nansum(eff[m]['pi'])/4)/(np.nansum(data_pi)*100)*100:.0f}%")

    # Taylor-rule full paths and fiscal effects (HANK models only).
    if (os.path.exists(TAYLOR_NOGT_PKL)
            and all(os.path.exists(MODELS[m]['est_taylor']) for m in TAYLOR_MODELS)):
        tay_nogt = pickle.load(open(TAYLOR_NOGT_PKL, 'rb'))
        print("=== Taylor-rule simulation and fiscal effect ===")
        for m in ['HANK_CC', 'HANK']:
            full = pickle.load(open(MODELS[m]['est_taylor'], 'rb'))[
                f'{m}_finalmodel_taylor']
            pi_full = np.asarray(full['pi_NL'], float) * 400.0
            pi_eff = pi_full - np.asarray(tay_nogt[m]['pi'], float) * 400.0
            print(f"  {m:8s}: full pi 2021Q2 = {pi_full[i_21q2]:+.2f}pp ann. | "
                  f"fiscal pi 2021Q2 = {pi_eff[i_21q2]:+.2f}pp ann. | "
                  f"fiscal cumulative = {np.nansum(pi_eff)/4:+.2f}pp")


# ---------------------------------------------------------------------------
def stage_figures():
    fig_4model_paths()
    fig_shock_series()
    fig_fiscal_effect_4model()
    fig_tailsplice(REPLAY_HWB_PKL,
                   f'{FIGDIR}/finalmodel_real_tailsplice_pi_hankref_hankwbeta_ss.pdf')
    if os.path.exists(REPLAY_ARMA_HWB_PKL):
        fig_tailsplice(REPLAY_ARMA_HWB_PKL,
                       f'{FIGDIR}/finalmodel_arma11_tailsplice_pi_hankref_hankwbeta_ss.pdf')
    fig_real_vs_taylor()
    fig_ablation_singles()
    fig_profits_hankonly()


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'figures'
    args = sys.argv[2:]
    if stage == 'figures':
        stage_figures()
    elif stage == 'numbers':
        stage_numbers()
    elif stage == 'replay':
        stage_replay(arma=('arma' in args))
    elif stage == 'ablation':
        stage_ablation()
    elif stage == 'causal':
        stage_causal()
    elif stage == 'taylor-nogt':
        stage_taylor_nogt()
    elif stage == 'estimate':
        # IKC estimation: one per household type (covers conv AND CC)
        for key in (args or ['HANK', 'RANK']):
            estimate_model(key)
    elif stage == 'estimate-arma':
        for key in (args or ['HANK', 'RANK']):
            estimate_model_arma(key)
    elif stage == 'estimate-taylor':
        # Taylor rule: inflation feeds back into demand -> per FULL model
        for key in (args or TAYLOR_MODELS):
            estimate_model(key, taylor=True)
    else:
        raise SystemExit(f"unknown stage '{stage}' "
                         "(figures|numbers|replay|ablation|causal|taylor-nogt|"
                         "estimate|estimate-arma|estimate-taylor)")


if __name__ == '__main__':
    main()

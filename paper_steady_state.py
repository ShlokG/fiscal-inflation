"""paper_steady_state.py — The paper's (Table 1) steady-state calibration

Calibration logic:
  1. Household side: (beta_hi, dbeta, omega) solved so that A = 20 (500%
     annual wealth/GDP), on-impact weighted MPC out of income = 0.20, and the
     model Lorenz curve matches the 2019 SCF on average.
  2. CC supply side at u_ss = 0.792: with the full gross industry markup
     m = (gamma/(gamma-1)) M(v_bar) and N = Y_bar curly-N(v_bar), the retail
     CES elasticity eta is calibrated so the LABOR SHARE w*N = 0.6:
         w = ((eta-1)/eta)/m  and  w N = 0.6  =>
         eta = (-N/(.6 m)) / (1 - N/(.6 m))  = 7.04
  3. Conventional model: same eta; a constant CES markup with
         gamma_conv = 1/(1 - 0.6 * eta/(eta-1)) = 3.33   (Table 1's 3.33)
     so that mc = (gamma_conv/(gamma_conv-1)) w pins w = 0.6 and the labor
     share is 0.6 there too (N = Y = 1), using the gamma-markup conventional
     `mc` block in paper_model.py.
  4. Wage-PC slope target: kappa_w = 0.0092703, the Calvo wage formula
         (1/(1+Gamma)) (1 - beta_bar (1-f)) f/(1-f),  Gamma=5, f=0.194
     evaluated at the model's beta_bar = 0.954 (paper_model.calvo_kappa_w).
     The raw coefficient is k_w = kappa_w / (vphi N^(1+nu)). Because the labor
     share is 0.6 in both models, vphi N^(1+nu) = (1-Tau) C^(-1/eis) * (w N)
     = 0.78 * 1.25 * 0.6 = 0.585 in BOTH models, so
         k_w = 0.0092703 / 0.585 = 0.01585        (Table 1's k_w = .016)
  5. RANK: same supply-side values; beta_hi = 1/(1+r_ss) = 0.995 (dbeta = 0),
     omega copied from the HANK calibration, and the SAME dividend share
     mu = 0.1026 as HANK.

Output: {'ss_cc', 'ss_conv', 'ss_rank', 'ss_rank_cc'} cached at
cache/paper_ss_laborshare_alphaTau{a}_kappaW{k}.pkl.

Run `python paper_steady_state.py` to build and print the calibrated eta,
k_w, residuals and slopes.
"""
import os
import pickle

import numpy as np
from scipy import optimize

import paper_model as PM
from paper_model import (hank_cc_real, hank_real, rank_cc_real, rank_real,
                         hh_extended, utilization_inverse, markup, labor_demand,
                         CALIB, ALPHA_TAU, calvo_kappa_w)

# Wage-PC slope target: Calvo formula at the model's beta_bar = 0.954
# (= paper_model.calvo_kappa_w(0.9539845169579878)).
KAPPA_W_SLOPE = 0.009270298798672702


def ss_pkl_path(alpha_Tau=ALPHA_TAU, kappa_w=KAPPA_W_SLOPE):
    return f'cache/paper_ss_laborshare_alphaTau{alpha_Tau}_kappaW{kappa_w:.6f}.pkl'


def build(alpha_Tau=ALPHA_TAU, kappa_w=KAPPA_W_SLOPE, labor_share=0.6,
          verbose=True):
    """Build all four steady states under the labor-share calibration."""
    calib = dict(CALIB)
    calib['alpha_Tau'] = alpha_Tau
    Tau = calib['Tau_ss']

    # ---- 1. household-side calibration -------------------------------------
    A_target, mpc_target = 20, 0.20
    lorenz_scf_raw = np.loadtxt('lorenz_nw_scf_2019.raw', delimiter=',')
    percentiles = np.arange(101) / 100
    lorenz_scf = np.array([np.interp(p, lorenz_scf_raw[:, 0], lorenz_scf_raw[:, 1])
                           for p in percentiles])

    def get_lorenz(ss):
        D = ss.internals['hh']['D'].sum(axis=0)
        a_grid = ss.internals['hh']['a_grid']
        return np.array([np.interp(p, D.cumsum(), (a_grid * D).cumsum()) / ss['A']
                         for p in percentiles])

    def evaluate_calibration(x):
        ss = hh_extended.steady_state(calib | dict(
            beta_hi=x[0], dbeta=x[1], omega=x[2],
            r=calib['r_ante'], Z=(1-calib['mu'])*(1-Tau)))
        mpc_labor = hh_extended.jacobian(ss, inputs=['Z'], outputs=['C'], T=2)['C', 'Z'][0, 0]
        return (ss['A'] - A_target, mpc_labor - mpc_target,
                (get_lorenz(ss) - lorenz_scf).sum())

    res = optimize.root(evaluate_calibration, np.array([1, 0.08, 0.5]))
    ss_hh = hh_extended.steady_state(calib | dict(
        beta_hi=res.x[0], dbeta=res.x[1], omega=res.x[2],
        r=calib['r_ante'], Z=(1-calib['mu'])*(1-Tau)))
    C = ss_hh['C']
    if verbose:
        print(f"  household calib: beta_hi={res.x[0]:.5f} "
              f"beta_lo={res.x[0]-res.x[1]:.5f} omega={res.x[2]:.5f}")

    # ---- 2. CC supply side + eta from the labor share ----------------------
    gamma, sigma_v = calib['gamma'], calib['sigma_v']
    Y, Y_bar = calib['Y'], calib['Y_bar']
    nu, eis = calib['nu'], calib['eis']
    mu_v = -sigma_v**2/2
    Theta = np.exp(mu_v/gamma + sigma_v**2/(2*gamma**2))**(gamma/(gamma - 1))
    v_bar = utilization_inverse(Y/(Y_bar*Theta), gamma, sigma_v)
    m = markup(v_bar, gamma, sigma_v)         # full gross markup (gamma/(gamma-1))M
    N = Y_bar*labor_demand(v_bar, sigma_v)
    eta = (-N/(labor_share*m))/(1 - N/(labor_share*m))     # labor share = 0.6
    w = 1/(eta/(eta - 1)*m)
    vphi = (1-Tau)*w*C**(-1/eis)/(N**nu)
    k_w = kappa_w/(vphi*N**(1+nu))
    ss_cc = hank_cc_real.steady_state({**ss_hh, **calib, 'w': w, 'vphi': vphi,
                                       'eta': eta, 'k_w': k_w})

    # ---- 3. conventional model: gamma_conv markup, labor share 0.6 ---------
    gamma1 = 1/(1 - labor_share*(eta/(eta-1)))             # = 3.33
    m1 = gamma1/(gamma1-1)
    w1 = 1/((eta/(eta-1))*m1)                              # = 0.6 (N = 1)
    vphi1 = (1-Tau)*w1*C**(-1/eis)
    k_w1 = kappa_w/vphi1
    ss_conv = hank_real.steady_state({**calib, **ss_hh, 'eta': eta, 'w': w1,
                                        'vphi': vphi1, 'k_w': k_w1,
                                        'gamma': gamma1})

    # ---- 4. RANK: Euler beta, HANK omega, same supply values, same mu ------
    beta = 1/(1 + ss_cc['r'])
    ss_rank = rank_real.steady_state({**calib, 'omega': ss_cc['omega'],
                                        'beta_hi': beta, 'dbeta': 0, 'eta': eta,
                                        'w': w1, 'vphi': vphi1, 'k_w': k_w1,
                                        'gamma': gamma1})
    ss_rank_cc = rank_cc_real.steady_state({**calib, 'omega': ss_cc['omega'],
                                            'beta_hi': beta, 'dbeta': 0, 'eta': eta,
                                            'w': w, 'vphi': vphi, 'k_w': k_w,
                                            'gamma': gamma})

    out = {'ss_cc': ss_cc, 'ss_conv': ss_conv, 'ss_rank': ss_rank,
           'ss_rank_cc': ss_rank_cc,
           'eta': float(eta), 'gamma_conv': float(gamma1),
           'k_w_cc': float(k_w), 'k_w_conv': float(k_w1),
           'kappa_w_slope': float(kappa_w), 'alpha_Tau': alpha_Tau}
    if verbose:
        print(f"  eta = {eta:.4f}   gamma_conv = {gamma1:.4f}")
        print(f"  wage-PC slope target kappa_w = {kappa_w:.7f} "
              f"(Calvo at beta_bar=0.954: {calvo_kappa_w(0.9539845169579878):.7f})")
        print(f"  raw k_w: CC = {k_w:.5f}   conventional = {k_w1:.5f}")
        for lab, s in [('ss_cc', ss_cc), ('ss_conv', ss_conv),
                       ('ss_rank', ss_rank), ('ss_rank_cc', ss_rank_cc)]:
            print(f"  {lab:10s}: goods={float(s['goods_mkt']):+.2e} "
                  f"asset={float(s['asset_mkt']):+.2e} "
                  f"res_pc={float(s['res_pc']):+.2e} "
                  f"res_wpc={float(s['res_wpc']):+.2e} "
                  f"w={float(s['w']):.4f} "
                  f"labor_share={float(s['w'])*float(s['N']):.4f} "
                  f"wpc_slope={float(s['vphi'])*float(s['N'])**(1+float(s['nu']))*float(s['k_w']):.6f}")
    return out


def load_or_build(alpha_Tau=ALPHA_TAU, kappa_w=KAPPA_W_SLOPE, rebuild=False,
                  verbose=True):
    path = ss_pkl_path(alpha_Tau, kappa_w)
    if not rebuild and os.path.exists(path):
        if verbose:
            print(f"loaded paper SS from {path}")
        return pickle.load(open(path, 'rb'))
    out = build(alpha_Tau, kappa_w, verbose=verbose)
    os.makedirs('cache', exist_ok=True)
    pickle.dump(out, open(path, 'wb'))
    if verbose:
        print(f"saved {path}")
    return out


if __name__ == '__main__':
    load_or_build(rebuild=True)

"""paper_model.py — Model and calibration constants for
"The Inflationary Effects of Fiscal Stimulus" (Goyal, Lipton, Zhu).

Contains everything the paper's Section 2 (Model) and Table 1 describe:

  * the household block (two-point discount-factor process, KMV income process)
  * the capacity-constraint supply block: lognormal-v integrals, utilization
    u(v_bar), capacity-cost markup M(v_bar), labor demand N(v_bar)
    [paper Sec 2.1 / Appendix A.1]
  * the nonlinear Calvo price Phillips curve  [paper Sec 2.2 / Appendix A.2]
  * the nonlinear wage Phillips curve         [paper Sec 2.4 / Appendix A.3]
  * mutual fund, fiscal rule, monetary rules (real-rate rule and Taylor rule)
  * model assembly: {HANK, RANK} x {conventional, capacity-constraint} under
    the real-rate rule, HANK x {conventional, capacity-constraint} under the
    Taylor rule, plus the `wpc_wexo` RANK variants used for the paper's
    "hankwbeta_ss" convention (RANK's wage-PC discount set to HANK's
    steady-state beta_bar = 0.954; see paper footnote in Sec 4)
  * the calibration solvers that produce gamma, sigma_v,
    theta (price-PC slope) and kappa_w (Calvo wage formula).

Steady states are built by paper_steady_state.py: eta is calibrated so the
labor share is 0.6 (=> eta = 7.04), the wage-PC slope target is the Calvo
value 0.0092703 at beta_bar = 0.954 (=> raw k_w = 0.0158 in both models), and
the conventional model carries the constant CES markup gamma_conv/(gamma_conv-1)
with gamma_conv = 3.33 so its labor share is also 0.6.

Run `python paper_model.py` to re-solve gamma, sigma_v, theta and kappa_w and
compare them with the constants below.

Data files required in the working directory:
  ymarkov_combined.txt, ygrid_combined.txt   (KMV 2018 income process)
  lorenz_nw_scf_2019.raw                     (2019 SCF net-worth Lorenz curve,
                                              read by paper_steady_state.py)
"""
import os
import pickle
from functools import lru_cache

import numpy as np
import sequence_jacobian as sj
from scipy import interpolate, linalg, optimize
from scipy.stats import norm


# =============================================================================
# 1. CALIBRATION (paper Table 1)
# =============================================================================
# Solved/derived upstream values, hardcoded here with their solvers below:
#   gamma, sigma_v : solve_supply_curve_calib(0.35, 1.71, 0.792)
#       matches the Boehm and Pandalai-Nayar (2022) bias-adjusted first (0.35) and second
#       (1.71) derivative of log M(u^{-1}(e^{log u})) wrt log u at u = 0.792.
#   Y_bar          : 1/(Theta * 0.792) -> mean capacity utilization 79.2%.
#   theta          : solves (1-theta)(1-beta*theta)/theta = 0.052, the
#       Gagliardone et al. marginal-cost price-PC slope, at beta = 1/1.005.
#   kappa_w (slope of the wage PC): Calvo wage formula
#       (1/(1+Gamma)) * (1 - beta*(1-f)) * f/(1-f)
#       with real-rigidity Gamma = 5 and quarterly wage-change frequency
#       f = 0.194 (Grigsby-Hurst-Yildirmaz 2021), evaluated at the model's
#       beta_bar = 0.954 => 0.00927 (paper_steady_state.KAPPA_W_SLOPE).
GAMMA   = 4.07220983
SIGMA_V = 0.70885349
Y_BAR   = 1.342977672452851
THETA   = 0.7982397291550708
ALPHA_TAU = 0.03            # fiscal rule Tau_t = Tau_ss + alpha*(B_{t-1}-B_ss)
ALPHA_PI, ALPHA_Y = 1.5, 1.0/12.0   # Taylor rule weights (Orchard and Ramey 2025)

CALIB = {
    'eis': 1,               # sigma = 1 (inverse EIS)
    'pi': 1,                # zero-net-inflation steady state
    'r_ante': 0.02/4,       # 2% annual steady-state real rate
    'rss': 0.02/4,
    'G': .2,                # government spending / GDP
    'T': 0,                 # transfers / GDP
    'nu': 1,                # inverse Frisch
    'gamma': GAMMA,
    'theta': THETA,
    'sigma_v': SIGMA_V,
    'Y': 1,                 # SS GDP normalized to 1
    'Y_bar': Y_BAR,
    'Tau_ss': .22,          # tax revenue / GDP -> 100% annual debt/GDP
    'alpha_Tau': ALPHA_TAU,
    'alpha_pi': ALPHA_PI,
    'mu': 4*.02/(1-.22),    # = 0.1026: dividend share of GDP -> p_ss = 16,
                            # A/GDP = 20 (500% annual wealth-to-GDP)
    'q': .01,               # prob. of new beta draw (generational turnover)
    'min_a': 0, 'max_a': 4_000, 'n_a': 200,
}


def solve_supply_curve_calib(m_prime_target=0.35, m_2prime_target=1.71, u=0.792):
    """Solve (gamma, sigma_v) to match the BP2022 log-markup derivatives at u.
    Returns array [gamma, sigma_v]; the values used are GAMMA, SIGMA_V above."""
    def log_markup_(v_bar, gamma, sigma_v):
        return (np.log(gamma/(gamma - 1))
                - (1/(gamma - 1))*np.log(integral_lower(v_bar, sigma_v)
                  + v_bar**((gamma - 1)/gamma)*integral_upper(v_bar, gamma, sigma_v)))

    def chi_(v_bar, gamma, sigma_v):
        return ((1/gamma)*v_bar**((gamma - 1)/gamma)
                * integral_upper(v_bar, gamma, sigma_v)/integral_lower(v_bar, sigma_v))

    def chi_prime_(v_bar, gamma, sigma_v):
        mu = -sigma_v**2/2
        gpdf = (1/(v_bar*sigma_v*np.sqrt(2*np.pi))
                * np.exp(-(np.log(v_bar) - mu)**2/(2*sigma_v**2)))
        return ((((gamma - 1)/gamma)*v_bar**(-1/gamma)
                 * integral_upper(v_bar, gamma, sigma_v)*integral_lower(v_bar, sigma_v)
                 - v_bar*gpdf*(integral_lower(v_bar, sigma_v)
                   + v_bar**((gamma - 1)/gamma)*integral_upper(v_bar, gamma, sigma_v)))
                / (gamma*integral_lower(v_bar, sigma_v)**2))

    def inverse_(u_, gamma, sigma_v):
        return optimize.brentq(lambda v: utilization(v, gamma, sigma_v) - u_,
                               .00001, 10_000)

    def derivs(u_, gamma, sigma_v):
        v_bar = inverse_(u_, gamma, sigma_v)
        m_prime = chi_(v_bar, gamma, sigma_v)
        m_2prime = (chi_prime_(v_bar, gamma, sigma_v)
                    / utilization_prime(v_bar, gamma, sigma_v))*u_
        return m_prime, m_2prime

    def f(x):
        mp, mpp = derivs(u, x[0], x[1])
        return np.array([mp - m_prime_target, mpp - m_2prime_target])
    return optimize.fsolve(f, [5, 1])


def solve_theta(slope_target=0.052, beta=1/1.005):
    """Price stickiness theta from the linearized price-PC slope
    (1-theta)(1-beta*theta)/theta = slope_target."""
    return optimize.brentq(
        lambda th: (1-th)*(1-beta*th)/th - slope_target, .1, .9)


def calvo_kappa_w(beta, Gamma=5, freq_w_q=0.194):
    """Wage-PC slope from the Calvo wage formula; calvo_kappa_w(0.954) = 0.00927."""
    return (1/(1+Gamma))*(1 - beta*(1-freq_w_q))*freq_w_q/(1-freq_w_q)


# =============================================================================
# 2. HOUSEHOLD BLOCK (paper Sec 2.3)
# ============================================================================= 
# Income process from Kaplan-Moll-Violante (2018): load transition matrix,
# convert to discrete time, normalize so rows sum to 1 and E[e] = 1.
Pi_e = linalg.expm(np.loadtxt('ymarkov_combined.txt'))
Pi_e /= np.sum(Pi_e, axis=1)[:, np.newaxis]
pi_e = sj.utilities.discretize.stationary(Pi_e)
e_grid_short = np.exp(np.loadtxt('ygrid_combined.txt'))
e_grid_short /= e_grid_short @ pi_e


def make_betas(beta_hi, dbeta, omega, q):
    """Beta grid [beta_hi-dbeta, beta_hi] and transition matrix; q is the
    probability of a new draw from [1-omega, omega]."""
    beta_lo = beta_hi - dbeta
    b_grid = np.array([beta_lo, beta_hi])
    pi_b = np.array([1 - omega, omega])
    Pi_b = (1-q)*np.eye(2) + q*np.outer(np.ones(2), pi_b)
    return b_grid, Pi_b, pi_b


@sj.het(exogenous='Pi', policy='a', backward='Va',
        backward_init=sj.hetblocks.hh_sim.hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    """EGM backward step; beta is a vector over the (beta x e) state."""
    uc_nextgrid = beta[:, np.newaxis] * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def make_grids(min_a, max_a, n_a, beta_hi, dbeta, omega, q):
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    b_grid_short, Pi_b, pi_b = make_betas(beta_hi, dbeta, omega, q)
    e_grid = np.kron(np.ones_like(b_grid_short), e_grid_short)
    beta = np.kron(b_grid_short, np.ones_like(e_grid_short))
    Pi = np.kron(Pi_b, Pi_e)
    pi_pdf = np.kron(pi_b, pi_e)
    return e_grid, Pi, a_grid, beta, pi_pdf


def income(e_grid, Z, T):
    # z_i = (1-tau)(1-mu) X e_i + T   with Z = (1-tau)(1-mu) X  (paper Sec 2.7)
    y = Z*e_grid + T
    return y


hh_extended = hh.add_hetinputs([make_grids, income])


def compute_mpc(c, a, a_grid, r, e_grid):
    """Approximate mpc out of wealth, with symmetric differences where possible,
    exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    mpc[a == a_grid[0]] = 1
    mpc = mpc*e_grid[:, np.newaxis]
    return mpc


hh_extended = hh_extended.add_hetoutputs([compute_mpc])


# =============================================================================
# 3. SUPPLY SIDE (paper Sec 2.1-2.2, Appendix A.1)
# =============================================================================
# Lognormal-v analytic integrals: v ~ logN(mu, sigma_v^2), mu = -sigma_v^2/2
# so that E[v] = 1.
def integral_lower(v_bar, sigma_v):
    """int_0^v_bar v g(v) dv  (paper App A.1)."""
    mu = -sigma_v**2/2
    return np.exp(mu + sigma_v**2/2)*norm.cdf((np.log(v_bar) - mu - sigma_v**2)/sigma_v)


def integral_upper(v_bar, gamma, sigma_v):
    """int_v_bar^inf v^(1/gamma) g(v) dv."""
    mu = -sigma_v**2/2
    return np.exp(mu/gamma + sigma_v**2/(2*gamma**2))*norm.cdf(
        (mu/gamma + sigma_v**2/gamma**2 - (1/gamma)*np.log(v_bar))/(sigma_v/gamma))


def utilization(v_bar, gamma, sigma_v):
    """u(v_bar) — paper eq. (utilization), App A.1."""
    mu = -sigma_v**2/2
    Theta = np.exp(mu/gamma + sigma_v**2/(2*gamma**2))**(gamma/(gamma - 1))
    return (1/Theta)*(v_bar**((1 - gamma)/gamma)*integral_lower(v_bar, sigma_v)
                      + integral_upper(v_bar, gamma, sigma_v))**(gamma/(gamma - 1))


def utilization_prime(v_bar, gamma, sigma_v):
    """Analytic du/dv_bar; used to Newton-polish the spline inverse."""
    mu = -sigma_v**2/2
    Theta = np.exp(mu/gamma + sigma_v**2/(2*gamma**2))**(gamma/(gamma - 1))
    return (-1/(Theta*v_bar**2)
            * (integral_lower(v_bar, sigma_v)
               + v_bar**((gamma - 1)/gamma)*integral_upper(v_bar, gamma, sigma_v))**(1/(gamma - 1))
            * integral_lower(v_bar, sigma_v))


@lru_cache(maxsize=None)
def _vbar_spline(gamma, sigma_v):
    """Cached cubic spline for the inverse map v_bar(u)."""
    vg = np.geomspace(1e-3, 1e2, 4000)
    ug = utilization(vg, gamma, sigma_v)
    keep = (ug > 1e-4) & (ug < 0.99999)
    ug, vg = ug[keep], vg[keep]
    order = np.argsort(ug)
    ug, vg = ug[order], vg[order]
    uniq = np.concatenate(([True], np.diff(ug) > 0))
    return interpolate.CubicSpline(ug[uniq], np.log(vg[uniq]))


def utilization_inverse(u, gamma, sigma_v):
    """v_bar(u): spline + two Newton steps to machine precision."""
    g, s = float(gamma), float(sigma_v)
    u_clipped = np.clip(u, 1e-4, 0.99999)
    v_bar = np.exp(_vbar_spline(g, s)(u_clipped))
    for _ in range(2):
        v_bar = v_bar - (utilization(v_bar, g, s) - u_clipped) / utilization_prime(v_bar, g, s)
    return float(v_bar) if np.ndim(u) == 0 else v_bar


def markup(v_bar, gamma, sigma_v):
    """Gross markup (gamma/(gamma-1)) * M(v_bar) — paper eq. for M, App A.1.
    NOTE this function returns gamma/(gamma-1)*M, i.e. the FULL gross markup."""
    return (gamma/(gamma-1))*(integral_lower(v_bar, sigma_v)
            + v_bar**((gamma - 1)/gamma)*integral_upper(v_bar, gamma, sigma_v))**(1/(1-gamma))


def labor_demand(v_bar, sigma_v):
    """curly-N(v_bar): N = Y_bar * curly-N — paper labor-demand eq."""
    mu = -sigma_v**2/2
    return (1/v_bar)*integral_lower(v_bar, sigma_v) + (1-norm.cdf((np.log(v_bar) - mu)/sigma_v))


@sj.simple
def util(Y, Y_bar, gamma, sigma_v):
    """Capacity utilization u = Y / (Theta * Y_bar)."""
    mu = -sigma_v**2/2
    Theta = (np.e**(mu/gamma + sigma_v**2/(2*gamma**2)))**(gamma/(gamma - 1))
    u = Y/(Y_bar*Theta)
    return u


@sj.simple
def mc_convex(u, w, gamma, sigma_v):
    """CAPACITY-CONSTRAINT marginal cost: mc = w * [gamma/(gamma-1)] * M(v_bar(u)).
    (markup() includes the gamma/(gamma-1) factor.)"""
    v_bar = u.apply(lambda u: utilization_inverse(u, gamma, sigma_v))
    m = v_bar.apply(lambda v_bar: markup(v_bar, gamma, sigma_v))
    mc = w*m
    return mc, m


@sj.simple
def mc(w, gamma):
    """CONVENTIONAL marginal cost: a constant
    industry-level CES markup gamma/(gamma-1) on top of the wage, so
    w_ss = ((eta-1)/eta)/(gamma/(gamma-1)) and the labor share can be
    calibrated (to 0.6) independently of eta. In the Table 1 calibration
    (paper_steady_state.py), gamma_conv = 1/(1 - 0.6*eta/(eta-1)) = 3.33."""
    mc = (gamma/(gamma-1))*w
    return mc


@sj.simple
def labor_convex(u, Y_bar, gamma, sigma_v):
    """CC labor demand N = Y_bar * curly-N(v_bar(u))."""
    v_bar = u.apply(lambda u: utilization_inverse(u, gamma, sigma_v))
    curly_N = v_bar.apply(lambda v: labor_demand(v, sigma_v))
    N = Y_bar*curly_N
    return N


@sj.simple
def labor(Y):
    """Conventional labor demand N = Y."""
    N = Y
    return N


# Nonlinear Calvo price Phillips curve (paper Sec 2.2 / App A.2):
#   p* = (eta/(eta-1)) P_N/P_D
#   P_N = mc X + theta pi(+1)^eta     /(1+r(+1)) P_N(+1)
#   P_D =    X + theta pi(+1)^(eta-1) /(1+r(+1)) P_D(+1)
#   1 = (1-theta) p*^(1-eta) + theta pi^(eta-1)
#   Delta_p = (1-theta) p*^(-eta) + theta pi^eta Delta_p(-1),  X = Y/Delta_p
@sj.solved(unknowns={'P_N': 4.65, 'P_D': 4.85, 'Delta_p': 1},
           targets=["res_pc1", "res_pc2", "res_disp"])
def pc(Y, pi, mc, r_ante, eta, theta, P_N, P_D, Delta_p):
    p_star = (eta/(eta-1))*P_N/P_D
    res_pc1 = P_N - mc*(Y/Delta_p) - (1/(1+r_ante))*theta*pi(+1)**(eta)*P_N(+1)
    res_pc2 = P_D - (Y/Delta_p) - (1/(1+r_ante))*theta*pi(+1)**(eta-1)*P_D(+1)
    res_pc = 1 - (1-theta)*p_star**(1-eta) - theta*pi**(eta-1)
    res_disp = Delta_p - (1-theta)*p_star**(-eta) - theta*pi**(eta)*Delta_p(-1)
    return res_pc, res_pc1, res_pc2, res_disp


# =============================================================================
# 4. WAGE PHILLIPS CURVE (paper Sec 2.4 / App A.3)
# =============================================================================
#   pi_w (pi_w - 1) = k_w N (vphi N^nu - (1-tau) w C^(-1/eis))
#                     + beta_bar pi_w(+1) (pi_w(+1) - 1),  pi_w = w pi / w(-1)
# beta_bar = omega*beta_hi + (1-omega)*beta_lo: HANK 0.954, RANK 0.995.
@sj.simple
def wpc(tau, pi, w, N, C, nu, k_w, beta_hi, dbeta, omega, vphi, eis):
    beta = omega*beta_hi + (1-omega)*(beta_hi - dbeta)
    slack = N*(vphi*N**nu - (1-tau)*w*C**(-1/eis))
    res_wpc = w*pi/w(-1)*(w*pi/w(-1)-1) - k_w*slack - beta*w(+1)*pi(+1)/w*(w(+1)*pi(+1)/w-1)
    return res_wpc


# Exogenous-discount wage PC: the forward term is priced with the separate
# input beta_w. Used for the paper's baseline "hankwbeta_ss" RANK variants:
# beta_w = HANK's beta_bar (0.954) + RANK's OWN estimated beta shock path,
# so the RANK/HANK wage PCs share the same steady-state discount (paper
# footnote, Sec 4) while RANK demand still runs on RANK's own beta_hi.
@sj.simple
def wpc_wexo(tau, pi, w, N, C, nu, k_w, beta_w, vphi, eis):
    slack = N*(vphi*N**nu - (1-tau)*w*C**(-1/eis))
    res_wpc = w*pi/w(-1)*(w*pi/w(-1)-1) - k_w*slack - beta_w*w(+1)*pi(+1)/w*(w(+1)*pi(+1)/w-1)
    return res_wpc


# =============================================================================
# 5. INCOME SPLIT, PROFITS, FISCAL, MUTUAL FUND, MONETARY (paper Sec 2.5-2.6)
# =============================================================================
@sj.simple
def after_tax_income(Y, Delta_p, mu, Tau):
    tau = Tau*Delta_p/Y                    # tau_t = Tau_t / X_t
    Z = (1-tau)*(1-mu)*Y/Delta_p           # Z = (1-tau)(1-mu) X
    div = (1-tau)*mu*Y/Delta_p             # dividends = (1-tau) mu X
    return Z, tau, div


@sj.simple
def profits(w, N, Y, Delta_p, mc):
    Pi = Y/Delta_p - w*N                   # total profits Pi = X - w N
    labor_income = w*N
    Pi_int = mc*Y - w*N                    # industry (intermediate) profits
    return Pi, labor_income, Pi_int


@sj.solved(unknowns={'B': (-100, 100)}, targets=["res_gov"])
def fiscal(B, G, T, r_post_bonds, alpha_Tau, Tau_ss):
    """Government budget + fiscal rule Tau_t = Tau_ss + alpha (B_{t-1} - B_ss)."""
    Tau = Tau_ss + alpha_Tau*(B(-1)-B.ss)
    res_gov = G + T + (1+r_post_bonds)*B(-1) - B - Tau
    return res_gov, Tau


@sj.solved(unknowns={'p': (0.001, 50)}, targets=['cap_cond'])
def capitalization(div, r_ante, p):
    """Firm value p and ex-post equity return (no-arbitrage; time-0 capital
    gains from unanticipated shocks show up in r_post_equity)."""
    cap_cond = p - (div(+1) + p(+1))/(1 + r_ante)
    r_post_equity = (div + p)/p(-1) - 1
    return r_post_equity, cap_cond


@sj.simple
def ex_post_r(r_post_equity, r_post_bonds, B, p):
    """Portfolio (mutual fund) ex-post return."""
    r = (p(-1) * r_post_equity + B(-1) * r_post_bonds) / (p(-1) + B(-1))
    return r


@sj.simple
def mkt_clearing(Y, C, A, Delta_p, B, G, p):
    GDP = Y/Delta_p
    goods_mkt = GDP - G - C
    asset_mkt = B + p - A
    sr = 1 - C/GDP
    return goods_mkt, asset_mkt, sr, GDP


@sj.simple
def ex_post_bonds(i, pi):
    """Fisher equation under a nominal rule."""
    r_ante = (1+i)/pi(+1) - 1
    r_post_bonds = r_ante(-1)
    return r_post_bonds, r_ante


@sj.simple
def ex_post_bonds_real(r_ante):
    """Real-rate rule: r_ante is an exogenous input."""
    r_post_bonds = r_ante(-1)
    return r_post_bonds


@sj.simple
def taylor(rss, alpha_pi, alpha_y, pi, Y, eps_r):
    """Static Taylor rule; eps_r = deviation from the rule (0 in SS)."""
    i = rss + alpha_pi*(pi - 1) + alpha_y*(Y-1) + eps_r
    return i


# =============================================================================
# 6. RANK HOUSEHOLD + MODEL ASSEMBLY
# =============================================================================
@sj.solved(unknowns={"C": 0.8, "A": 20.0}, targets=["euler", "budget"])
def ra(C, A, Z, T, r, beta_hi, eis, omega, dbeta):
    """Representative agent on the Euler equation (RANK). At SS, beta = 1/(1+r)
    = 0.995 (omega = 1, dbeta = 0)."""
    beta = omega*beta_hi + (1-omega)*(beta_hi - dbeta)
    euler = C**(-1/eis) - beta*(1 + r(+1))*C(+1)**(-1/eis)
    budget = A - ((1 + r)*A(-1) + Z + T - C)
    return euler, budget


_COMMON = [after_tax_income, pc, fiscal, mkt_clearing, capitalization, ex_post_r, profits]
_CC_SUPPLY   = [mc_convex, util, labor_convex]
_CONV_SUPPLY = [mc, labor]

# --- real-rate rule (paper baseline) ---
hank_cc_real = sj.create_model([hh_extended, wpc, ex_post_bonds_real] + _CC_SUPPLY + _COMMON,
                               name="HANK with Capacity Constraints and Real Rate Rule")
hank_real    = sj.create_model([hh_extended, wpc, ex_post_bonds_real] + _CONV_SUPPLY + _COMMON,
                               name="Conventional HANK with Real Rate Rule")
rank_cc_real = sj.create_model([ra, wpc, ex_post_bonds_real] + _CC_SUPPLY + _COMMON,
                               name="RANK with Capacity Constraints and Real Rate Rule")
rank_real    = sj.create_model([ra, wpc, ex_post_bonds_real] + _CONV_SUPPLY + _COMMON,
                               name="Conventional RANK with Real Rate Rule")

# --- RANK wexo variants: wage-PC discount = exogenous beta_w (hankwbeta_ss) ---
rank_cc_real_wexo = sj.create_model([ra, wpc_wexo, ex_post_bonds_real] + _CC_SUPPLY + _COMMON,
                                    name="RANK CC, Real Rate Rule, Exogenous Wage-PC Discount")
rank_real_wexo    = sj.create_model([ra, wpc_wexo, ex_post_bonds_real] + _CONV_SUPPLY + _COMMON,
                                    name="Conventional RANK, Real Rate Rule, Exogenous Wage-PC Discount")

# --- IKC (demand-block-only) models: no supply side, Delta_p fixed at 1 ---
# GDP is fully demand-determined under the real-rate rule (paper Sec 2.8), and
# the demand block is IDENTICAL across the conventional and CC variants (same
# mu, taxes, transfers). So the (beta_t, rho_t) estimation only needs these
# IKC models — one estimation per household type covers both supply variants.
hank_ikc = sj.create_model([hh_extended, after_tax_income, fiscal, mkt_clearing,
                            capitalization, ex_post_bonds_real, ex_post_r],
                           name="IKC HANK")
rank_ikc = sj.create_model([ra, after_tax_income, fiscal, mkt_clearing,
                            capitalization, ex_post_bonds_real, ex_post_r],
                           name="IKC RANK")

# --- static Taylor rule ---
hank_cc_taylor = sj.create_model([hh_extended, wpc, ex_post_bonds, taylor] + _CC_SUPPLY + _COMMON,
                                 name="HANK with Capacity Constraints and Taylor Rule")
hank_taylor    = sj.create_model([hh_extended, wpc, ex_post_bonds, taylor] + _CONV_SUPPLY + _COMMON,
                                 name="Conventional HANK with Taylor Rule")


# =============================================================================
# 7. STEADY STATES
# =============================================================================
# Widen solved-block brackets so capitalization (p_ss = 16) and fiscal converge.
capitalization.unknowns = {'p': (0.001, 500.0)}
fiscal.unknowns         = {'B': (-100.0, 500.0)}

# Steady states are built by paper_steady_state.py (labor-share calibration).


def hank_wpc_beta_bar(ss_conv):
    """HANK steady-state wage-PC discount beta_bar = omega beta_hi +
    (1-omega) beta_lo = 0.954 — the value RANK's wpc_wexo is fed in the
    paper's hankwbeta_ss baseline."""
    return (float(ss_conv['omega']) * float(ss_conv['beta_hi'])
            + (1 - float(ss_conv['omega']))
            * (float(ss_conv['beta_hi']) - float(ss_conv['dbeta'])))


if __name__ == '__main__':
    print("Verifying calibration solvers against the constants used...")
    g, s = solve_supply_curve_calib()
    print(f"  gamma={g:.8f} (used {GAMMA}), sigma_v={s:.8f} (used {SIGMA_V})")
    th = solve_theta()
    print(f"  theta={th:.10f} (used {THETA})")
    print(f"  kappa_w Calvo at beta_bar=0.954 -> {calvo_kappa_w(0.9539845169579878):.7f} "
          f"(paper_steady_state slope target)")
    import paper_steady_state as PSS
    ss = PSS.load_or_build()
    print(f"  HANK wage-PC beta_bar = {hank_wpc_beta_bar(ss['ss_conv']):.4f} (paper: 0.954)")
    print(f"  RANK beta_hi = {float(ss['ss_rank']['beta_hi']):.5f} (paper: 0.995)")
    print(f"  eta = {ss['eta']:.4f} (Table 1: 7.04); "
          f"k_w = {ss['k_w_cc']:.5f} (Table 1: .016)")

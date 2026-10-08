"""paper_irfs.py — Supply-curve, iMPC and fiscal impulse-response figures
(paper Sections 2-3 and Appendix A).

Produces, in Figures/:
  supply_curve.pdf                                (Fig: supply curve, Sec 2.1)
  size_dependence_real_fiscal_{GDP,pi,w,Pi_int,Pi_tot,lab_inc}.pdf
                                                  (Fig: size dependence, Sec 3)
  IRF_{GDP,pi,w}.pdf and IRF_{GDP,pi,w}_cc.pdf    (Fig: fiscal IRFs, Sec 3)
  MPC.pdf                                         (Fig: iMPCs, Appendix A)

METHOD (paper Sec 3): one-time unanticipated fiscal shocks (transfers T or
government spending G) at t=0; perfect foresight of no further shocks;
nonlinear impulse under the real-rate rule; HANK conventional vs HANK CC.
  * size dependence: on-impact responses over shock sizes 0..10% of SS GDP
    (incl. on-impact labor income w*N, percent of SS GDP).
  * IRF panels: full impulse paths at a 10%-of-GDP shock; each wage response
    is normalized by its own model's steady-state wage.

Steady states / calibration: paper_steady_state.py (Table 1).

Usage:  python paper_irfs.py [supply|mpc|sizedep|irf]   (default: all)
"""
import os
import pickle
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict

import paper_model as PM
import paper_steady_state as PSS
from paper_model import (hank_real, hank_cc_real, hh_extended,
                         utilization_inverse, markup, labor_demand,
                         GAMMA, SIGMA_V)

FIGDIR = 'Figures'
T_SIM = 200
UNKS, TARGETS = ['Y', 'w', 'pi'], ['asset_mkt', 'res_pc', 'res_wpc']

# figure style
title_fontsize, label_fontsize, legend_fontsize = 18, 15, 15
line_width, tick_size = 2, 16
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
C_G, C_T = default_colors[0], default_colors[1]   # blue = G, orange = T


def _load_ss():
    d = PSS.load_or_build(verbose=False)
    ss_cc, ss_conv = d['ss_cc'], d['ss_conv']
    for s in (ss_cc, ss_conv):
        if 'Pi' not in s.toplevel:
            s['Pi'] = float(s['Y']) / float(s['Delta_p']) - float(s['w']) * float(s['N'])
        if 'Pi_int' not in s.toplevel:
            s['Pi_int'] = float(s['mc']) * float(s['Y']) - float(s['w']) * float(s['N'])
    return ss_cc, ss_conv


def _jac(model, ss, tag):
    """SS Jacobians for the quasi-Newton solver (cached in cache/)."""
    path = f'cache/cached_jac_paperls_{tag}.pkl'
    if os.path.exists(path):
        Js, H_U = pickle.load(open(path, 'rb'))
    else:
        inputs = ['r_ante', 'beta_hi', 'G', 'T'] + (['Y_bar'] if tag.endswith('CC') else [])
        Js = model.partial_jacobians(ss, inputs=set(inputs) | set(UNKS),
                                     outputs=set(TARGETS), T=T_SIM)
        H_U = model.jacobian(ss, UNKS, TARGETS, T=T_SIM, Js=Js)
        try:
            pickle.dump((Js, H_U), open(path, 'wb'))
        except Exception:
            pass
    return Js, FactoredJacobianDict(H_U, T_SIM)


def _solve(model, ss, Js, H_U_f, shocks):
    return model.solve_impulse_nonlinear(
        ss, unknowns=UNKS, targets=TARGETS, inputs=shocks,
        H_U_factored=H_U_f, Js=Js, maxit=500, tol=1e-8, verbose=False,
        internals=['hh'])


# ---------------------------------------------------------------------------
# 1. Partial-equilibrium supply curve (paper Fig 'supply_curve')
# ---------------------------------------------------------------------------
def fig_supply_curve():
    gamma, sigma_v = GAMMA, SIGMA_V
    mu = -sigma_v**2/2
    Theta = np.exp(mu/gamma + sigma_v**2/(2*gamma**2))**(gamma/(gamma - 1))
    uss = .792

    def M_of_u(u):       # capacity-cost markup M (net of the CES gamma/(gamma-1))
        v = utilization_inverse(u, gamma, sigma_v)
        return markup(v, gamma, sigma_v) * (gamma - 1) / gamma

    def N_of_u(u):       # labor-demand scale curly-N = N/Y_bar
        v = utilization_inverse(u, gamma, sigma_v)
        return labor_demand(v, sigma_v)

    h = 1e-6
    m_ss = M_of_u(uss)
    m_prime = (np.log(M_of_u(np.exp(np.log(uss)+h))) - np.log(M_of_u(np.exp(np.log(uss)-h)))) / (2*h)

    u = np.linspace(.55, .99, 100)
    m = np.array([M_of_u(ui) for ui in u])
    N = np.array([N_of_u(ui) for ui in u])
    m_lin = m_ss + m_prime*(m_ss/uss)*(u - uss)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(u, m, '-', label="Capacity Constraint")
    plt.plot(u, np.ones(len(u)), '--', label="Conventional")
    plt.plot(u, m_lin, '-.', label="Capacity Constraint: Linearized")
    plt.axvline(x=uss, color='black', linestyle='--', label="Steady State")
    plt.xlabel("Capacity Utilization")
    plt.title("Capacity Cost Markup")
    plt.legend(framealpha=0)

    plt.subplot(1, 2, 2)
    plt.plot(u, N, '-', label="Capacity Constraint")
    plt.plot(u, Theta*u, '-.', label="Conventional")
    plt.axvline(x=uss, color='black', linestyle='--', label="Steady State")
    plt.title("Labor Demand")

    os.makedirs(FIGDIR, exist_ok=True)
    out = f'{FIGDIR}/supply_curve.pdf'
    plt.savefig(out)
    plt.close()
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# 2. iMPC figure (paper Fig 'mpcs', Appendix A)
# ---------------------------------------------------------------------------
def fig_mpc():
    _, ss_conv = _load_ss()
    J = hh_extended.jacobian(ss_conv, inputs=['Z', 'r', 'T'], T=100)
    fig, ax = plt.subplots()
    ax.plot(J['C']['Z'][:, 0][0:25], '-', label="Weighted iMPC out of income (model)")
    ax.plot(J['C']['T'][:, 0][0:25], '--', label="Unweighted iMPC out of income (model)")
    ax.plot(J['C']['r'][:, 0][0:25] / ss_conv['A'], '-.',
            label="iMPC out of capital gains (model)")
    ax.legend(framealpha=0, fontsize=10)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel("Marginal Propensity to Consume", fontsize=10)
    os.makedirs(FIGDIR, exist_ok=True)
    out = f'{FIGDIR}/MPC.pdf'
    plt.savefig(out)
    plt.close(fig)
    print(f'Wrote {out}')


# ---------------------------------------------------------------------------
# 3. Size dependence (paper Fig 'size_dep')
# ---------------------------------------------------------------------------
def fig_size_dependence(n_grid=10, shock_max=0.10):
    ss_cc, ss_conv = _load_ss()
    Js_cc, HU_cc = _jac(hank_cc_real, ss_cc, 'HANK_CC')
    Js_c, HU_c = _jac(hank_real, ss_conv, 'HANK')
    T = np.linspace(.0001, shock_max, n_grid)

    keys = ['GDP', 'pi', 'lab_inc', 'w', 'Pi_tot', 'Pi_int']
    lines = {(m, s): {k: np.zeros(n_grid) for k in keys}
             for m in ('cc', 'conv') for s in ('T', 'G')}
    for mtag, (model, ss, Js, HU) in {
            'cc': (hank_cc_real, ss_cc, Js_cc, HU_cc),
            'conv': (hank_real, ss_conv, Js_c, HU_c)}.items():
        for which in ('T', 'G'):
            L = lines[(mtag, which)]
            for i, size in enumerate(T):
                d = size * 0.0 ** np.arange(T_SIM)
                irf = _solve(model, ss, Js, HU, {which: d})
                L['GDP'][i] = 100 * float(irf['GDP'][0])
                L['pi'][i] = 100 * 4 * float(irf['pi'][0])
                L['lab_inc'][i] = 100 * float(irf['labor_income'][0])
                L['w'][i] = 100 * float(irf['w'][0]) / float(ss['w'])
                L['Pi_tot'][i] = 100 * float(irf['Pi'][0])
                L['Pi_int'][i] = 100 * float(irf['Pi_int'][0])
                print(f"  size-dep {mtag}/{which} {i+1}/{n_grid}", flush=True)

    # ---- per-panel plotting ----------------------------------------------
    panel_figsize = (4, 4)
    panel_margins = dict(left=0.28, right=0.96, top=0.96, bottom=0.16)
    os.makedirs(FIGDIR, exist_ok=True)

    def plot_panel(key, ylabel, outname, ylim=None, legend=False, xlabel=False):
        fig, ax = plt.subplots(figsize=panel_figsize)
        fig.subplots_adjust(**panel_margins)
        ax.plot(100 * T, lines[('conv', 'T')][key], '--', color=C_T,
                label="Conventional (T)", linewidth=line_width)
        ax.plot(100 * T, lines[('conv', 'G')][key], '--', color=C_G,
                label="Conventional (G)", linewidth=line_width)
        ax.plot(100 * T, lines[('cc', 'T')][key], '-', color=C_T,
                label="Capacity Constraint (T)", linewidth=line_width)
        ax.plot(100 * T, lines[('cc', 'G')][key], '-', color=C_G,
                label="Capacity Constraint (G)", linewidth=line_width)
        ax.axhline(y=0, linestyle=':', color='black', linewidth=line_width - 1)
        ax.tick_params(axis='both', labelsize=tick_size)
        if xlabel:
            ax.set_xlabel("Fiscal Shock (Percent SS GDP)", fontsize=label_fontsize - 3)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.yaxis.set_label_coords(-0.24, 0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if legend:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.35 * (hi - lo))
            ax.legend(loc="upper left", framealpha=0, fontsize=tick_size - 3,
                      handlelength=1.2, handletextpad=0.3,
                      labelspacing=0.2, borderaxespad=0.2)
        fig.savefig(f"{FIGDIR}/{outname}.pdf", bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {FIGDIR}/{outname}.pdf')

    plot_panel('GDP', "Percent steady state GDP", "size_dependence_real_fiscal_GDP",
               legend=True, xlabel=True)
    plot_panel('pi', "Percentage points", "size_dependence_real_fiscal_pi")
    plot_panel('lab_inc', "Percent steady state GDP", "size_dependence_real_fiscal_lab_inc")
    plot_panel('w', "Percent", "size_dependence_real_fiscal_w")
    plot_panel('Pi_int', "Percent steady state GDP", "size_dependence_real_fiscal_Pi_int",
               ylim=(0, 8.5))
    plot_panel('Pi_tot', "Percent steady state GDP", "size_dependence_real_fiscal_Pi_tot",
               ylim=(0, 8.5))


# ---------------------------------------------------------------------------
# 4. IRF time paths at a 10%-of-GDP fiscal shock (paper Fig 'irf_fiscal')
# ---------------------------------------------------------------------------
def fig_irfs(size=0.10, horizon=25):
    ss_cc, ss_conv = _load_ss()
    Js_cc, HU_cc = _jac(hank_cc_real, ss_cc, 'HANK_CC')
    Js_c, HU_c = _jac(hank_real, ss_conv, 'HANK')
    d = size * 0.0 ** np.arange(T_SIM)

    irf = {('cc', 'T'):   _solve(hank_cc_real, ss_cc, Js_cc, HU_cc, {'T': d}),
           ('cc', 'G'):   _solve(hank_cc_real, ss_cc, Js_cc, HU_cc, {'G': d}),
           ('conv', 'T'): _solve(hank_real, ss_conv, Js_c, HU_c, {'T': d}),
           ('conv', 'G'): _solve(hank_real, ss_conv, Js_c, HU_c, {'G': d})}

    irf_figsize = (4, 2.8)
    irf_margins = dict(left=0.23, right=0.96, top=0.95, bottom=0.21)
    os.makedirs(FIGDIR, exist_ok=True)

    def plot_irf(y_T, y_G, ylabel, outname, ylim=None, legend=False):
        fig, ax = plt.subplots(figsize=irf_figsize)
        fig.subplots_adjust(**irf_margins)
        ax.plot(y_T, color=C_T, label="Transfers", linewidth=line_width)
        ax.plot(y_G, '--', color=C_G, label="Government Spending", linewidth=line_width)
        ax.axhline(y=0, linestyle=':', color='black', linewidth=line_width - 1)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if legend:      # the legend panel also carries the x-axis label
            ax.set_xlabel("Quarter", fontsize=label_fontsize)
            ax.legend(loc="best", framealpha=0, fontsize=tick_size - 3,
                      handlelength=1.2, handletextpad=0.3,
                      labelspacing=0.2, borderaxespad=0.2)
        fig.savefig(f"{FIGDIR}/{outname}.pdf")
        plt.close(fig)
        print(f'Wrote {FIGDIR}/{outname}.pdf')

    h = horizon
    # Normalize each wage response by its own model's steady-state wage, as in
    # the size-dependence panels and the figure note in the paper.
    w_cc = float(ss_cc['w'])
    w_conv = float(ss_conv['w'])
    plot_irf(100*np.asarray(irf[('cc', 'T')]['GDP'][:h]),
             100*np.asarray(irf[('cc', 'G')]['GDP'][:h]),
             "Percent", "IRF_GDP_cc", legend=True)
    plot_irf(100*np.asarray(irf[('conv', 'T')]['GDP'][:h]),
             100*np.asarray(irf[('conv', 'G')]['GDP'][:h]),
             "Percent", "IRF_GDP")
    plot_irf(400*np.asarray(irf[('cc', 'T')]['pi'][:h]),
             400*np.asarray(irf[('cc', 'G')]['pi'][:h]),
             "Percentage points", "IRF_pi_cc", ylim=(0, 1.6))
    plot_irf(400*np.asarray(irf[('conv', 'T')]['pi'][:h]),
             400*np.asarray(irf[('conv', 'G')]['pi'][:h]),
             "Percentage points", "IRF_pi", ylim=(0, 1.6))
    plot_irf(100*np.asarray(irf[('cc', 'T')]['w'][:h])/w_cc,
             100*np.asarray(irf[('cc', 'G')]['w'][:h])/w_cc,
             "Percent", "IRF_w_cc", ylim=(-0.15, 0.16))
    plot_irf(100*np.asarray(irf[('conv', 'T')]['w'][:h])/w_conv,
             100*np.asarray(irf[('conv', 'G')]['w'][:h])/w_conv,
             "Percent", "IRF_w", ylim=(-0.15, 0.16))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('supply', 'all'):
        fig_supply_curve()
    if which in ('mpc', 'all'):
        fig_mpc()
    if which in ('sizedep', 'all'):
        fig_size_dependence()
    if which in ('irf', 'all'):
        fig_irfs()


if __name__ == '__main__':
    main()

"""Estimate the calibration regressions used in the paper.

1. Fiscal debt feedback (``alpha``, paper appendix "Calibration")::

       TaxReceipts_t / trendGDP_t
           = constant + alpha * Debt_{t-1} / trendGDP_t + error_t,

   on 2020Q1--2025Q2, with trend GDP from an HP(1600) filter on the full
   sample.

2. Real-rate persistence (``rho_r``).  The ex-ante quarterly real rate is the
   quarterly-average federal funds rate deflated by next-quarter headline PCE
   inflation; AR(1) on 1959Q2--2026Q1.

3. Capacity persistence (``rho_Ybar``).  Capacity is the HP(1600) cycle of
   log manufacturing capacity, expressed around the model's steady-state
   level; AR(1) on 1967Q2--2026Q1.

All FRED series are pulled as of a fixed ALFRED vintage (FRED_VINTAGE = 2026-08-06),
so the estimates are unaffected by later data releases or revisions.
Requires the FRED_API_KEY environment variable.

Usage:
    python paper_regressions.py
    python paper_regressions.py --summary            # full OLS tables
    python paper_regressions.py --vintage 2026-09-30 # different data vintage
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

from paper_data import Y_BAR_SS, fred_q


FRED_VINTAGE = '2026-08-06'        # ALFRED vintage used for the paper
ALPHA_REG_START = '2020-01-01'
ALPHA_REG_END = '2025-04-01'       # 2025Q2
PERSISTENCE_END = '2026-01-01'     # 2026Q1
HP_LAMBDA = 1600


def _fred_q(series, units='lin'):
    """Quarterly FRED series observed at the ALFRED vintage FRED_VINTAGE."""
    return fred_q(series, FRED_VINTAGE, units=units)


def estimate_alpha():
    """Return ``(ols_result, regression_sample)`` for the paper regression."""
    taxes = _fred_q('NA000327Q').rename('tax_receipts')
    debt = _fred_q('GFDEBTN').rename('federal_debt')
    gdp = _fred_q('GDP').rename('gdp')

    _, trend_gdp = hpfilter(gdp.dropna(), lamb=HP_LAMBDA)
    data = pd.concat([
        (taxes / trend_gdp).rename('tax_receipts_to_trend_gdp'),
        (debt / trend_gdp).rename('debt_to_trend_gdp'),
    ], axis=1)
    data['lagged_debt_to_trend_gdp'] = data['debt_to_trend_gdp'].shift(1)
    sample = data.loc[ALPHA_REG_START:ALPHA_REG_END].dropna(
        subset=['tax_receipts_to_trend_gdp', 'lagged_debt_to_trend_gdp'])

    X = sm.add_constant(sample['lagged_debt_to_trend_gdp'])
    result = sm.OLS(sample['tax_receipts_to_trend_gdp'], X).fit()
    return result, sample


def estimate_ar1(series, name):
    """Estimate ``x_t = constant + rho*x_{t-1} + error_t``."""
    current = series.rename(name)
    lag_name = f'lagged_{name}'
    sample = pd.concat([current, current.shift(1).rename(lag_name)], axis=1)
    sample = sample.dropna(subset=[name, lag_name])
    result = sm.OLS(sample[name], sm.add_constant(sample[lag_name])).fit()
    return result, sample


def estimate_rho_r():
    """Estimate persistence of the ex-ante quarterly real rate.

    DFF is an annual rate and is converted to a quarterly gross rate;
    headline PCE inflation is quarterly percent change and is shifted one
    quarter backward so pi[t+1] deflates the nominal rate at t.
    """
    dff = _fred_q('DFF') / 100
    pce_inflation = _fred_q('PCEPI', units='pch') / 100
    real_rate = ((1 + dff) ** 0.25 / (1 + pce_inflation.shift(-1)) - 1)
    real_rate = real_rate.loc[:PERSISTENCE_END].dropna()
    return estimate_ar1(real_rate, 'real_rate')


def estimate_rho_ybar():
    """Estimate persistence of the HP-filtered manufacturing-capacity gap."""
    capacity = _fred_q('CAPB50001SQ').loc[:PERSISTENCE_END].dropna()
    capacity_cycle, _ = hpfilter(np.log(capacity), lamb=HP_LAMBDA)
    ybar_gap = np.exp(capacity_cycle + np.log(Y_BAR_SS)) - Y_BAR_SS
    return estimate_ar1(ybar_gap, 'ybar_gap')


def print_result(label, parameter, result, sample, regressor):
    coefficient = float(result.params[regressor])
    se = float(result.bse[regressor])
    intercept = float(result.params['const'])
    print(f'\n{label}')
    print(f'Sample: {sample.index.min().date()}--{sample.index.max().date()} '
          f'(N={int(result.nobs)})')
    print(f'{parameter} = {coefficient:.6f} (SE {se:.6f}; '
          f'rounded {coefficient:.2f})')
    print(f'intercept = {intercept:.6f}; R-squared = {result.rsquared:.4f}')


def main(show_summary=False, vintage=None):
    global FRED_VINTAGE
    if vintage:
        FRED_VINTAGE = vintage
    print(f'FRED vintage: {FRED_VINTAGE}')

    alpha_result, alpha_sample = estimate_alpha()
    rho_r_result, rho_r_sample = estimate_rho_r()
    rho_ybar_result, rho_ybar_sample = estimate_rho_ybar()

    results = [
        ('Fiscal debt feedback', 'alpha', alpha_result, alpha_sample,
         'lagged_debt_to_trend_gdp'),
        ('Ex-ante real-rate persistence', 'rho_r', rho_r_result, rho_r_sample,
         'lagged_real_rate'),
        ('Capacity persistence', 'rho_Ybar', rho_ybar_result,
         rho_ybar_sample, 'lagged_ybar_gap'),
    ]
    for args in results:
        print_result(*args)

    if show_summary:
        for label, _, result, _, _ in results:
            print(f'\n=== {label}: full OLS summary ===')
            print(result.summary())


if __name__ == '__main__':
    args = sys.argv[1:]
    main(show_summary='--summary' in args,
         vintage=args[args.index('--vintage') + 1] if '--vintage' in args else None)

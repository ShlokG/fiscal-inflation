"""paper_data.py — Raw data -> data caches for the COVID simulations.
Builds the two pickles that paper_covid_simulations.py and
paper_robustness.py read:

  cache/revision_covid_cache.pkl  (DATA_CACHE)
      COVID-quarter shock paths (r_ante, Y_bar, T, G), observed outcomes
      (GDP, headline pi, core pi), the SS real rate, and the SPF real-rate
      and real-GDP expectations.
  cache/ffr_covid_cache.pkl       (FFR_CACHE)
      Actual Fed-Funds-Rate path in model units (Taylor-rule rate matching):
      model i - rss = data FFR - mean(FFR, 1990-2019); the rule's inflation
      gap stays model pi - 1 <-> data inflation - 2%/yr.

Construction (paper appendix "Data Inputs to Quantitative Exercise"):

  * quarterly data; shock window 2020Q2-2023Q4 (15 quarters)
  * GDP and capacity are the cyclical component of an HP(1600) filter on
    logs — GDP over the full sample from 1960Q1, capacity over its own
    availability (1967Q1 on); interest rates and headline inflation are
    demeaned against their 1990Q1-2019Q4 average; the observed
    core-inflation comparison is measured relative to the 2% annual target;
    fiscal shocks are the BEA pandemic-response outlays divided by trend
    real GDP
  * headline PCE is the deflator and the SPF inflation concept; core PCE is
    additionally stored for the observed inflation line in the figures
  * SPF real-rate expectations use one-quarter-ahead expected PCE; horizons
    past the SPF's reach are AR(1)-extended at rho_r = 0.82
  * SPF expected real GDP is scaled so the nowcast matches actual GDP, is
    kept from crossing 0 late in the window, and its deviations are measured
    against the same GDP trend as the observed-GDP deviations

Needs network access to FRED (fredapi), an API key in the environment
variable FRED_API_KEY, and the SPF / BEA files in Data/:
  Mean_TBILL_Level.xlsx, Mean_PCE_Level.xlsx, Mean_NGDP_Level.xlsx,
  NEA-CU23-Pandemic-Response-Q-and-A-Federal.xlsx

REPRODUCIBILITY: every FRED series is pulled as of a fixed ALFRED vintage
(FRED_VINTAGE = 2026-06-21, the date the paper's data caches were built), so
a rebuild returns the same data no matter when it is run. Pass
`--vintage YYYY-MM-DD` to build from a different vintage (the model must then
be re-estimated).

Usage:
    python paper_data.py                              # build caches that are missing
    python paper_data.py --force                      # rebuild both caches
    python paper_data.py --force --vintage 2026-09-30 # rebuild with new data vintage
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# ALFRED vintage: every FRED series is pulled AS OF this date (realtime_start
# = realtime_end), i.e. exactly the data that was available when the paper's
# data caches were built. Later data releases and revisions (which move the
# HP trends through the endpoint) therefore cannot change the output.
FRED_VINTAGE = '2026-06-21'

DATA_CACHE = 'cache/revision_covid_cache.pkl'
FFR_CACHE = 'cache/ffr_covid_cache.pkl'

SHOCK_START, SHOCK_END = '2020-04-01', '2023-10-01'   # 2020Q2..2023Q4
FILTER_START = '1960-01-01'    # GDP HP-filter sample start (full data; the
                               #   deflated series itself begins 1959Q1)
FILTER_END = '2019-10-01'      # pre-COVID sample end (rate/inflation means)
DEMEAN_START = '1990-01-01'    # demeaning window start for rates/inflation
LAMB = 1600                    # HP-filter smoothing (quarterly)
RHO_R = 0.82                   # AR(1) extension of SPF real-rate expectations
T_CACHE = 200                  # horizon the cached r_expect rows are extended to
Y_SS = 1.0                     # model SS GDP (paper_model CALIB['Y'])
Y_BAR_SS = 1.342977672452851   # model SS capacity (paper_model.Y_BAR)
PI_SS_ANN = 2.0                # inflation target (annual %); 0 in the model
RSS = 0.005                    # model SS nominal rate (quarterly net)

# BEA pandemic-response file: row codes summed into G and T (transfers);
# row 25 is the Child Tax Credit, of which only the pandemic EXPANSION counts
# (34.5 $bn/yr pre-expansion level in 2021, growing 4.3 $bn/yr into 2022).
G_CATEGORIES = [21, 42, 43, 44, 45, 46, 51, 52, 53, 54]
T_CATEGORIES = [26, 27, 28, 29, 30, 31, 33, 36, 37, 38]


def fred_client():
    """FRED API client; the key is read from the FRED_API_KEY environment
    variable (free key: https://fred.stlouisfed.org/docs/api/api_key.html)."""
    key = os.environ.get('FRED_API_KEY')
    if not key:
        raise RuntimeError('Set the FRED_API_KEY environment variable to a '
                           'FRED API key.')
    return Fred(api_key=key)


def fred_q(series, vintage, units='lin'):
    """Quarterly-average FRED series as of the ALFRED vintage `vintage`
    (YYYY-MM-DD)."""
    return fred_client().get_series(series, frequency='q', units=units,
                                    aggregation_method='avg',
                                    realtime_start=vintage, realtime_end=vintage)


def _fred_q(series, units='lin'):
    return fred_q(series, FRED_VINTAGE, units=units)


# ---------------------------------------------------------------------------
# COVID fiscal outlays (BEA pandemic-response file)
# ---------------------------------------------------------------------------
def covid_fiscal_sums(pcepi):
    """Real pandemic G and T outlays, quarterly rate, 2020Q2-2022Q4 (11
    quarters). Nominal $bn/yr sums over the category rows, /4 to a quarterly
    rate, deflated by headline PCE."""
    gt = pd.read_excel('Data/NEA-CU23-Pandemic-Response-Q-and-A-Federal.xlsx',
                       sheet_name='2023 CU Quarters')

    def cat_sum(cats):
        rows = gt[gt.iloc[:, 0].isin(cats)]
        return rows.iloc[:, 5:16].apply(pd.to_numeric, errors='coerce')\
                   .sum(axis=0, skipna=True) / 4

    G_sums, T_sums = cat_sum(G_CATEGORIES), cat_sum(T_CATEGORIES)
    child = gt[gt.iloc[:, 0] == 25]
    T_sums[5:7] = T_sums[5:7].astype(float) \
        + (child.iloc[0, 10:12].values.astype(float) - 34.5) / 4
    T_sums[7:11] = T_sums[7:11].astype(float) \
        + (child.iloc[0, 12:16].values.astype(float) - (34.5 + 4.3)) / 4

    defl = pcepi[SHOCK_START:'2022-10-01'].values
    return G_sums.values / defl * 100, T_sums.values / defl * 100


# ---------------------------------------------------------------------------
# SPF expectations
# ---------------------------------------------------------------------------
def _survey_dates(df):
    return pd.to_datetime(df['YEAR'].astype(str) + '-'
                          + ((df['QUARTER'] - 1) * 3 + 1).astype(str) + '-01')


def _tbill_extended(row):
    """Expected 3M T-bill rate (annual %) at horizons h=1..13 for one SPF
    survey row. TBILL1..TBILL6 are direct; TBILL7..13 are backed out of the
    annual-average forecasts TBILLB/C/D, splitting the next-year average into
    its not-yet-covered quarters (quarter-of-year specific)."""
    tb = {h: float(row[f'TBILL{h}']) for h in range(1, 7)}
    q = int(row['QUARTER'])
    if q == 1:
        nxt = (row['TBILLB'] - row['TBILL6'] / 4) / 3 * 4
        ext = {7: nxt, 8: nxt, 9: nxt,
               10: row['TBILLC'], 11: row['TBILLC'],
               12: row['TBILLC'], 13: row['TBILLC']}
    elif q == 2:
        nxt = (row['TBILLB'] - (row['TBILL5'] + row['TBILL6']) / 4) / 2 * 4
        ext = {7: nxt, 8: nxt,
               9: row['TBILLC'], 10: row['TBILLC'],
               11: row['TBILLC'], 12: row['TBILLC'], 13: row['TBILLD']}
    elif q == 3:
        nxt = (row['TBILLB']
               - (row['TBILL4'] + row['TBILL5'] + row['TBILL6']) / 4) * 4
        ext = {7: nxt,
               8: row['TBILLC'], 9: row['TBILLC'],
               10: row['TBILLC'], 11: row['TBILLC'],
               12: row['TBILLD'], 13: row['TBILLD']}
    else:
        ext = {7: row['TBILLC'], 8: row['TBILLC'],
               9: row['TBILLC'], 10: row['TBILLC'],
               11: row['TBILLD'], 12: row['TBILLD'], 13: row['TBILLD']}
    tb.update({h: float(v) for h, v in ext.items()})
    return tb


def _pce_expected_one_ahead(row):
    """Expected headline PCE inflation (annual %) at h=2..13, shifted ONE
    QUARTER AHEAD (deflates the h-quarter T-bill with h+1 expected inflation,
    the ex-ante real-rate convention). Horizons the survey cannot reach are
    NaN — the caller maps those to 'unobserved'."""
    pc = {h: float(row[f'PCE{h + 1}']) for h in range(2, 6)}
    q = int(row['QUARTER'])
    if q == 1:
        nxt = (row['PCEB'] - row['PCE6'] / 4) / 3 * 4
        ext = {6: nxt, 7: nxt, 8: nxt,
               9: row['PCEC'], 10: row['PCEC'], 11: row['PCEC'],
               12: row['PCEC'], 13: row['PCEC']}
    elif q == 2:
        nxt = (row['PCEB'] - (row['PCE5'] + row['PCE6']) / 4) / 2 * 4
        ext = {6: nxt, 7: nxt,
               8: row['PCEC'], 9: row['PCEC'], 10: row['PCEC'],
               11: row['PCEC'], 12: row['PCEC'], 13: np.nan}
    elif q == 3:
        nxt = (row['PCEB']
               - (row['PCE4'] + row['PCE5'] + row['PCE6']) / 4) * 4
        ext = {6: nxt,
               7: row['PCEC'], 8: row['PCEC'], 9: row['PCEC'],
               10: row['PCEC'], 11: row['PCEC'], 12: np.nan, 13: np.nan}
    else:
        ext = {6: row['PCEC'], 7: row['PCEC'], 8: row['PCEC'],
               9: row['PCEC'], 10: row['PCEC'],
               11: np.nan, 12: np.nan, 13: np.nan}
    pc.update({h: float(v) for h, v in ext.items()})
    return pc


def spf_survey_rows():
    """SPF T-bill + headline-PCE survey rows for the 15 shock quarters."""
    tbill = pd.read_excel('Data/Mean_TBILL_Level.xlsx')
    pce = pd.read_excel('Data/Mean_PCE_Level.xlsx')
    m = pd.merge(tbill, pce, on=['YEAR', 'QUARTER'])
    m['date'] = _survey_dates(m)
    m = m[(m['date'] >= SHOCK_START) & (m['date'] <= SHOCK_END)]
    return m.reset_index(drop=True)


def spf_real_rate_expectations(r_ss, survey, dff_pct, pi):
    """(15, 11) expected real-rate deviations from r_ss at horizons h=1..11.

    real_h = (1+TBILL_h)/(1+E[PCE]_{h+1}) - 1, each survey row shifted by the
    current-quarter gap between the actual ex-ante real FFR and the survey's
    own-quarter real rate (T-bill != FFR, and the SPF nowcast can be off).
    Horizons the survey cannot reach come out as exactly 0 (the AR(1)
    extension picks up from the last nonzero entry).

    dff_pct: actual FFR (annual %); pi: quarterly headline inflation (decimal)
    — the same series build_covid_cache uses everywhere else."""
    pi_next = pi.copy()
    pi_next.index = pi_next.index - pd.DateOffset(months=3)  # deflate qtr t by pi_{t+1}
    ffr_real = ((1 + dff_pct / 100) / (1 + pi_next * 4) - 1) * 100

    r_vals = np.zeros((len(survey), 11))
    for i in range(len(survey)):
        row = survey.iloc[i]
        tb, pc = _tbill_extended(row), _pce_expected_one_ahead(row)
        real = {h: ((1 + tb[h] / 100) / (1 + pc[h] / 100) - 1) * 100
                for h in range(2, 14)}
        ffr_diff = ffr_real.get(row['date'], np.nan) - real[2]
        rv = np.array([real[h] for h in range(3, 14)]) + ffr_diff  # h=1..11
        r_vals[i] = np.where(np.isnan(rv), r_ss * 400, rv) / 400 - r_ss
    return r_vals


def extend_ar1(rows, rho, T):
    """Continue each row with AR(1) decay at rho from its last nonzero
    (genuinely observed) entry. Same logic as
    paper_covid_simulations.extend_r_expect_ar1."""
    rows = np.asarray(rows, float)
    out = np.zeros((rows.shape[0], T))
    for i, row in enumerate(rows):
        nz = np.where(row != 0)[0]
        k, v = (nz[-1], row[nz[-1]]) if len(nz) else (-1, 0.0)
        out[i] = np.concatenate((row[:k + 1], v * rho ** np.arange(1, T - k)))
    return out


def spf_gdp_expectations(gdp_n, pce_ind, cyc, trend):
    """(16, 4) SPF-expected real-GDP deviations at h=1..4, surveys
    2020Q1..2023Q4 (the caller drops the 2020Q1 row).

    Real GDP = NGDP forecast (rescaled so the nowcast matches actual GDP)
    deflated by cumulated expected headline PCE; deviation = log difference
    from the HP(1600) trend of log deflated GDP. expect_to_0: from 2022 on,
    an expectation path that crosses 0 while the current cycle is positive is
    rescaled to decay toward 0 instead.

    cyc/trend: the HP components of log deflated GDP computed ONCE in
    build_covid_cache (full sample, FILTER_START on) — the same trend the
    observed-GDP deviations are measured against, so data and expectations
    are consistent by construction. (The exact filter start barely matters
    numerically: the HP(1600) trend 50+ years in is insensitive to it,
    ~1e-12.)"""
    ngdp = pd.read_excel('Data/Mean_NGDP_Level.xlsx')
    pce = pd.read_excel('Data/Mean_PCE_Level.xlsx')
    ngdp = ngdp[(ngdp['YEAR'] >= 2020) & (ngdp['YEAR'] <= 2023)]
    pce = pce[(pce['YEAR'] >= 2020) & (pce['YEAR'] <= 2023)]

    out = np.zeros((len(ngdp), 4))
    for i in range(len(ngdp)):
        d = pd.Timestamp(year=int(ngdp['YEAR'].iloc[i]),
                         month=(int(ngdp['QUARTER'].iloc[i]) - 1) * 3 + 1, day=1)
        adj_ngdp = ngdp.iloc[i, 4:8].values.astype(float) \
            * float(gdp_n[d]) / float(ngdp.iloc[i, 3])
        pce_expect = float(pce_ind[d]) \
            * np.cumprod(1 + pce.iloc[i, 4:8].values.astype(float) / 400)
        pos = trend.index.get_loc(d) + 1
        cy = np.log(adj_ngdp / pce_expect * 100 / 4) - trend.values[pos:pos + 4]
        cy = np.exp(cy + np.log(Y_SS)) - Y_SS
        c0 = cyc.values[pos - 1]
        if c0 > 0 and (cy < 0).any() and d >= pd.Timestamp('2022-01-01'):
            cy = (cy - cy.min()) / (c0 - cy.min()) * c0
        out[i] = cy[:4]
    return out


# ---------------------------------------------------------------------------
# Cache builders
# ---------------------------------------------------------------------------
def build_covid_cache(verbose=True):
    """Everything paper_covid_simulations.load_data() reads (DATA_CACHE)."""
    # -- FRED pulls (quarterly averages) --------------------------------
    pcepi = _fred_q('PCEPI')                       # headline PCE index
    pi = _fred_q('PCEPI', 'pch').dropna() / 100    # headline inflation (q rate)
    pi_core = _fred_q('PCEPILFE', 'pch').dropna() / 100  # core inflation
    gdp_n = _fred_q('GDP')                         # nominal GDP
    dff_pct = _fred_q('DFF').dropna()              # nominal FFR (annual %)
    capb = _fred_q('CAPB50001SQ').dropna()         # industrial capacity index

    # -- real (headline-PCE-deflated, quarterly-rate) series ------------
    Y = (gdp_n / pcepi * 100 / 4).dropna()
    # ex-ante real rate: nominal deflated by NEXT quarter's inflation
    r_ante = (1 + dff_pct / 100) ** .25 / (1 + pi.shift(-1)) - 1

    demean = slice(DEMEAN_START, FILTER_END)
    shockw = slice(SHOCK_START, SHOCK_END)

    r_ss = float(r_ante[demean].mean())

    # -- detrending ('p_demean_old') ------------------------------------
    # GDP over the full sample (FILTER_START on), filtered ONCE here and
    # shared: the log-HP cycle is the observed-GDP deviation and its trend
    # anchors the SPF GDP expectations; the LEVEL trend deflates the fiscal
    # shocks. Capacity is filtered over its own availability (1967Q1 on).
    lev_cyc, lev_trend = hpfilter(Y[FILTER_START:], lamb=LAMB)
    log_cyc, log_trend = hpfilter(np.log(Y[FILTER_START:]), lamb=LAMB)
    data_Y = np.exp(log_cyc[shockw] + np.log(Y_SS)) - Y_SS

    ybar_cyc, _ = hpfilter(np.log(capb), lamb=LAMB)
    ybar_shock = np.exp(ybar_cyc[shockw] + np.log(Y_BAR_SS)) - Y_BAR_SS

    data_pi = pi[shockw] - float(pi[demean].mean())
    # The core-PCE series is used only as the observed excess-inflation line
    # in the figures and reported data comparison. Measure it against the
    # literal 2% annual target: 2/400 = 0.5% per quarter.
    data_pi_core = pi_core[shockw] - PI_SS_ANN / 400.0
    r_shock = r_ante[shockw] - r_ss

    # fiscal: pandemic outlays over trend real GDP; 0 after 2022Q4
    G_sums, T_sums = covid_fiscal_sums(pcepi)
    trend_w = lev_trend[shockw]
    pad = np.zeros(len(trend_w) - len(G_sums))
    G_shock = np.concatenate((G_sums, pad)) / trend_w
    T_shock = np.concatenate((T_sums, pad)) / trend_w

    # -- SPF expectations ------------------------------------------------
    survey = spf_survey_rows()
    r_expect_raw = spf_real_rate_expectations(r_ss, survey, dff_pct, pi)  # (15, 11)
    r_expect = extend_ar1(r_expect_raw, RHO_R, T_CACHE)                   # (15, 200)
    ngdp_expect = spf_gdp_expectations(gdp_n, pcepi, log_cyc, log_trend)  # (16, 4)

    cache = {
        'shocks': {'r_ante': r_shock, 'Y_bar': ybar_shock,
                   'T': T_shock, 'G': G_shock},
        'out_vals': {'Y': np.asarray(data_Y), 'pi': np.asarray(data_pi)},
        'out_vals_core_pi': np.asarray(data_pi_core),
        'r_ss': r_ss,
        'r_expect': r_expect,
        'ngdp_expect': np.asarray(ngdp_expect),
        'T': T_CACHE,
        'shock_start_qtr': SHOCK_START,
        'shock_end_qtr': SHOCK_END,
        'fred_vintage': FRED_VINTAGE,
    }
    if verbose:
        print(f"  GDP filter sample {Y[FILTER_START:].index.min().date()}+, "
              f"capacity {capb.index.min().date()}+   r_ss = {r_ss:.6f}")
        print(f"  shocks 2020Q2..2023Q4: G[0]={float(G_shock.iloc[0]):+.4f} "
              f"T[0]={float(T_shock.iloc[0]):+.4f} "
              f"r[0]={float(r_shock.iloc[0]):+.4f} "
              f"Ybar[0]={float(ybar_shock.iloc[0]):+.4f}")
        print(f"  observed Y: {np.asarray(data_Y)[:5].round(4)}")
        print(f"  ngdp_expect {ngdp_expect.shape}, r_expect {r_expect.shape}")
    return cache


def build_ffr_cache(verbose=True):
    """Actual FFR path in model units (FFR_CACHE), DEVIATION convention:

        model i - rss  =  data FFR - mean(FFR, 1990-2019)      [= 2.90 %/yr]

    i.e. the data's own pre-COVID mean nominal rate is the Taylor-rule
    intercept, exactly parallel to how the real-rate shocks (demeaned by the
    1990-2019 mean real rate) and the inflation outcomes (demeaned by the
    1990-2019 mean inflation) are aligned. The rule's inflation gap is
    unaffected: model pi - 1 still corresponds to data inflation - 2%/yr.
    The data ZLB (0% nominal) maps to rss - mean/400."""
    dff = _fred_q('DFF').dropna()
    ffr_mean = float(dff[DEMEAN_START:FILTER_END].mean())
    win = dff[SHOCK_START:SHOCK_END]
    ffr_ann = np.asarray(win.values, float)
    assert len(ffr_ann) == 15, f"expected 15 quarters, got {len(ffr_ann)}"

    di_target = (ffr_ann - ffr_mean) / 400.0
    out = {
        'ffr_ann': ffr_ann,
        'ffr_mean_9019': ffr_mean,
        'i_target_level': RSS + di_target,     # model-unit nominal level
        'di_target': di_target,
        'i_floor_level': float(RSS + (0.0 - ffr_mean) / 400.0),
        'rss': RSS,
        'quarters': [str(d.date()) for d in win.index],
        'fred_vintage': FRED_VINTAGE,
    }
    if verbose:
        print(f"  FFR 1990-2019 mean = {ffr_mean:.3f} %/yr")
        print(f"  di_target (ann%): {np.round(out['di_target'] * 400, 2)}")
    return out


def main(force=False, vintage=None):
    global FRED_VINTAGE
    if vintage:
        FRED_VINTAGE = vintage
    print(f"FRED vintage: {FRED_VINTAGE}")
    os.makedirs('cache', exist_ok=True)
    for path, builder in [(DATA_CACHE, build_covid_cache),
                          (FFR_CACHE, build_ffr_cache)]:
        if os.path.exists(path) and not force:
            print(f"{path} exists — skipped (--force rebuilds it)")
            continue
        print(f"building {path} ...")
        pickle.dump(builder(), open(path, 'wb'))
        print(f"saved {path}")


if __name__ == '__main__':
    args = sys.argv[1:]
    main(force='--force' in args,
         vintage=args[args.index('--vintage') + 1] if '--vintage' in args else None)

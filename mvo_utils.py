from math import *
from datetime import datetime, date, time, timedelta
from time import sleep
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import cvxpy as cvx
import re, os
import matplotlib.pyplot as plt

component_path = "./sector_components/"
pricing_path = "./pricing/"

#variables for data download
frame = 20 #for limiting the range of optimizations, 1 year
hist_window = frame * 5 #for historical pricing

date_fmt = '%m-%d-%Y'
start_date = datetime.now() - timedelta(hist_window)
start_date = start_date.strftime(date_fmt)
sleep_time = 5

sector_tickers_map = {}
companies = pd.DataFrame([])

# Mean variance optimization
def get_mean_variance(rets):
    w_len = rets.shape[1] # number of columns
    eq_weights = np.asarray([1/w_len for _ in range(w_len)]) #default weights
    mu = rets.mean()
    std_dev = rets.std()
    cov_matrix = rets.cov()
    return w_len, eq_weights, mu.values, std_dev, cov_matrix.values    

def show_weights(weights, labels, ret, sigma):
    df = pd.DataFrame(weights, columns=labels)
    df['return'] = ret * 252
    df['sigma'] = sigma * np.sqrt(252)
    df['sharpe'] = df['return'] / df['sigma']
    return df

def get_weights(px, freq, lb, min_sum, max_sum, min_w, max_w, gamma):
    px = clean_nas(px)
    returns = px.sort_index().pct_change(); returns.iloc[0] = 0
    intervals = pd.to_datetime(date_intervals(returns, freq).index.tolist())
    valid_dates = [d for d in intervals if d in returns.index]    
    hist_alloc = pd.DataFrame(np.zeros((returns.shape)), index=returns.index, columns=returns.columns)
    for i in valid_dates:
        lb_returns = returns.loc[:i.date()].tail(lb).dropna()
        weights = np.array([0 for _ in range(len(returns.columns))])
        if (len(lb_returns) > 2):
            n, weights, mu_ret, std_dev, cov_mtrx = get_mean_variance(lb_returns)
            weights = get_mvo_allocations(
                n, mu_ret, cov_mtrx, min_sum, max_sum, min_w, max_w, gamma)
        hist_alloc.loc[i.date()] = weights
    hist_alloc = hist_alloc.loc[returns.index].replace(0, np.nan).fillna(method='ffill')
    hist_alloc.replace(np.nan, 0, inplace=True)
    return returns, hist_alloc

def get_mvo_allocations(n, mu_ret, cov_mtrx, min_sum, max_sum, min_w, max_w, gamma_val):
    mu = mu_ret.T
    Sigma = cov_mtrx
    w = cvx.Variable(n)
    gamma = cvx.Parameter(sign='positive')
    ret = mu.T * w 
    risk = cvx.quad_form(w, Sigma)
    prob = cvx.Problem(cvx.Maximize(ret - gamma*risk), 
        [cvx.sum_entries(w) >= min_sum, cvx.sum_entries(w) <= max_sum, 
         w > min_w, w < max_w])
    gamma.value = gamma_val
    prob.solve()
    if prob.status == 'optimal': 
        return [i[0] for i in w.value.tolist()]

def recommend_allocs(px, frame, lb, freq, min_sum, max_sum, min_w, max_w, gamma):
    px = clean_nas(px)
    px_portion = px[-abs(frame):].copy() 
    returns, alloc = get_weights(
        px_portion, freq, lb, min_sum, max_sum, min_w, max_w, gamma)
    port_perf = calc_port_performance(returns.values, alloc.values)
    pdf = pd.DataFrame(port_perf, index=returns.index, columns=["M2-cvxopt"])
    return px_portion, returns, alloc, pdf

def period_allocs(w, irange):
    w = (recomend_allocs(w, irange) / max_w).astype(np.int)
    return w
# show top holdings and last recomended holding set

def selected_allocs(alloc, frame, freq, periods=99):
    w = alloc[-frame:].astype(np.float16)
    intervals = pd.to_datetime(date_intervals(w, freq).index.tolist())
    w = w.loc[intervals[-periods:]].sum(axis=0).sort_values(ascending=False)
    return w[w > 0]

def last_allocation(alloc, min_weight):
    last_alloc = alloc[-1:].T
    last_alloc.columns = ['Allocation']
    last_alloc = last_alloc[last_alloc[last_alloc.columns[0]] > min_weight]
    return last_alloc

# Portfolio utils
p_template = "{0} Return: {1:.2f}, StdDev: {2:.2f}, Sharpe: {3:.2f}"

def calc_port_performance(arr, weights):
    return np.cumprod(np.sum(arr * weights, axis=1) + 1)

def date_rules(date_range, tgt_date_str, freq):
    #return a list of dates
    tgt_dt = datetime.strptime(tgt_date_str, date_fmt)
    return date_range[:date_range.index(tgt_dt)+1][::-freq]

def date_intervals(df, freq):
    #using pandas
    return df.resample(freq, closed='left', label='left').mean()

# This calculates the variance of a time series, not a portfolio
def portfolio_metrics(name, pdf):
    timespan = len(pdf)
    ret = (pdf.pct_change().mean() * timespan).values[0]
    std = (pdf.pct_change().std() * sqrt(timespan)).values[0]
    if log: print(p_template.format(name, ret, std, ret / std))
    return ret, std, ret / std

# this method is not correct, don't use it
def _compute_metrics(px_hold, rec, lb):
    tickers = rec.index.tolist()
    sel_df = px_hold[tickers][-lb:].pct_change()[1:]
    sel_df_ret = sel_df.mean() * lb; sel_df_std = sel_df.std() * sqrt(lb)
    sel_df_sharpe = sel_df_ret / sel_df_std
    sel_df_spread = sel_df_ret - sel_df_std
    vals = np.array(
        [rec.values, sel_df_ret.values, sel_df_std.values, 
         sel_df_sharpe.values, sel_df_spread,
         [dwld_key for i in range(len(tickers))]])
    sector_df = pd.DataFrame(vals.T, index=tickers, columns=sum_cols)
    return sector_df

def port_metrics(px, rec):
    # this is supposed to be the righ way to calculate the portfolio risk
    px.sort_index(inplace=True)
    returns = px[rec.index.tolist()].pct_change()
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    weights = np.asarray(rec.values)
    mult = len(mean_daily_returns)
    #port_return = np.sum(mean_daily_returns.values * weights) * mult # bug fix
    port_return = np.dot(mean_daily_returns.values, weights) * mult
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))) * np.sqrt(mult)
    return port_return[0], port_risk[0][0]

# Sector analytics
def check_sector_vars(group, dwld_key, frame, gamma):
    lb_range = [i for i in range(5, 25, 10)]
    w_range = [i/100 for i in range(5, 20, 5)]
    for l in lb_range:
        for w in w_range:
            _, _, _, pdf, benchmark = run_sector_opt(group, dwld_key, frame, l, w, gamma)
            plt.plot(pdf, "-", alpha=0.5, label="lb:" + str(l) + "-w:" + str(w))
    plt.plot(benchmark, "g:", label=dwld_key)
    plt.legend(loc='best')
    plt.grid(axis='both', linestyle=':', linewidth=0.5)
    plt.title("M2 vs. " + dwld_key)
    plt.show()
    
def run_sector_opt(group, dwld_key, frame, lback, w, gamma):
    px = load_pricing(dwld_key + '-hold-pricing.csv', 'Date') #this is inneficient, fix
    px_spy_etfs = load_pricing(group, 'Date') #this is inneficient, fix
    spyder_etf = px_spy_etfs[dwld_key].copy()
    # Run a sector specific optimization
    px_portion, returns, alloc, pdf = recommend_allocs(px, frame, lback, frequency, min_gross, max_gross, min_w, w, gamma)
    benchmark = (spyder_etf[-len(px_portion):].pct_change() + 1).cumprod()
    if log == True: 
        portfolio_metrics('M2', pdf);
        portfolio_metrics(dwld_key, pd.DataFrame(benchmark));
    return px_portion, returns, alloc, pdf, benchmark



# Plot utilities
# this method is not that useful, not thought out correctly
def plot_recomendations(picks, lb):
    # 1) get sectors 2) find stocks in sector pricing 3) merge all dfs into one 4) plot all as index
    sel_sectors = pd.unique(picks['Sector']) #1
    consol_px = pd.DataFrame([])
    idx_range = px_spy[-lb:].index
    for s in sel_sectors:
        s_px = load_pricing(s + '-hold-pricing.csv', 'Date').loc[idx_range].sort_index()
        s_px = clean_nas(s_px)
        s_tickers = picks[picks['Sector'] == s].index.tolist()
        s_df = s_px[s_tickers] #2
        consol_px = consol_px.merge(s_df, left_index=True, right_index=True, how='outer') #3
    compound(consol_px).plot() #4 plot compound
    print("From:", consol_px.index[0], "to: ", consol_px.index[-1])
    return consol_px

def plot_two_series(tsa, tsb, label1, label2, xlabel, ylabel, title):
    ax = tsa.plot(); tsb.plot(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.set_title(title)
    
# DOWNLOAD / LOAD Utility Methods

# Load component from ETF holding CSVs
col_names = ['Symbol','Company', 'Weight']
def load_components(cos, pattern, cols, idxcol, sectors, srows=1):
    flist = os.listdir(component_path)
    files = [f for f in flist if f.startswith(pattern)]
    for s in sectors:
        fname = component_path + pattern + s.lower() + '.csv'
        df = pd.read_csv(fname, skiprows=srows, index_col=idxcol, usecols=cols)
        df.index.name = col_names[0]
        df.columns = col_names[1:]
        df = clean_idx(df, ' ')
        df['ETF'] = s
        sector_tickers_map[s] = df.index.tolist()
        cos = cos.append(df)
    return cos

# Load pricing from hard drive
def load_pricing(f, idx_col):
    fname = pricing_path + f
    px = pd.read_csv(fname, index_col=idx_col, parse_dates=True)
    px.sort_index(ascending=True, inplace=True)
    if log: print("Loaded pricing for {}, with shape {}".format(f, px.shape))
    return px

# Downloads pricing on all components for each ETF
def get_pricing(fname, ticker_list, start_date):
    if log: print("Getting pricing for:", fname, start_date)
    px = web.DataReader(ticker_list,data_source='yahoo',start=start_date)['Adj Close']
    px.sort_index(ascending=True, inplace=True)
    px.to_csv(pricing_path + fname)
    return px

# Exception safe downloader
def get_safe_pricing(fname, ticker_list, s_date):
    while True:
        try:
            get_pricing(fname, ticker_list, s_date); break
        except Exception as err:
            print("Error: {0}, waiting to try again in {1}".format(err, sleep_time))
            sleep(sleep_time)

#For each ETF downloads 
def refresh_components(etfs):
    while len(etfs) > 0: 
        val = etfs[-1]; 
        tickers = sector_tickers_map[val] # for individual components
        get_safe_pricing(val + '-hold-pricing.csv', tickers, start_date)
        etfs.pop()

def load_hold_and_benchmark(key):
    px_hold = load_pricing(key + '-hold-pricing.csv', 'Date')
    px_bench = px_spy_etfs[key].copy()
    return px_hold, px_bench

def load_spy_consol_px():
    consol_px = pd.DataFrame([])
    for dwld_key in ticker_map['spy_sectors']:
        px = load_pricing(dwld_key + '-hold-pricing.csv', 'Date')[frame:].copy()
        consol_px = consol_px.merge(px, left_index=True, right_index=True, how='outer')
    return consol_px

# CLEAN UTILITIES

cleanmin = lambda x: max(float(x), 1)
short_float = lambda x: '%.3f' % x
def compound(df):
    pct = df.pct_change() + 1
    pct.iloc[0] = 1
    return pct.cumprod()

def clean_load(pattern, idxcol, cols, col_names, s, srows=0):
    fname = component_path + pattern + s.lower() + '.csv'
    df = pd.read_csv(fname, skiprows=srows, index_col=idxcol, usecols=cols)
    df.index.name = col_names[0]
    df.columns = col_names[1:]
    return df

def clean_idx(df, s):
    dfidx = df.index.dropna()
    df = df.loc[dfidx].copy()
    rows = df[df.index.str.contains(s) == True]
    if len(rows) > 0:
        idx = df[df.index.str.contains(s) == True].index
        df = df.drop(idx, axis=0)
    return df

def clean_nas(df):
    cols = df.count().sort_values()[df.count().sort_values() < 1].index.tolist()
    df = df.drop(cols, axis=1)
    df.fillna(method='pad', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df = df.applymap(cleanmin)
    return df
"""Cycle-finding tools.
"""
from collections import namedtuple
from datetime import date
import math
from pathlib import Path
import pickle
from warnings import warn

import numpy as np
from scipy.optimize import minimize_scalar

import pandas as pd
import yfinance as yf

import sys

_DATA_DIR = Path(__file__).parent / "data"


class Portfolio(object):
    """Portfolio object.

    Object attributes:
    capital: float, initialize with amount of investing capital;
        portfolio value will be stored here
    stock_data: DataFrame indexed by dates, columns are symbols, rows
        are closing prices for stocks in scope
    positions: DataFrame indexed by dates, columns are symbols, rows
        are number of shares held, first column is cash
    shares_owned: number of shared owned at last transaction
    """

    path = Path(_DATA_DIR)

    def __init__(self, start_date, capital=0.0, stock_list=[], path=None):
        """Method for initializing an instance of a Portfolio class

        Inputs:
        start_date: string 'YYYY-MM-DD' date of portfolio creation
        capital: float, initial capital
        stock_list: list of ticker symbols in scope for portfolio,
            populate this at portfolio creation and all data will
            be loaded when portfolio is initialized
        path: path to data (directory of $TICKER.pkl files)
        """
        if path is not None:
            self.path = path
        self.capital = capital
        self.stock_data = read_adj_close(stock_list)
        # set start_date to next trading day (could be start_date)
        start_date_t = pd.to_datetime(start_date)
        # initialize portfolio to be all cash and 0 shares of each symbol
        self.positions = pd.DataFrame(
            np.array([capital]),
            index=pd.date_range(start_date_t, periods=1),
            columns=["cash"],
        )
        self.shares_owned = 0.0

    def value(self, val_date=None):
        """Method for computing portfolio value for a date

        Inputs:
        val_date : optional string 'YYYY-MM-DD', date on which to
            calculate value, uses previous trading day if val_date
            is not a trading day and uses last activity day in
            portfolio if val_date is unspecified
        """
        if val_date is None:
            val_date_t = self.positions.index[-1]
        else:
            val_date_t = pd.to_datetime(val_date)
        start_date_t = self.positions.index[0]
        # return 0 if val_date is earlier than portfolio creation
        if val_date_t < start_date_t:
            return 0.0
        # get stock values for previous trading day (could be val_date)
        sval_idx = self.stock_data.index.get_loc(val_date_t, method="ffill")
        sval_date_t = self.stock_data.index[sval_idx]
        share_vals = np.array([1.0])
        for symbol in self.positions.columns[1:]:
            share_vals = np.append(share_vals, self.stock_data.at[sval_date_t, symbol])
        # get shares owned for previous trading day (could be val_date)
        pval_idx = self.positions.index.get_loc(val_date_t, method="ffill")
        pval_date_t = self.positions.index[pval_idx]
        n_shares = self.positions[pval_date_t:pval_date_t].values[0, :]
        return (share_vals * n_shares).sum()

    def buy(self, symbol, buy_date, n_shares=-1):
        """Buy method for Portfolio class.

        Inputs:
        symbol : string, stock to buy
        buy_date : string 'YYYY-MM-DD', date to buy, uses next trading
            day if buy_date is not a trading day
        n_shares : float (optional), number of shares to buy, will
            buy as much as possible if n_shares is -1
        """
        # load stock data if we don't have it
        if symbol not in self.stock_data.columns:
            tickers = list(self.stock_data.columns) + [symbol]
            self.stock_data = read_adj_close(tickers, path=self.path)
            # return if could not load data for new symbol
            if symbol not in self.stock_data.columns:
                return
        # set buy_date to next trading day (could be buy_date)
        buy_date_t = pd.to_datetime(buy_date)
        buy_idx = self.stock_data.index.get_loc(buy_date_t, method="bfill")
        buy_date_t = self.stock_data.index[buy_idx]
        # return if buy_date < last activity in portfolio
        last_activity_t = self.positions.index[-1]
        if buy_date_t < last_activity_t:
            print(f"[e] buy date {buy_date} < last activity date")
            return
        # add a column if we haven't owned this stock yet
        if symbol not in self.positions.columns:
            self.positions[symbol] = np.zeros(len(self.positions))
        # add a row if buy_date > last date in DataFrame
        if buy_date_t > last_activity_t:
            new_row = pd.DataFrame(
                self.positions[-1:].values,
                index=pd.date_range(buy_date_t, periods=1),
                columns=self.positions.columns,
            )
            self.positions = self.positions.append(new_row)
        # get buy price
        buy_price = self.stock_data.at[buy_date_t, symbol]
        # perform transaction on last row
        max_can_buy = np.floor(self.positions.at[buy_date_t, "cash"] / buy_price)
        if n_shares == -1 or max_can_buy < n_shares:
            # print(f"[i] buying maximum shares of {symbol}: {max_can_buy}")
            n_shares = max_can_buy
        self.positions.at[buy_date_t, "cash"] -= n_shares * buy_price
        self.positions.at[buy_date_t, symbol] += n_shares
        self.shares_owned += n_shares

    def sell(self, symbol, sale_date, n_shares=-1):
        """Sell method for Portfolio class.

        Inputs:
        symbol : string, stock to sell, use 'all' to sell everything
        sale_date : string 'YYYY-MM-DD', date to sell, uses next trading
            day if sale_date is no a trading day
        n_shares : float (optional), number of shares to sell, all
            will be sold if n_shares is -1
        """
        # set sale_date to next trading day (could be sale_date)
        sale_date_t = pd.to_datetime(sale_date)
        sale_idx = self.stock_data.index.get_loc(sale_date_t, method="bfill")
        sale_date_t = self.stock_data.index[sale_idx]
        # return if sale_date < last activity in portfolio
        last_activity_t = self.positions.index[-1]
        if sale_date_t < last_activity_t:
            print(f"[e] sale date {sale_date} < last activity date")
            return
        # add a row if sale_date > last date in DataFrame
        if sale_date_t > last_activity_t:
            new_row = pd.DataFrame(
                self.positions[-1:].values,
                index=pd.date_range(sale_date_t, periods=1),
                columns=self.positions.columns,
            )
            self.positions = self.positions.append(new_row)
        if not symbol == "all":
            have_some = (self.positions[-1:] > 0).any()
            # return if not selling all and don't own any shares of symbol
            if symbol not in have_some.index[have_some]:
                print(f"[e] own zero shares: {symbol}")
                return
            # create stocks_to_sell list
            stocks_to_sell = [symbol]
        else:
            # selling everything
            stocks_to_sell = list(self.positions.columns[1:])
        # loop over stocks to sell
        for symbol in stocks_to_sell:
            # get sale_price
            sale_price = self.stock_data.at[sale_date_t, symbol]
            # perform transaction on last row
            n_shares_owned = self.positions.at[sale_date_t, symbol]
            if n_shares == -1:
                n_shares = n_shares_owned
            else:
                n_shares = min(n_shares_owned, n_shares)
            self.positions.at[sale_date_t, "cash"] += n_shares * sale_price
            self.positions.at[sale_date_t, symbol] -= n_shares
            self.shares_owned -= n_shares


def read_tickers(fname):
    """Reads a list of tickers from a plain text file, one symbol per line.

    Inputs:
    fname   : File name containing ticker symbols

    Outputs:
    tickers : List of symbols

    Example:
    In [1]: from cycles import *
    In [2]: tickers = read_tickers('otherlisted.txt')
    """

    with open(fname) as f:
        tickers = f.read().splitlines()
    f.close()
    # remove comments and blank lines
    tickers = [t for t in tickers if len(t) > 0 and not str.isspace(t) and t[0] != "#"]

    return tickers


def update_adj_close(tickers, path=None):
    """Reads adjusted close price for a list of ticker symbols from
    Yahoo Finance from 1990-01-01 to present and saves each as a
    Pandas DataFrame.

    Inputs:
    tickers : List of symbols
    path    : Path to a directory where results should be stored,
              file name is path/TICKER.pkl; default is $PWD/data

    Outputs:
    none

    Example:
    In [3]: from cycles import * # done previously in above example
    In [4]: mkdir tmp
    In [5]: update_adj_close(['IBM','FCEL',], 'tmp')
    """
    if path is None:
        path = _DATA_DIR
    else:
        path = Path(path)
    start_date = "1900-01-01"

    today = date.today()
    end_date = today.strftime("%Y-%m-%d")

    for t in tickers:
        print(f"loading: {t}")
        _df = yf.download(t, start=start_date, end=end_date)
        f = open(path / f"{t}.pkl", "wb")
        df = pd.DataFrame(
            _df["Adj Close"],
            index=_df.index,
            columns=[
                "Adj Close",
            ],
        )
        pickle.dump(df, f)
        f.close()


def read_adj_close(tickers, path=None, min_len=2500):
    """Reads adjusted close prices saved by update_adj_close() above
    and returns a Pandas DataFrame indexed by the largest common
    date range with the symbol names as column headers.

    Inputs:
    tickers : List of symbols
    path    : Path to a directory where results should be stored,
              file name is path/TICKER.pkl; default is $PWD/data
    min_len : Do not load stocks with fewer than min_len data points

    Outputs:
    stock_data : Pandas DataFrame as described above
    """
    if path is None:
        path = _DATA_DIR
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"Path '{path}' does not exist!")

    stock_data = pd.DataFrame()
    for t in tickers:
        try:
            with open(path / f"{t}.pkl", "rb") as f:
                new_stock_data = pickle.load(f)
        except:
            print(f"[e] could not load: {t}")
            continue
        if len(new_stock_data) < min_len:
            print(f"[i] skipping {t}, length {len(new_stock_data)} < {min_len}")
            continue
        # first pass through loop
        if stock_data.empty:
            stock_data = pd.DataFrame(
                new_stock_data.values, index=new_stock_data.index, columns=[t]
            )
        # not first pass through - add a column
        else:
            try:
                # truncate data if current symbol has less data than previous
                if (
                    new_stock_data.index[0] > stock_data.index[0]
                    or new_stock_data.index[-1] < stock_data.index[-1]
                ):
                    new_idx = stock_data.index.intersection(new_stock_data.index)
                    stock_data = stock_data.reindex(new_idx)
                # append column restricting to days corresponding to stock_data
                stock_data[t] = new_stock_data.loc[stock_data.index]
            # when data has duplicate indexes (FIXME: clean up instead of skip)
            except ValueError:
                print(f"[e] value error: {t}")
                continue
            # why does this happen? (FIXME: understand, clean up instead of skip)
            except KeyError:
                print(f"[e] key error: {t}")
                continue
    return stock_data


#### Backtesting baseline strategies


class Strategies:
    """Strategies object with methods for various strategies to backtest.

    Object attributes:
    portfolio : Portfolio class instance with initial positions
    end_date  : date to stop using selected strategy
    """

    def __init__(self, p, end_date):
        """Initialize Strategies class instance.

        Arguments
        ---------
        p : Portfolio class instance
        end_date : string
            YYYY-MM-DD date to stop using selected strategy.

        Returns
        -------
        Nothing
        """

        # return if end date is prior to last activity in portfolio
        if pd.to_datetime(end_date) < p.positions.index[-1]:
            raise Exception(f"End date {end_date} < last portfolio activity date")
        self.portfolio = p
        self.end_date = end_date
        return

    def close(self, portfolio):
        """End a backtest - just adds a row in the portfolio history
        corresponding to the date a test ends.

        Arguments
        ---------
        portfolio : Portfolio class instance used for backtest

        Returns
        -------
        Nothing
        """
        # add a row at end_date if it doesn't already exist
        portfolio.buy(portfolio.positions.columns[-1], self.end_date, n_shares=0)
        return

    def hold_until(self, **args):
        """Most basic buy and hold backtesting strategy for baseline against
        index funds.

        Arguments in **args
        -------------------
        unused

        Returns
        -------
        portfolio : Portfolio class object
            Portfolio resulting from the buy-and-hold-until backtest.
        """
        # make copy of initial portfolio
        from copy import deepcopy

        portfolio = deepcopy(self.portfolio)
        # nothing to do
        self.close(portfolio)
        return portfolio

    def rnd_buy_sell(self, **args):
        """Random buy and sell backtesting strategy for baseline against
        index funds. This strategy starts at the last day of activity
        in the portfolio, moves ahead by args['day_step'] number of
        days, and buys or sells everything if a random number between
        0 and 1 exceeds args['thresh'].  It only works with one stock
        at a time.

        Arguments in **args
        -------------------
        day_step   : days to step ahead
        buy_thresh : buy as much as possible if a random number between
            0 and 1 exceeds buy_thresh
        sell_thresh : sell everything if a randon number between 0 and 1
            exceeds sell_thresh

        Returns
        -------
        portfolio : Portfolio class object
            Portfolio resulting from the random buy and sell backtest.
        """
        # make copy of initial portfolio
        from copy import deepcopy

        portfolio = deepcopy(self.portfolio)
        # unpack args
        try:
            day_step = args["day_step"]
            buy_thresh = args["buy_thresh"]
            sell_thresh = args["sell_thresh"]
        except:
            print(
                "[e] rnd_buy_sell() has required args day_step, buy_thresh, sell_thresh"
            )
            return
        # initialization
        # get first stock in portfolio - this is the one this strategy will use
        symbol = portfolio.positions.columns[1]
        # create objects for doing time arithmetic
        day_step_t = pd.DateOffset(days=day_step)
        end_date_t = pd.to_datetime(self.end_date)
        # prepare to enter loop
        cur_day_t = portfolio.positions.index[-1] + day_step_t
        next_action = "sell"
        # loop, incrementing by day_step until surpass end_date
        while cur_day_t < end_date_t:
            if next_action == "sell" and np.random.random() > sell_thresh:
                portfolio.sell(symbol, cur_day_t)
                next_action = "buy"
            elif np.random.random() > buy_thresh:
                portfolio.buy(symbol, cur_day_t)
                next_action = "sell"
            cur_day_t += day_step_t
        # done
        self.close(portfolio)
        return portfolio

    def cycles(
        self, method="absolute", buy_thresh=4, sell_thresh=4, min_len_factor=2.0, **args
    ):
        """Cycles backtesting strategy.

        Arguments
        ---------
        method : string
            "relative" means thresh_pct of total number of eligible cycles
            (eligible as in period is > min_len_factor * predict_days) must
            signal buy or sell in order to buy or sell, "absolute" for thresh
            eligible cycles must signal buy or sell.
        buy_thresh : float
            For "relative" method, this is the required percentage of
            eligible cycles with mins in the prediction window or
            increasing through the whole prediction window in order
            to buy.
            For "absolute" method, this is the required number of cycles
            with mins in the prediction window in order to buy.
        sell_thresh : float
            For "relative" method, this is the required percentage of
            eligible cycles with maxes in the prediction window or
            decreasing through the whole prediction window in order
            to sell.
            For "absolute" method, this is the required number of cycles
            with maxes in the prediction window in order to sell.
        min_len_factor: float
            Only cycles with period > min_len_factor * predict_days are
            considered for determining buy or sell signal, must be >= 2
            to exclude cycles that predict both buy and sell in the
            prediction window. Cycles that have long enough period are
            the eligible cycles referenced above in buy_thresh, sell_thresh.
        Arguments in **args
        -------------------
        rel_thresh : integer
            Used for "relative" method - this many cycles must be eligible
            in order to check for relative buy_thresh or sell_thresh;
            defaults to 4 if unset.
        train_days : integer
            Number of training days to use for finding cycles.
        predict_days : integer
            Number of days to look ahead for buy or sell determination.
        max_rank : integer
            Number of cycles to fit to the signal.

        Returns
        -------
        portfolio : Portfolio class object
            Portfolio resulting from the specified cycles backtest.
        """
        # Make copy of initial portfolio
        from copy import deepcopy

        p = deepcopy(self.portfolio)
        # Unpack args
        try:
            train_days = args["train_days"]
            predict_days = args["predict_days"]
            max_rank = args["max_rank"]
            if method != "relative" and method != "absolute":
                print(f"[w] method {method} unknown - using absolute")
        except:
            raise Exception(
                "[e] cycles() has required args train_days, predict_days, max_rank"
            )

        # Initialize
        if min_len_factor < 2:
            raise Exception(f"[e] min_len_factor {min_len_factor} must be 2 or larger")
        if "rel_thresh" in args.keys():
            rel_thresh = args["rel_thresh"]
        else:
            rel_thresh = 4
        symbol = p.positions.columns[-1]
        raw_data = p.stock_data[symbol]
        close_price = raw_data.values
        t = raw_data.index

        # Actual dates
        t_t = np.asarray((t[:-1] - t[0]) / pd.to_timedelta("1Day"))

        # Compute signal
        y = np.diff(np.log(close_price))

        # Equally-spaced intervals
        x = np.arange(len(y))

        # Initialize indices
        t_0 = max(np.where(t == p.positions.index[-1])[0][0] - train_days, 0)
        t_n = t_0 + train_days
        p_0 = t_0 + train_days
        p_n = t_0 + train_days + predict_days

        # FIXME: stop when p.stop_date is surpassed
        while p_n + predict_days < len(t) and t[p_n + predict_days] < pd.to_datetime(
            self.end_date
        ):
            # Extract indices and signal, removing the mean
            x_ = x[t_0:t_n]
            y_ = y[t_0:t_n] - y[t_0:t_n].mean()

            # Here we create a new MatchCycles instance using a single-precision dictionary
            match = MatchCycles(t=x_, N_df=4.0)

            # Run it through matching pursuit.  (I removed the D0 and ws arguments as these are now
            # stored in the match object.)
            (D, a, fs, r_norms) = orthogonal_matching_pursuit(
                y_, match, max_rank=max_rank
            )

            sell = buy = 0
            num_pass = 0
            for i in range(len(fs)):
                w = fs[i][1]
                period = 2 * np.pi / w
                if period < min_len_factor * predict_days:
                    # print(f"cycle {i} too small (period {period} days)")
                    continue
                else:
                    num_pass += 1
                    Ft = match.eval(np.linspace(p_0, p_n, 100), fs)
                    deriv = np.diff(Ft[:, i : i + 1] @ a[i : i + 1])
                    asc = np.any(np.where(deriv > 0))
                    desc = np.any(np.where(deriv < 0))
                    # look for max or descending through whole window
                    if (deriv[0] > 0 and desc) or (deriv[0] < 0 and not asc):
                        sell += 1
                    # look for min or ascending through whole window
                    elif (deriv[0] < 0 and asc) or (deriv[0] > 0 and not desc):
                        buy += 1
            # test if criteria are met to check for buy or sell signal
            buy_signal = sell_signal = False
            if method == "relative" and num_pass >= rel_thresh:
                if buy / num_pass >= buy_thresh:
                    buy_signal = True
                if sell / num_pass >= sell_thresh:
                    sell_signal = True
            elif method == "absolute":
                if buy >= buy_thresh:
                    buy_signal = True
                if sell >= sell_thresh:
                    sell_signal = True
            if p.shares_owned > 0:
                if sell_signal:
                    print(
                        f"({t[t_n]}) num_pass: {num_pass}, sell: {sell}, threshold: {sell_thresh} - selling"
                    )
                    p.sell(symbol, t[p_n])
            else:
                if buy_signal:
                    print(
                        f"({t[t_n]}) num_pass: {num_pass}, buy: {buy}, threshold: {buy_thresh} - buying"
                    )
                    p.buy(symbol, t[p_n])
            t_0 += predict_days
            t_n += predict_days
            p_0 += predict_days
            p_n += predict_days
            # if t_0 // (250 * predict_days) == (t_0 / (250*predict_days)):
            # print(f"processing date: {t_0}:{t[t_0]}")

        self.close(p)
        return p


######################################################################
# Matching Pursuit
def match(s):
    """Return the best normalized match `(d, a, f_d)` the signal `s`.

    This defines the required interface and behavior for the `match` function used in
    `matching_pursuit()` and `orthogonal_matching_pursuit()`. Here `a*d` should be the
    closest match to the signal `s` over states `d` in the dictionary in the sense of
    maximizing `braket(d, s)/norm(d)` where `braket` is an appropriate inner-product
    (usually just the dot product) and `norm(d) = sqrt(braket(d, d))`.

    For cycles, this functionality can be provided by making instances of the
    `MatchCycles` class.

    Arguments
    ---------
    s : vector
       Signal to match.
    initialize : bool
       If `True`, then reset the counter.  Otherwise, the function should never return
       keep track of all calls and never return a duplicate match.

    Returns
    -------
    d : array
       Best normalized match to `s` in the dictionary.
    a : float or complex
       Projection of `s` along `d` so that `a*d` is the best approximation to `s` over
       states `d` in the dictionary.
    f_d : Any
       Information used to extrapolate to the domain of interest.  Could be a set of
       functions, or information such as the amplitude, frequency, and phase. This is
       not used by the algorithm, but returned so that the signal can be extrapolated.
       If these are functions, then summing over `a*f_d(t)` should give the
       extrapolator.

    """
    raise NotImplementedError


class MatchCycles(object):
    """Class for matching periodic cycles.

    This class takes a series of "times" `t` and prepares a dictionary for matching
    signals and residuals.  The main purposes is the `__call__()` function which allows
    MatchCycles instances to be used with the matching pursuit algorithms.

    Attributes
    ----------
    t : array
       Array of "times" at which the signal to be matched will be sampled.
       (This is typically an array of date objects, but there is no formal requirement).
    N_df : float
       The frequency interval will be `df = 1/T/N_df` where `T` is the maximum time
       interval.  This places `N_df` points in each `sinc()` cycle.
    optimize_tol : None, float
       If not `None`, then perform a two-stage match: first use the dictionary to find
       the best interval, then run a local optimizer to obtain the best match with this
       tolerance on the frequency.  This essentially makes the dictionary hugely
       overcomplete with `df = optimize_tol`.  The discrete dictionary, with `N_df >~ 2`
       is then used to ensure that local maxima are not missed.
    dictionary_match : bool
       If `True`, then the old algorithm is used where the best entry in the dictionary
       is the entry with the best individual sine or cosine match.  If `False`, then the
       best arbitrary phase match is used instead.
    max_mem_GB : float
       Maximum memory allowed for the dictionary.  If it would take more memory than
       this, we will compute it dynamically.
    dtype : np.dtype
       Floating point type for dictionary.  If memory is a concern, one could use
       `np.float16`, but this is a lot slower in most cases since machines are usually
       optimized for single or double precision.  Single-precision `np.float32` is
       generally fastest.  Only the dictionary matches are done in low precision.  All
       other calculations are done in double precision.
    """

    def __init__(
        self,
        t,
        N_df=4.0,
        optimize_tol=None,
        dictionary_match=False,
        max_mem_GB=2.0,
        dtype=np.float32,
    ):
        self.t = t
        self.N_df = N_df
        self.optimize_tol = optimize_tol
        self.dictionary_match = dictionary_match
        self.max_mem_GB = max_mem_GB
        self.dtype = dtype
        self.init()

    def __repr__(self):
        args = ", ".join(
            f"key={getattr(self, key)}" for key in ["t", "N_df", "max_mem_GB", "dtype"]
        )
        return f"{self.__class__}({args})"

    def init(self):
        self.T = self.t.max() - self.t.min()
        self.dt = np.diff(np.sort(self.t)).min()
        self.df = 1 / self.T / self.N_df
        f_max = 0.5 / self.dt
        Nf = max(3, np.ceil(f_max / self.df))
        self.df = f_max / Nf

        # Skip zero-frequency.  We will cover this my subtracting the mean.  (Including
        # this would require special-casing sin(0*t) = 0.)
        self.ws = 2 * np.pi * np.arange(Nf)[1:] * self.df

        # Check chunk size
        dict_size_GB = (
            2  # Sine + Cosine
            * len(self.ws)  # Angular frequencies
            * len(self.t)  # Samples
            * (np.finfo(self.dtype).bits / 8)  # Size of each item in bytes
            / 1024 ** 3  # In GB
        )

        if dict_size_GB > self.max_mem_GB:
            # We do not yet support blocking.
            raise NotImplementedError(
                f"dict size = {dict_size_GB:.4f}GB > max_mem_GB={self.max_mem_GB:.4f}"
            )

        DT = np.empty((2, len(self.ws), len(self.t)), dtype=self.dtype)
        DT_norms = np.empty((2, len(self.ws)))

        # Construct dictionary in 128MB chunks to save memory - we do the calculations in
        # full precision.
        chunk_size_B = 128 * 1024 ** 2
        Nw = int(np.floor(chunk_size_B / (len(self.t) * 16)))
        assert Nw > 1
        _nw = range(0, len(self.ws), Nw)
        _t = np.asarray(self.t)[np.newaxis, :]
        for i0, i1 in zip(_nw, list(_nw[1:]) + [None]):
            cs = np.exp(1j * self.ws[i0:i1, np.newaxis] * _t)
            for _i, _d in enumerate([cs.real, cs.imag]):
                # Iterate through cos and sin
                _norm = np.linalg.norm(_d, axis=-1)
                DT_norms[_i, i0:i1] = _norm
                DT[_i, i0:i1, :] = _d / _norm[..., np.newaxis]

        assert DT.nbytes == dict_size_GB * 1024 ** 3
        self._DT = DT

    def get_phase(self, w, s):
        """Return `(a, d, A, phi)` such that `s ~ a*d ~ A*cos(w * self.t + phi)`."""
        t = np.asarray(self.t)
        Nt = len(t)
        cs = np.exp(1j * w * t).view(dtype=float).reshape((Nt, 2))
        Q, R = np.linalg.qr(cs)
        assert Q.shape == (Nt, 2)
        assert R.shape == (2, 2)
        phi = np.linalg.solve(R, 2 * Q.T @ s)
        phi /= np.linalg.norm(phi)
        psi = cs @ phi
        phi = np.arctan2(-phi[1], phi[0])
        psi_norm = np.linalg.norm(psi)
        a = psi @ s / psi_norm
        d = psi / psi_norm
        A = 1 / psi_norm
        return (a, d, A, phi)

    Fd = namedtuple("Fd", ("A", "w", "phi"))

    def match(self, s, optimize_tol=None):
        """Return the best normalized match `(d, a, f_d)` the signal `s`."""
        _s = np.asarray(s, dtype=self.dtype)

        # This is the slow operation:
        overlaps = abs(self._DT @ _s)

        if self.dictionary_match:
            # Old algorithm.  Find the component with maximum cosine or sine match:
            _ic, _is = np.argmax(overlaps, axis=-1)
            if overlaps[0, _ic] > overlaps[1, _is]:
                _i = _ic
            else:
                _i = _is
        else:
            # New algorithm: find the best match
            oc, os = overlaps
            _i = np.argmax(oc ** 2 + os ** 2)

        # Now that we have the overlaps, we can compute the overlaps with full
        # precision.
        if optimize_tol is None:
            optimize_tol = self.optimize_tol

        if optimize_tol is not None:
            bracket = self.ws[max(0, _i - 1) : _i + 2]
            w = self.optimize(s, bracket=bracket, tol=optimize_tol)
        else:
            w = self.ws[_i]

        a, d, A, phi = self.get_phase(w, s=s)
        return (d, a, self.Fd(A, w, phi))

    # __call__ is an alias for match() so that instances can be used as functors.
    __call__ = match

    def optimize(self, s, bracket, tol, **kw):
        """Return `w`, the best matching frequency.

        Arguments
        ---------
        s : array-like
           Signal.
        bracket : (wa, wb, wc)
           Bracketing interval to pass to `minimize_scalar`.  `wb` should be the best
           match of these.
        """

        def fun(w):
            return -self.get_phase(w=w, s=s)[0]

        if len(bracket) == 3:
            # Check that we actually have a bracket
            fa, fb, fc = list(map(fun, bracket))
            if not fb < min(fa, fc):
                bracket = bracket[1:]

        self._optimize_res = res = minimize_scalar(fun, bracket=bracket, tol=tol, **kw)
        if not res.success:
            raise Exception(res.message)
        w = res.x
        return w

    def eval(self, t, f_d):
        """Return `[f(t)]` - the extrapolation functions evaluated on `t`.

        Arguments
        ---------
        t : array
           Times.
        f_d : list
           List of `f_d = (A, w, phi)` values returned by `self()[-1]`.

        Returns
        -------
        Ft : array
           Array who columns are the functions `A*cos(w*t+phi)`.  Form the rank `k`
           extrapolation as::

               Ft[:, :k] @ a[:k]

           If `self.t` is passed in, then `Ft` is the best-match dictionary `D` returned
           by the `orthogonal_matching_pursuit` algorithm.
        """
        As, ws, phis = np.asarray(f_d).T
        return As * np.cos(ws * np.asarray(t)[:, np.newaxis] + phis)


def matching_pursuit(s, match, rtol=0.05, max_rank=100, norm=np.linalg.norm):
    """Return `(D, alphas, rs, fs)` where `D @ alphas` approximates `s`.

    Arguments
    ---------
    s : array-like
       Input signal of length n.
    match : function
       Match function `match(s)` which returns the best match
       to the signal `s` normalized.
    rtol : float
       The residual will have `norm(r) < rtol*norm(s)` unless
       maximum rank reached.
    max_rank : int
       Maximum rank of solution.  Algorithm will return an approximation with at most
       this many components.
    norm : None, function
       Optional method for customizing the inner product.  If None, then we assume that
       the standard dot product and L2 norm are used.

    Returns
    -------
    D : array
       Columns are the best matches.
    alphas : array
       Vector of coefficients.
    fs : list
       List of extrapolation functions.  Dot with a to get
       the prediction.
    r_norms : array
       Vector of residual norms.
    """
    # Accumulators: (d, alpha, f, r_norm)
    # d: Matching vector
    # alpha: Corresponding coefficient
    # f: Extrapolation information
    # r_norm: Relative norms of the residual

    res = []

    # Start with the full signal as the current residual.
    r = np.copy(s)  # Ensure we have a copy as we will mutate this.
    r_norm = s_norm = norm(s)
    r_stop = rtol * s_norm

    while len(res) < max_rank and r_norm > r_stop:
        d, alpha, f = match(r)
        r_norm = norm(r)
        res.append((d, alpha, f, r_norm))
        r -= d * alpha  # Remove current vector from signal

    # Unpack results
    D, alphas, fs, r_norms = zip(*res)
    D = np.transpose(D)  # Arrange so that columns are the components.
    alphas = np.asarray(alphas)
    r_norms = np.asarray(r_norms) / s_norm  # Make these relative

    return D, alphas, fs, r_norms


def orthogonal_matching_pursuit(
    s, match, rtol=0.05, max_rank=100, inner_product=np.dot, rank_tol=1e-12
):
    """Return `(D, a, fs, r_norms)` where `D @ R` approximates `s` with upper triangular `R`.

    This is similar to matching_pursuit` but uses a QR decomposition via Gram Schmidt to
    ensure that the residuals at each step are always orthogonal to the currently
    selected subspace.

    Arguments
    ---------
    s : array-like
       Input signal of length n.
    match : function
       Match function `match(s)` which returns the best match
       to the signal `s` normalized.
    rtol : float
       The residual will have `norm(r) < rtol*norm(s)` unless
       maximum rank reached.
    max_rank : int
       Maximum rank of solution.  Algorithm will return an approximation with at most
       this many components.
    norm : None, function
       Optional method for customizing the inner product.  If None, then we assume that
       the standard dot product and L2 norm are used.
    rank_tol : float
       If the norm of the orthogonal component of the best match is less than this, then
       we bail because we cannot further reduce the residual.

    Returns
    -------
    D : array
       Columns are the best matches.
    a : array
       Vector of coefficients.
    fs : list
       List of extrapolation information.  Dot with a to get the prediction.
    r_norms : array
       Vector of residual norms.
    """
    # Accumulators: (d, alpha, f, r_norm)
    # d: Matching vector
    # alpha: Corresponding coefficient
    # f: Extrapolation information
    # r_norm: Relative norms of the residual

    res = []

    if inner_product is np.dot:
        norm = np.linalg.norm
    else:

        def norm(r):
            return math.sqrt(inner_product(r, r))

    # Start with the full signal as the current residual.
    r = np.copy(s)  # Ensure we have a copy as we will mutate this.
    r_norm = s_norm = norm(s)
    r_stop = rtol * s_norm

    # Here we work with D = [d] i.e. the rows of D are the vectors returned by match().
    # The orthonormal basis Q is such that L @ Q = D where L is a lower-triangular matrix.
    D = []
    Q = []
    L = []
    a = []

    while len(res) < max_rank and r_norm > r_stop:
        d, alpha, f = match(r)

        # Gram Schmidt orthogonalization of q
        # Invariant  d = l[:i] @ Q[:i, ...] + q
        q = np.copy(d)
        l = []
        for q_ in Q:
            l.append(inner_product(q, q_))
            q -= q_ * l[-1]

        tmp = norm(q)
        if tmp < rank_tol:
            # d depends linearly on Q... cannot reduce r
            warn(
                f"Best match already in subspace. "
                + f"Bailing with rtol={r_norm/s_norm}, rank={len(res)}."
            )
            break

        l.append(tmp)
        q /= l[-1]
        Q.append(q)
        for _l in L:
            _l.append(0)
        L.append(l)
        D.append(d)
        a.append(inner_product(q, s))

        assert np.allclose(d, np.dot(l, Q))
        assert np.allclose(D, np.dot(L, Q))
        assert np.allclose(np.eye(len(L)), Q @ np.transpose(Q))

        r_norm = norm(r)
        res.append((d, alpha, f, r_norm))
        r -= Q[-1] * inner_product(Q[-1], r)

    # Unpack results
    D, alphas, fs, r_norms = zip(*res)
    D = np.transpose(D)  # Arrange so that columns are the components.
    a = np.linalg.solve(np.transpose(L), a)
    r_norms = np.asarray(r_norms) / s_norm  # Make these relative

    MatchResults = namedtuple("MatchResults", ["D", "a", "f_d", "r_norms"])

    return MatchResults(D, a, fs, r_norms)

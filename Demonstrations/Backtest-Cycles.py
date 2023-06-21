# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# %%
# # !pip3 install shutup # Kills warning/error? messages.
# # !pip3 install yfinance
# # !pip3 install uncertainties
import shutup
shutup.please()

# %%
# %pylab inline --no-import-all
# %load_ext autoreload
from sys import path; path.append('../')
from cycles.cycles import *
import pickle

# %% [markdown]
# Load data from Yahoo Finance

# %%
update_adj_close(["^IXIC","^DJI"],path="../cycles/data")

# %%
start_date = "1992-01-02"
p = Portfolio(start_date, 50000)

# %% [markdown]
# Buy as many shares of ^IXIC as possible on `start_date`

# %%
symbol = "^IXIC"
p.buy(symbol, start_date)

# %% [markdown]
# Initialize a test object, which will test from the last activity date of the portfolio (currently `start_date`) through 2021-02-26

# %%
end_date = "2021-02-26"
test = Strategies(p, end_date)

# %% [markdown]
# Run a cycles test using "absolute" cycles requirement

# %%
train_days = 60 # number of days to train on
predict_days = 6 # prediction window
max_rank = 30 # number of cycles to fit to training data
method = "absolute" # absolute cycles requirement
sell_thresh = 4 # required number of cycles with maxes in prediction window to buy
buy_thresh = 4 # required number of cycles with mins in prediction window to sell
min_len_factor = 2 # eligible cycles have period > min_len_factor * predict_days

print("processing:",symbol)
p_new = test.cycles(\
    train_days=train_days,
    predict_days=predict_days,
    max_rank=max_rank,
    method=method,
    min_len_factor=min_len_factor,
    sell_thresh=sell_thresh,
    buy_thresh=buy_thresh
    )
print("close value:",p_new.value())

# %%
p_new.positions

# %% [markdown]
# Plot showing price history with buy and sell events

# %%
data = read_adj_close([symbol])
data_dates = data.index

s_idx = np.where(data_dates == start_date)[0][0]
e_idx = np.where(data_dates == end_date)[0][0]

plt.plot(data[symbol].index[s_idx:e_idx],data[symbol].values[s_idx:e_idx])
for d in p_new.positions.index[np.where(p_new.positions[symbol].values == 0)]:
    plt.axvline(d, c='r')

for d in p_new.positions.index[np.where(p_new.positions[symbol].values != 0)]:
    plt.axvline(d, c='g')

plt.axvline(pd.to_datetime(start_date), c='g', label='buy')
plt.axvline(pd.to_datetime(end_date), c='r', label='sell')
plt.xlabel('year')
plt.ylabel('closing price')
plt.legend(loc=2)
plt.title(f"cycles: {symbol}")

# %%

# %%

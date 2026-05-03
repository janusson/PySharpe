import pandas as pd
import json
import numpy as np
import cvxpy as cp
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage

with open('portfolio_config.json', 'r') as f:
    config = json.load(f)

df = pd.read_csv('data/exports/canadian_etfs_collated.csv', index_col=0, parse_dates=True).dropna()

mu = ema_historical_return(df)
cov = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, cov)

ef.add_sector_constraints(
    config['geo_mapping'],
    sector_lower=config['constraints']['geo_lower_bounds'],
    sector_upper=config['constraints']['geo_upper_bounds'],
)

ef.add_constraint(
    lambda w: w @ np.array([config['mer_mapping'].get(t, 0.0) for t in ef.tickers]) - config['constraints']['max_portfolio_mer'] * cp.sum(w) <= 0
)

try:
    ef.max_sharpe()
    print("Cleaned weights:", ef.clean_weights())
except Exception as e:
    print("FAILED:", e)


from src.core.portfolios import Optimizer
import yfinance as yf


prices = yf.download("SPY, AGG, GSG, TLT")["Adj Close"]

opt = Optimizer.from_prices(prices=prices)

opt.minimized_volatility()
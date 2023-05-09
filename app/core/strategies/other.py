

class HierarchicalEqualRiskContribution(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        weights = opt.hierarchical_equal_risk_contribution()
        return weights


class HierarchicalRiskParity(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        return opt.hierarchical_risk_parity()


class RiskParity(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.reb_prices, halflife=63
        )
        correlation_matrix = estimators.to_correlation_matrix(
            prices=self.reb_prices, halflife=63
        )

        opt = optimizer.Optimizer(
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            **kwargs,
        )
        return opt.risk_parity()


class MaxSharpe(Strategy):
    def rebalance(self, **kwargs):
        prices = self.prices.loc[: self.date].iloc[-252:]
        cov = prices.pct_change().fillna(0).cov() * (252**0.5)
        er = prices.pct_change().fillna(0).mean() * (252)
        opt = optimizer.Optimizer(expected_returns=er, covariance_matrix=cov, **kwargs)
        return opt.maximized_sharpe_ratio()


class InverseVariance(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.inverse_variance()


class TargetVol(Strategy):
    def rebalance(self, **kwargs):
        covariance_matrix = estimators.to_covariance_matrix(
            prices=self.prices.loc[: self.date].iloc[-252:]
        )
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.minimized_volatility()


class Momentum(Strategy):
    def rebalance(self, **kwargs) -> Optional[pd.Series]:
        prices = self.prices.loc[: self.date]
        momentum_1y = prices.iloc[-1] / prices.iloc[-21]

        momentum_1y = momentum_1y.dropna().nsmallest(6)

        prices = prices[momentum_1y.index]
        covariance_matrix = estimators.to_covariance_matrix(prices=prices, halflife=21)
        opt = optimizer.Optimizer(covariance_matrix=covariance_matrix, **kwargs)
        return opt.hierarchical_equal_risk_contribution()


class MeanReversion(Strategy):
    """
    What is Mean Reversion?
        According to Investopedia, mean reversion, or reversion to the mean, is
        a theory used in finance (rooted in a concept well known as regression
        towards the mean) that suggests that asset price volatility and
        historical returns eventually will revert to the long-run mean or
        average level of the entire dataset. Mean is the average price and
        reversion means to return to, so mean reversion means “return to the
        average price”.

        While an assets price tends to revert to the average over time, this
        does not always mean or guarantee that the price will go back to the
        mean, nor does it mean that the price will rise to the mean.

    What Is A Mean Reversion Trading Strategy ?
        A mean reversion trading strategy is a trading strategy that focuses on
        when a security moves too far away from some average. The theory is that
        the price will move back toward that average at some point in time.
        There are many different ways to look at this strategy, for example by
        using linear regression, RSI, Bollinger Bands, standard deviation,
        moving averages etc. The question is how far from the average / mean is
        too far ?
    """

    def rebalance(self, **kwargs) -> pd.Series:
        return super().rebalance(**kwargs)

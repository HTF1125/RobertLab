from typing import Optional, Union, Tuple, List
from dateutil import parser
import numpy as np
import pandas as pd


def to_pri_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    def to_pri_return(price: pd.Series) -> float:
        return price.dropna().pct_change().fillna(0)

    return prices.apply(to_pri_return)


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return to_pri_returns(prices=prices).apply(np.log1p)


def to_num_years(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """

    def to_num_year(price) -> float:
        p = price.dropna()
        start = parser.parse(str(p.index[0]))
        end = parser.parse(str(p.index[-1]))
        return (end - start).days / 365.0

    return prices.apply(to_num_year, axis=0)


def to_num_bars(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        float: _description_
    """

    def to_num_bar(price) -> float:
        return len(price.dropna())

    return prices.apply(to_num_bar, axis=0)


def to_ann_factors(prices: pd.DataFrame) -> float:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_num_bars(prices=prices) / to_num_years(prices=prices)


def to_cum_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_pri_returns(prices=prices).add(1).prod() - 1


def to_ann_returns(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return (
        to_pri_returns(prices=prices).add(1).prod() ** (1 / to_num_years(prices=prices))
        - 1
    )


def to_ann_variances(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)
    return to_pri_returns(prices=prices).var() * to_ann_factors(prices=prices)


def to_ann_volatilites(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return to_ann_variances(prices=prices, ann_factors=ann_factors).apply(np.sqrt)


def to_ann_semi_variances(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    pri_returns = to_pri_returns(prices=prices)
    positive_pri_returns = pri_returns[pri_returns >= 0]
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)
    return positive_pri_returns.var() * ann_factors


def to_ann_semi_volatilities(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    return to_ann_semi_variances(prices=prices, ann_factors=ann_factors) ** 0.5


def to_drawdown(
    prices: pd.DataFrame,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (Optional[int], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    if window:
        return prices / prices.rolling(window=window, min_periods=min_periods).max() - 1
    return prices / prices.expanding(min_periods=min_periods or 1).max() - 1


def to_max_drawdown(
    prices: pd.DataFrame,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        window (Optional[int], optional): _description_. Defaults to None.
        min_periods (Optional[int], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    return to_drawdown(prices=prices, window=window, min_periods=min_periods).min()


def to_sharpe_ratios(
    prices: pd.DataFrame, risk_free: float = 0.0, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        risk_free (float, optional): _description_. Defaults to 0..
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    excess_returns = to_ann_returns(prices=prices) - risk_free
    return excess_returns / to_ann_volatilites(prices=prices, ann_factors=ann_factors)


def to_sortino_ratios(
    prices: pd.DataFrame, ann_factors: Optional[float] = None
) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        ann_factors (Optional[float], optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    if not ann_factors:
        ann_factors = to_ann_factors(prices=prices)

    ann_returns = to_ann_returns(prices=prices)
    ann_semi_volatilities = to_ann_semi_volatilities(
        prices=prices, ann_factors=ann_factors
    )

    return ann_returns / ann_semi_volatilities


def to_tail_ratios(prices: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        pd.Series: _description_
    """

    def to_tail_ratio(pri_return: pd.Series, alpha: float) -> float:

        r = pri_return.dropna()
        return -r.quantile(q=alpha) / r.quantile(q=1 - alpha)

    return to_pri_returns(prices=prices).apply(to_tail_ratio, alpha=alpha)


def to_skewnesses(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """

    def to_skewness(pri_return: pd.Series) -> float:

        return pri_return.dropna().skew()

    return to_pri_returns(prices=prices).apply(to_skewness)


def to_kurtosises(prices: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """

    def to_kurtosis(pri_return: pd.Series) -> float:

        return pri_return.dropna().kurt()

    return to_pri_returns(prices=prices).apply(to_kurtosis)


def to_value_at_risks(prices: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        pd.Series: _description_
    """

    def to_value_at_risk(pri_return: pd.Series, alpha: float) -> float:

        r = pri_return.dropna()
        return r.quantile(q=alpha)

    return to_pri_returns(prices=prices).apply(to_value_at_risk, alpha=alpha)


def to_expected_shortfalls(prices: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    def to_expected_shortfall(pri_return: pd.Series, alpha: float) -> float:

        r = pri_return.dropna()
        var = r.quantile(q=alpha)
        return r[r <= var].mean()

    return to_pri_returns(prices=prices).apply(to_expected_shortfall, alpha=alpha)


def cov_to_corr(covariance_matrix: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        covariance_matrix (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    vol = np.sqrt(np.diag(covariance_matrix))
    corr = covariance_matrix / np.outer(vol, vol)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def recursive_bisection(sorted_tree) -> List[Tuple[List[int], List[int]]]:
    """_summary_

    Args:
        sorted_tree (_type_): _description_

    Returns:
        List[Tuple[List[int], List[int]]]: _description_
    """

    if len(sorted_tree) < 3:
        return

    num = len(sorted_tree)
    bis = int(num / 2)
    left = sorted_tree[0:bis]
    right = sorted_tree[bis:]
    if len(left) > 2 and len(right) > 2:
        return [(left, right), recursive_bisection(left), recursive_bisection(right)]
    return (left, right)


def get_cluster_assets(clusters, node, num_assets) -> List:
    """_summary_

    Args:
        clusters (_type_): _description_
        node (_type_): _description_
        num_assets (_type_): _description_

    Returns:
        List: _description_
    """
    if node < num_assets:
        return [int(node)]
    row = clusters[int(node - num_assets)]
    return get_cluster_assets(clusters, row[0], num_assets) + get_cluster_assets(
        clusters, row[1], num_assets
    )


def to_covariance_matrix(
    prices: pd.DataFrame,
    ann_factors: Optional[Union[int, float, pd.Series]] = None,
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> pd.DataFrame:

    pri_returns = to_pri_returns(prices=prices)
    if ann_factors is None:
        ann_factors = to_ann_factors(prices=prices)

    alpha = exponential_alpha(com=com, span=span, halflife=halflife)

    if alpha is None:
        return pri_returns.cov() * ann_factors

    exp_covariance_matrix = (
        pri_returns.ewm(alpha=alpha).cov().unstack().iloc[-1].unstack() * ann_factors
    )

    return exp_covariance_matrix.loc[prices.columns, prices.columns]


def exponential_alpha(
    com: Optional[float] = None,
    span: Optional[float] = None,
    halflife: Optional[float] = None,
) -> float:
    """_summary_

    Args:
        com (Optional[float], optional): _description_. Defaults to None.
        span (Optional[float], optional): _description_. Defaults to None.
        halflife (Optional[float], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
    if com:
        return 1 / (1 + com)
    if span:
        return 2 / (span + 1)
    if halflife:
        return 1 - np.exp(-np.log(2) / halflife)

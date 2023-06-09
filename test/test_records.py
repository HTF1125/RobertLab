# Generated by CodiumAI
from src.backend.core.strategies.book import Records
import pandas as pd


"""
Code Analysis

Main functionalities:
The Book class is designed to represent a book of investments, with methods to track the value, cash, shares, weights, and capital of the book over time. It also includes a Records class to store and retrieve historical data about the book's performance, cash, allocations, weights, and trades.

Methods:
- new(): a class method that creates a new instance of the Book class with specified inception date and initial investment
- __init__(): the constructor method for the Book class, which initializes the fields of the book and creates a new instance of the Records class to store historical data
- dict(): a method that returns a dictionary representation of the book's current state
- date getter and setter: getter and setter methods for the date field of the book, which also update the historical data in the Records class

Fields:
- value: the current value of the book
- cash: the current amount of cash in the book
- shares: a pandas Series representing the number of shares held in the book
- weights: a pandas Series representing the weights of each investment in the book
- capitals: a pandas Series representing the capital invested in each investment in the book
- records: an instance of the Records class to store historical data about the book's performance, cash, allocations, weights, and trades
- date: a pandas Timestamp representing the current date of the book
"""


class TestRecords:
    # Tests that the 'performance' property returns a pd.Series object with the correct name and index
    def test_performance_returns_correct_series(self):
        records = Records(value={"2022-01-01": 1000, "2022-01-02": 2000})
        assert records.performance.name == "performance"
        assert records.performance.index[0] == pd.Timestamp("2022-01-01")
        assert records.performance.index[1] == pd.Timestamp("2022-01-02")
        assert records.performance[0] == 1000
        assert records.performance[1] == 2000

    # Tests that the 'cash' property returns a pd.Series object with the correct name
    def test_cash_returns_correct_series(self):
        records = Records(cash={"2022-01-01": 1000, "2022-01-02": 2000})
        assert records.cash.name == "cash"
        assert records.cash.index[0] == pd.Timestamp("2022-01-01")
        assert records.cash.index[1] == pd.Timestamp("2022-01-02")
        assert records.cash[0] == 1000
        assert records.cash[1] == 2000

    # Tests that the 'allocations' property returns a pd.DataFrame object with the correct shape
    def test_allocations_returns_correct_dataframe(self):
        records = Records(
            allocations={
                "2022-01-01": {"AAPL": 0.5, "GOOG": 0.5},
                "2022-01-02": {"AAPL": 0.4, "GOOG": 0.6},
            }
        )
        assert records.allocations.shape == (2, 2)
        assert records.allocations.index[0] == pd.Timestamp("2022-01-01")
        assert records.allocations.index[1] == pd.Timestamp("2022-01-02")
        assert records.allocations.columns[0] == "AAPL"
        assert records.allocations.columns[1] == "GOOG"
        assert records.allocations.iloc[0, 0] == 0.5
        assert records.allocations.iloc[1, 1] == 0.6

    # Tests that the 'weights' property returns a pd.DataFrame object with the correct shape
    def test_weights_returns_correct_dataframe(self):
        records = Records(
            weights={
                "2022-01-01": {"AAPL": 0.5, "GOOG": 0.5},
                "2022-01-02": {"AAPL": 0.4, "GOOG": 0.6},
            }
        )
        assert records.weights.shape == (2, 2)
        assert records.weights.index[0] == pd.Timestamp("2022-01-01")
        assert records.weights.index[1] == pd.Timestamp("2022-01-02")
        assert records.weights.columns[0] == "AAPL"
        assert records.weights.columns[1] == "GOOG"
        assert records.weights.iloc[0, 0] == 0.5
        assert records.weights.iloc[1, 1] == 0.6

    # Tests that the 'trades' property returns a pd.DataFrame object with the correct shape
    def test_trades_returns_correct_dataframe(self):
        records = Records(
            trades={
                "2022-01-01": {"AAPL": 100, "GOOG": -50},
                "2022-01-02": {"AAPL": -50, "GOOG": 100},
            }
        )
        assert records.trades.shape == (2, 2)
        assert records.trades.index[0] == pd.Timestamp("2022-01-01")
        assert records.trades.index[1] == pd.Timestamp("2022-01-02")
        assert records.trades.columns[0] == "AAPL"
        assert records.trades.columns[1] == "GOOG"
        assert records.trades.iloc[0, 0] == 100
        assert records.trades.iloc[1, 1] == 100

    # Tests that the 'performance' property returns an empty pd.Series object when 'value' is empty
    def test_performance_returns_empty_series_when_value_empty(self):
        records = Records()
        assert records.performance.empty

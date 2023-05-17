import concurrent.futures
import yfinance as yf
import timeit

def download_ticker(ticker):
    return yf.download(ticker, progress=False)

def main():
    tickers = ["SPY"] * 50

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(download_ticker, tickers)

    return list(results)

if __name__ == "__main__":
    start_time = timeit.default_timer()
    data = main()
    end_time = timeit.default_timer()
    print(data)
    print(f"Runtime: {end_time - start_time} seconds")

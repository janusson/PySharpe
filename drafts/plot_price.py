# plot_price.py
# Plot closing price of a user-defined ticker symbol
import matplotlib.pyplot as plt
import pandas as pd
from pysharpe.data_collector import SecurityDataCollector
import os

# Plotting Functions
def plot_price_history(ticker: str):
    """Plot closing price data.

    Args:
        ticker (str): Ticker symbol to plot
        data_history (dataframe): Pandas dataframe of price data

    Returns:
        None: Plots ticker symbol price data with matplotlib
    """
    csv_path = f'./data/price_hist/{ticker}_hist.csv'
    try:
        # Read saved data file for plotting
        data_history = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Download if not present, then read
        os.makedirs('./data/price_hist', exist_ok=True)
        SecurityDataCollector(ticker).download_price_hist()
        data_history = pd.read_csv(csv_path)
    plt.plot(data_history["Close"])
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()
    return print(f'Plotting {ticker} price...')

def main(ticker: str = "MSFT"):
    plot_price_history(ticker)
    return None

if __name__ == '__main__':
    main()

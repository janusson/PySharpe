# ./financio/data_collector.py
import yfinance as yf
import json
import os
import pandas as pd
from typing import Set, List

#? Utility functions and logging
import logging
import datetime

def setup_logging():
    '''Basic logging configuration and setup.'''
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d_log.lo")
    logging.basicConfig(level=logging.INFO, filename=f'./logs/{log_filename}',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# Distinct portfolios to be optimized
PORTFOLIO_DIR = r'./data/portfolio/'
PORTFOLIO_FILES = [os.path.join(PORTFOLIO_DIR, f) for f in os.listdir(PORTFOLIO_DIR) if f.endswith('.csv')]

logger.info(f"Portfolio directory: {PORTFOLIO_DIR}")
logger.info(f"Portfolio files: {PORTFOLIO_FILES}")

def download_portfolio_prices(tickers: Set[str]) -> None:
    """Download price history for a set of tickers."""
    for ticker in tickers:
        logger.info(f"Downloading price history for: {ticker}")
        print(f"Downloading price history for: {ticker}")
        try:
            yf.Ticker(ticker).history(period="max").to_csv(
                f'./data/price_hist/{ticker}_hist.csv')
        except Exception as e:
            logger.error(f'Error downloading price history for {ticker}: {e}')

class SecurityDataCollector:
    """Provides methods to collect information about a security from Yahoo Finance when a ticker symbol is provided."""

    def __init__(self, ticker: str):
        """Initialize class and create YF Ticker object with provided symbol."""
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)

    def get_company_name(self) -> str:
        """Fetch the company name from Yahoo Finance.

        Returns:
            str: Name of the company.
        """
        try:
            self.company_name = self.yf_ticker.info['shortName']
        except KeyError as e:
            try:
                logger.error(f'No company shortName for {self.ticker}: {e}')
                self.company_name = self.yf_ticker.info['longName']
            except KeyError as e:
                logger.error(f'No company longName for {self.ticker}: {e}')
                self.company_name = "Unknown"
        return self.company_name

    def get_company_info(self) -> dict:
        """Fetch company information from Yahoo Finance.

        Returns:
            dict: Dictionary of company information.
        """
        self.company_info = self.yf_ticker.info
        return self.company_info

    def download_info(self):
        """Download company info as a JSON file."""
        os.makedirs('./data/info', exist_ok=True)
        with open(f'./data/info/{self.get_company_name()} summary.json', 'w') as outfile:
            json.dump(self.get_company_info(), outfile, indent=4)

    def get_news(self) -> list:
        """Return list of company news from Yahoo Finance.

        Returns:
            list: List of dicts of company news.
        """
        try:
            self.news = self.yf_ticker.news
        except Exception as e:
            logger.error(f'Error retrieving news for {self.ticker}: {e}')
        return self.news

    def get_options(self):
        """Return options information."""
        try:
            self.options = self.yf_ticker.options
        except Exception as e:
            logger.error(f'Error retrieving options for {self.ticker}: {e}')

    def get_earnings_dates(self):
        """Return earnings release dates."""
        try:
            self.earnings_dates = self.yf_ticker.earnings_dates
        except Exception as e:
            logger.error(f'Error retrieving earnings for {self.ticker}: {e}')

    def get_recommendations(self):
        """Return analyst recommendations."""
        try:
            self.recommendations = self.yf_ticker.recommendations
            self.recommendations_summary = self.yf_ticker.recommendations_summary
            self.upgrades_downgrades = self.yf_ticker.upgrades_downgrades
        except Exception as e:
            logger.error(f'Error retrieving recommendations for {self.ticker}: {e}')

    def get_holders(self):
        """Return institutional, mutual fund, and insider holders."""
        try:
            self.major_holders = self.yf_ticker.major_holders
            self.inst_holders = self.yf_ticker.institutional_holders
            self.mutualfund_holders = self.yf_ticker.mutualfund_holders
            self.insider_purchases = self.yf_ticker.insider_purchases
            self.insider_roster_holders = self.yf_ticker.insider_roster_holders
        except Exception as e:
            logger.error(f'Error retrieving holders of {self.ticker}: {e}')

    def get_financials(self) -> pd.DataFrame:
        """Fetch financial data from Yahoo Finance and return as a DataFrame.

        Returns:
            pd.DataFrame: Financial statements of the company.
        """
        try:
            self.financial_data = {
                'Income Statement': self.yf_ticker.financials,
                'Quarterly Income Statement': self.yf_ticker.quarterly_financials,
                'Balance Sheet': self.yf_ticker.balance_sheet,
                'Quarterly Balance Sheet': self.yf_ticker.quarterly_balance_sheet,
                'Cash Flow': self.yf_ticker.cashflow,
                'Quarterly Cash Flow': self.yf_ticker.quarterly_cashflow
            }
        except Exception as e:
            logger.error(f'Error retrieving financials for {self.ticker}: {e}')
            self.financial_data = {}

        return pd.concat(self.financial_data, axis=1) if self.financial_data else pd.DataFrame()

    def get_actions(self):
        """Return security actions such as dividends and splits."""
        try:
            self.actions = self.yf_ticker.actions
            self.dividends = self.yf_ticker.dividends
            self.splits = self.yf_ticker.splits
            self.capital_gains = self.yf_ticker.capital_gains  # funds only
            self.shares_hist = self.yf_ticker.get_shares_full()  # historical share count
        except Exception as e:
            logger.error(f'Error retrieving actions for {self.ticker}: {e}')

    def get_summary(self) -> dict:
        """Retrieves and returns a dictionary of key company information.

        Returns:
            dict: Summary of key company information.
        """
        if not hasattr(self, 'company_info'):
            self.get_company_info()
        summary_keys = ['longName', 'country', 'industry', 'sector', 'overallRisk', 'dividendYield', 'previousClose', 'payoutRatio', 'currency', 'forwardPE', 'volume', 'marketCap', 'priceToBook', 'forwardEps', 'pegRatio', 'symbol', 'currentPrice',
                        'recommendationMean', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'financialCurrency', 'trailingPegRatio']
        summary_dict = {key: self.company_info.get(key) for key in summary_keys}
        return summary_dict

    def download_price_hist(self):
        """Download ticker closing price history as a CSV file."""
        try:
            self.yf_ticker.history(period="max").to_csv(
                f'./data/price_hist/{self.ticker}_hist.csv')
        except Exception as e:
            logger.error(f'Error downloading price history for {self.ticker}: {e}')

# Fix for PortfolioTickerReader initialization
class PortfolioTickerReader:
    """PortfolioTickerReader reads ticker symbols from CSV portfolio files to populate sets."""

    def __init__(self):
        """Initialize class with empty ticker symbol sets."""
        self.portfolio_tickers: dict[str, set[str]] = {}
        portfolio_files = get_csv_file_paths(os.path.abspath(PORTFOLIO_DIR))  # Ensure absolute path

        if not portfolio_files:
            logger.error(f"No portfolio files found in directory: {PORTFOLIO_DIR}")
            return  # Exit early if no files are found

        for portfolio_file in portfolio_files:
            portfolio_name = os.path.splitext(os.path.basename(portfolio_file))[0]
            logger.info(f"Processing portfolio file: {portfolio_file}, portfolio name: {portfolio_name}")
            self.portfolio_tickers[portfolio_name] = set()
            try:
                with open(portfolio_file, 'r', encoding='utf-8') as csv_file:
                    for line in csv_file:
                        ticker_symbol = line.strip()
                        if ticker_symbol:
                            self.portfolio_tickers[portfolio_name].add(ticker_symbol)
                if not self.portfolio_tickers[portfolio_name]:
                    logger.warning(f"No tickers found in portfolio file: {portfolio_file}")
                else:
                    logger.info(f"Tickers for {portfolio_name}: {self.portfolio_tickers[portfolio_name]}")
            except FileNotFoundError as e:
                logger.error(f'Error reading file: {e}')
            except Exception as e:
                logger.error(f'Unknown error with PortfolioTickerReader: {e}')

    def get_portfolio_tickers(self, portfolio_name: str) -> Set[str]:
        """Returns set of retrieved tickers by portfolio.

        Args:
            portfolio_name (string): Name of the portfolio.

        Returns:
            set: Set of ticker symbols in portfolio.
        """
        logger.info(f"Retrieving tickers for portfolio: {portfolio_name}")
        return self.portfolio_tickers.get(portfolio_name, set())

def get_csv_file_paths(directory: str) -> List[str]:
    """Get all CSV file paths in the specified directory."""
    try:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {directory}. Error details: {e}")
        return []

'''def get_csv_file_paths(directory_path: str) -> List[str]:
    """
    Retrieves the relative paths of all CSV files in the specified directory.

    Args:
        directory (str): Path to the directory containing CSV files.

    Returns:
        list: List of relative file paths for CSV files.
    """
    csv_files = []
    for root, dirs, files, in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files'''

def process_portfolio(portfolio: str) -> Set[str]:
    """Read tickers from portfolio and download price history for each ticker symbol.

    Args:
        portfolio (str): Full path to the portfolio file.

    Returns:
        set: Set of ticker symbols in portfolio.
    """
    portfolio_name = os.path.basename(portfolio).split('.')[0]  # Extract portfolio name from file path
    ticker_reader = PortfolioTickerReader()
    tickers = ticker_reader.get_portfolio_tickers(portfolio_name)  # Use the correct portfolio name
    print(f"Retrieving tickers for portfolio: {portfolio_name}")
    print(f"Tickers: {tickers}")
    
    if tickers:
        logger.info(f"Processing portfolio: {portfolio_name}")
        logger.info(f"Tickers to download: {', '.join(tickers)}")
        print(f"Processing portfolio: {portfolio_name}")
        print(f"Tickers to download: {', '.join(tickers)}")
        download_portfolio_prices(tickers)
    else:
        logger.warning(f'No tickers found for portfolio: {portfolio_name}')
        print(f'No tickers found for portfolio: {portfolio_name}')
    return tickers


def collate_prices(portfolio: str, csv_files: List[str], portfolio_tickers: Set[str]) -> pd.DataFrame:
    """Processes a list of CSV files containing stock price histories.
    Collates closing prices if present in set of portfolio tickers.
    Saves cleaned dataframe as CSV file.

    Args:
        portfolio (str): Name of the portfolio.
        csv_files (list): List of paths to price history CSV files.
        portfolio_tickers (set): List of ticker symbols to collect.

    Returns:
        DataFrame: Combined DataFrame of closing prices.
    """
    combined_df = pd.DataFrame()
    portfolio_name = portfolio.split('/')[-1].split('.')[0]  # Extract portfolio name from file path
    logger.info(f"Collating prices for portfolio: {portfolio_name}")
    print(f"Collating prices for portfolio: {portfolio_name}")
    # Ensure the exports directory exists
    os.makedirs('./data/exports', exist_ok=True)
    # Iterate through each CSV file and extract closing prices
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['Date'] = df['Date'].str.split(' ').str[0]
        df.set_index('Date', inplace=True)
        # Extract the ticker symbol from the filename
        filename = os.path.basename(csv_file)
        ticker_symbol = filename.split('_')[0]
        # Check if the ticker symbol is in the portfolio
        if ticker_symbol in portfolio_tickers:
            # Add a column to the combined DataFrame with the closing prices
            combined_df[f"{ticker_symbol}"] = df["Close"]
    # cleaned_df = combined_df.dropna()  # Remove missing values #! This is breaking something.
    cleaned_df = combined_df.dropna(axis=1, how='all') #? This fixed the problem
    # cleaned_df = combined_df.fillna(method='ffill') #? Can maybe fill or impute missing values
    # Save the cleaned DataFrame to a CSV file
    cleaned_df.to_csv(f'./data/exports/{portfolio_name}_collated.csv')
    return cleaned_df


def main():
    """Main function to download and collate portfolio prices."""
    # Get price history CSV files
    csv_list = get_csv_file_paths(r'./data/price_hist/')

    # Download and collate portfolio prices
    for portfolio in PORTFOLIO_FILES:
        tickers = process_portfolio(portfolio)
        collate_prices(portfolio, csv_list, tickers)
    return None

if __name__ == '__main__':
    main()

'''
This script takes a ticker symbol input by the user to fetch security information on Yahoo Finance.
'''
import yfinance as yf
import json
import os
import pandas as pd


class SecurityDataCollector:
    """Retrieves financial and company information from Yahoo Finance for a given ticker."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        self.company_info = self.yf_ticker.get_info()
        if not self.company_info:
            raise ValueError(f"Could not retrieve data for ticker: {ticker}")

    def get_company_name(self) -> str:
        """Fetch company name, prioritizing shortName."""
        return self.company_info.get('shortName', self.company_info.get('longName', 'Unknown'))

    def get_news(self) -> list:
        """Fetch company news."""
        try:
            news_data = self.yf_ticker.get_news()
        except Exception as e:
            print(f"Error fetching news for {self.ticker}: {e}")
            return []
        if not news_data:
            print(f"No news data available for {self.ticker}.")
            return []
        # Convert to a list of dictionaries if not already
        if isinstance(news_data, dict):
            return [news_data]
        elif isinstance(news_data, list):
            return news_data
        else:
            print(f"Unexpected news data format for {self.ticker}.")
            return []

    def get_financials(self) -> pd.DataFrame:
        """Fetch financial data as a DataFrame."""
        financial_data = {key: getattr(self.yf_ticker, key, pd.DataFrame()) for key in [
            'financials', 'quarterly_financials', 'balance_sheet', 'quarterly_balance_sheet',
            'cashflow', 'quarterly_cashflow']}
        return pd.concat(financial_data, axis=1) if financial_data else pd.DataFrame()

    def download_info(self):
        """Download company info as JSON."""
        os.makedirs('./data/info', exist_ok=True)
        with open(f'./data/info/{self.get_company_name()}_summary.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.company_info, outfile, indent=4, ensure_ascii=False)

    def download_price_hist(self):
        """Download historical price data as CSV."""
        os.makedirs('./data/price_hist', exist_ok=True)
        self.yf_ticker.history(period="max").to_csv(f'./data/price_hist/{self.ticker}_hist.csv')


def main():
    ticker_symbol = input("Enter ticker symbol (default is 'MSFT'): ")
    if not ticker_symbol.strip():
        print("No ticker symbol provided, using default 'MSFT'.")
        ticker_symbol = 'MSFT'

    collector = SecurityDataCollector(f"{ticker_symbol}")
    print(f"Company Name: {collector.get_company_name()}")
    print("News:", collector.get_news())
    print("Financials:", collector.get_financials())
    collector.download_info()
    collector.download_price_hist()


if __name__ == "__main__":
    main()


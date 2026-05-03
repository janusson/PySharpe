import numpy as np
import pandas as pd
import pytest

from pysharpe.data.linkage import DataLinker


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2023-01-01", periods=10)
    # 10, 20, 30, ..., 100
    prices = np.arange(10, 110, 10)
    return pd.DataFrame(
        {"date": dates, "price": prices, "volume": np.random.randint(100, 1000, 10)}
    )


@pytest.fixture
def sample_macro_data():
    dates = pd.date_range("2023-01-01", periods=10)
    # Assume interest rates are slowly rising
    interest_rates = np.linspace(2.0, 3.0, 10)
    return pd.DataFrame({"date": dates, "interest_rate": interest_rates})


def test_data_linker_initialization():
    linker = DataLinker()
    assert linker.conn is not None
    linker.close()


def test_register_and_execute(sample_market_data):
    linker = DataLinker()
    linker.register_data("my_table", sample_market_data)

    result = linker.execute_query("SELECT COUNT(*) as cnt FROM my_table")
    assert result["cnt"].iloc[0] == 10

    linker.close()


def test_register_invalid_data():
    linker = DataLinker()
    with pytest.raises(TypeError):
        linker.register_data("invalid", [1, 2, 3])
    linker.close()


def test_get_enhanced_market_data_no_macro(sample_market_data):
    linker = DataLinker()
    linker.register_data("market_data", sample_market_data)

    # 3-day rolling window
    result = linker.get_enhanced_market_data(
        market_table="market_data", rolling_window=3
    )

    # Check columns
    assert "price_rolling_avg" in result.columns
    assert "price_lag_1" in result.columns

    # Day 1: lag is NaN, rolling is 10
    assert pd.isna(result["price_lag_1"].iloc[0])
    assert result["price_rolling_avg"].iloc[0] == 10.0

    # Day 2: lag is 10, rolling is (10+20)/2 = 15
    assert result["price_lag_1"].iloc[1] == 10.0
    assert result["price_rolling_avg"].iloc[1] == 15.0

    # Day 3: lag is 20, rolling is (10+20+30)/3 = 20
    assert result["price_lag_1"].iloc[2] == 20.0
    assert result["price_rolling_avg"].iloc[2] == 20.0

    # Day 4: lag is 30, rolling is (20+30+40)/3 = 30
    assert result["price_rolling_avg"].iloc[3] == 30.0

    linker.close()


def test_get_enhanced_market_data_with_macro(sample_market_data, sample_macro_data):
    linker = DataLinker()
    linker.register_data("market_data", sample_market_data)
    linker.register_data("macro_data", sample_macro_data)

    result = linker.get_enhanced_market_data(
        market_table="market_data", macro_table="macro_data", rolling_window=3
    )

    # Check joined columns
    assert "interest_rate" in result.columns
    assert len(result) == 10
    assert result["interest_rate"].iloc[0] == 2.0
    assert result["interest_rate"].iloc[-1] == 3.0

    linker.close()

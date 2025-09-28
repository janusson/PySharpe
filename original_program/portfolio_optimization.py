# ./financio/portfolio_optimization.py
import os
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from matplotlib import pyplot as plt

# Define portfolio to optimize (CAD_equities, CAD_ETFs, USD_equities, USD_ETFs, CAD_portfolio, USD_portfolio)
#TODO: Detect each portfolio file and run optimization
#TODO: Define limiting time constraint for portfolios
#TODO: Asset contraints
#TODO: Geographical exposure 

def plot_portfolio_allocation(weights, portfolio_name):
    # Filter out assets with 0 weighting
    filtered_weights = {asset: weight for asset, weight in weights.items() if weight > 0}
    labels = filtered_weights.keys()
    sizes = filtered_weights.values()
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'{portfolio_name} Allocation')
    plt.savefig(f'./data/exports/{portfolio_name}_allocation.png')
    plt.show()

def optimize_portfolio(portfolio_name, time_constraint=None, asset_constraints=None, geo_exposure=None):
    try:
        portfolio_price_data = pd.read_csv(f'./data/exports/{portfolio_name}_collated.csv', parse_dates=True, index_col='Date')
    except FileNotFoundError:
        print(f"Error: File for {portfolio_name} not found. Skipping optimization.")
        return None, None
    except Exception as e:
        print(f"Error reading file for {portfolio_name}: {e}")
        return None, None

    if time_constraint:
        portfolio_price_data = portfolio_price_data.loc[time_constraint:]

    if portfolio_price_data.empty:
        print(f"Error: No data available for {portfolio_name} after applying time constraint. Skipping optimization.")
        return None, None

    if portfolio_price_data.isnull().values.any():
        print(f"Warning: Missing data found in {portfolio_name}. Filling missing values with forward fill.")
        portfolio_price_data = portfolio_price_data.fillna(method='ffill')

    mu = mean_historical_return(portfolio_price_data)
    S = CovarianceShrinkage(portfolio_price_data).ledoit_wolf()

    ef = EfficientFrontier(mu, S)

    if asset_constraints:
        ef.add_constraint(lambda w: w >= asset_constraints['min_weight'])
        ef.add_constraint(lambda w: w <= asset_constraints['max_weight'])

    if geo_exposure:
        # Implement geographical exposure constraints here
        pass

    try:
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
    except Exception as e:
        print(f"Error during optimization for {portfolio_name}: {e}")
        return None, None

    # Plot portfolio allocation
    plot_portfolio_allocation(cleaned_weights, portfolio_name)

    output_dir = './data/exports/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'{portfolio_name}_weights.txt')
    ef.save_weights_to_file(output_file)

    performance = ef.portfolio_performance(verbose=True)
    print(f'Predicted performance for {portfolio_name} (Expected Annual Return, Annual Volatility, Sharpe Ratio): \n {performance}')

    performance_file = os.path.join(output_dir, f'{portfolio_name}_performance.txt')
    with open(performance_file, 'w') as file:
        file.write(f'Expected annual return: {performance[0]*100:.2f}%\n')
        file.write(f'Annual volatility: {performance[1]*100:.2f}%\n')
        file.write(f'Sharpe Ratio: {performance[2]:.2f}')

    return cleaned_weights, performance

def main():
    portfolio_dir = './data/portfolio/'
    try:
        portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Portfolio directory '{portfolio_dir}' not found.")
        return
    except Exception as e:
        print(f"Error accessing portfolio directory: {e}")
        return

    portfolios = [f.replace('.csv', '') for f in portfolio_files]

    print(f'Found {len(portfolios)} portfolios:')
    for portfolio in portfolios:
        try:
            portfolio_data = pd.read_csv(os.path.join(portfolio_dir, f'{portfolio}.csv'))
            num_equities = portfolio_data.shape[0]
            print(f'- {portfolio}: {num_equities} equities')
        except Exception as e:
            print(f"Error reading portfolio file for {portfolio}: {e}")
            continue

    while True:
        proceed = input('Do you want to optimize all portfolios found? (yes/no): ').strip().lower()
        if proceed in ['yes', 'no']:
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    if proceed == 'yes':
        for portfolio in portfolios:
            optimize_portfolio(portfolio, time_constraint='1980-01-01')
    else:
        print('Optimization aborted.')

if __name__ == '__main__':
    main()
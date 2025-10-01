import matplotlib.pyplot as plt

# Example: Portfolio weights
portfolio_weights = {'AVGO': 0.04069, 'COST': 0.19271, 'LLY': 0.46045, 'NVDA': 0.29677, 'SMCI': 0.00938}

# Filter out zero weights
filtered_weights = {k: v for k, v in portfolio_weights.items() if v > 0}

# Plotting
labels = list(filtered_weights.keys())
sizes = list(filtered_weights.values())

plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Portfolio Weights')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


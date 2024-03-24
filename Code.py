#XGBoost Algorithm
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fetch Bitcoin data from Yahoo Finance
eth_data = yf.download('ETH-USD', start='2016-01-01', end='2023-12-31')

# Feature Engineering
eth_data['Returns'] = eth_data['Close'].pct_change()
eth_data['Price_Rate_Of_Change'] = eth_data['Close'].pct_change(periods=5)
eth_data['Volume_Rate_Of_Change'] = eth_data['Volume'].pct_change(periods=5)

# Drop NaN values
eth_data.dropna(inplace=True)

# Define features and target variable
X = eth_data[['Returns', 'Price_Rate_Of_Change', 'Volume_Rate_Of_Change']]
y = np.where(eth_data['Close'].shift(-1) > eth_data['Close'], 1, 0)  # 1 if price increases, 0 if price decreases

# Split the data into training and testing sets
X_train, X_test = X.loc[:'2022-06-30'], X.loc['2022-09-01':'2023-12-31']
y_train, y_test = y[:len(X_train)], y[len(X_train):len(X_train)+len(X_test)]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [200],
    'max_depth': [6], 
    'learning_rate': [0.01], 
}

xgb_classifier = XGBClassifier(random_state=42)
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_xgb_classifier = XGBClassifier(**best_params, random_state=42)
best_xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = best_xgb_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Use the model to predict signals
predicted_signals_test = best_xgb_classifier.predict(scaler.transform(X_test))
test_data = eth_data.loc['2022-09-01':'2023-12-31']

# Convert predicted_signals_test to a pandas Series
predicted_signals_series = pd.Series(predicted_signals_test, index=test_data.index)

# Calculate actual returns for the testing data
test_data['Actual_Return'] = test_data['Close'].pct_change()

# Convert predicted_signals_test to a pandas Series
predicted_signals_series = pd.Series(predicted_signals_test, index=test_data.index)

# Calculate strategy returns for the testing data
test_data['Strategy_Return'] = test_data['Actual_Return'] * predicted_signals_series.shift(1)

# Calculate cumulative strategy returns for the testing data
test_data['Cumulative_Strategy_Return'] = (1 + test_data['Strategy_Return']).cumprod()

# Calculate cumulative actual returns for the testing data
test_data['Cumulative_Actual_Return'] = (1 + test_data['Actual_Return']).cumprod()

# Calculate metrics
percent_drawdown = ((test_data['Cumulative_Strategy_Return'] - test_data['Cumulative_Strategy_Return'].cummax()) / test_data['Cumulative_Strategy_Return'].cummax()).min()
annual_return_rate = (test_data['Strategy_Return'].mean() * 252)
sharpe_ratio = (annual_return_rate / (test_data['Strategy_Return'].std() * np.sqrt(252)))

# Print metrics
print("Max Percent Drawdown:", percent_drawdown)
print("Annual Return Rate:", annual_return_rate)
print("Sharpe Ratio:", sharpe_ratio)

# Annual Return
annual_return = test_data['Strategy_Return'].mean() * 252

# Cumulative Returns
cumulative_return = test_data['Cumulative_Strategy_Return'][-1]

# Annual Volatility
annual_volatility = test_data['Strategy_Return'].std() * np.sqrt(252)

# Calmar Ratio
max_drawdown = abs(percent_drawdown)
calmar_ratio = annual_return / max_drawdown

# Stability
stability = annual_return / annual_volatility

# Omega Ratio
risk_free_rate = 0.03  # Assumed risk-free rate
omega_ratio = (annual_return - risk_free_rate) / abs(max_drawdown)

# Sortino Ratio
downside_returns = test_data[test_data['Strategy_Return'] < 0]['Strategy_Return']
downside_deviation = downside_returns.std() * np.sqrt(252)
sortino_ratio = (annual_return - risk_free_rate) / downside_deviation

# Skew
skewness = test_data['Strategy_Return'].skew()

# Kurtosis
kurtosis = test_data['Strategy_Return'].kurtosis()

# Tail Ratio
negative_returns = test_data[test_data['Strategy_Return'] < 0]['Strategy_Return']
positive_returns = test_data[test_data['Strategy_Return'] > 0]['Strategy_Return']
tail_ratio = abs(negative_returns.mean() / positive_returns.mean())

# Daily Value at Risk (VaR) at 95% confidence level
daily_var = test_data['Strategy_Return'].quantile(0.05)

# Winning Trades, Losing Trades, Win Rate
winning_trades = (test_data['Strategy_Return'] > 0).sum()
losing_trades = (test_data['Strategy_Return'] < 0).sum()
win_rate = winning_trades / (winning_trades + losing_trades)

# Print all calculated metrics
print("Annual Return:", annual_return)
print("Cumulative Returns:", cumulative_return)
print("Annual Volatility:", annual_volatility)
print("Calmar Ratio:", calmar_ratio)
print("Stability:", stability)
print("Omega Ratio:", omega_ratio)
print("Sortino Ratio:", sortino_ratio)
print("Skew:", skewness)
print("Kurtosis:", kurtosis)
print("Tail Ratio:", tail_ratio)
print("Daily Value at Risk (VaR):", daily_var)
print("Winning Trades:", winning_trades)
print("Losing Trades:", losing_trades)
print("Win Rate:", win_rate)

# Plot cumulative returns over the testing data
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Cumulative_Actual_Return'], label='Actual Returns', color='blue')
plt.plot(test_data.index, test_data['Cumulative_Strategy_Return'], label='Strategy Returns', color='orange')
plt.title('Cumulative Returns (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Plot strategy signals over the testing data
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close'], label='Price', color='black')
buy_signals = test_data.loc[predicted_signals_series == 1]
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
plt.title('Price and Strategy Signals (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot drawdown over the testing data
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, ((test_data['Cumulative_Strategy_Return'] - test_data['Cumulative_Strategy_Return'].cummax()) / test_data['Cumulative_Strategy_Return'].cummax()), label='Drawdown', color='red')
plt.title('Drawdown (Testing Period)')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.show()

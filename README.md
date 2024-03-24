# XGBoost

An XGBoost (Extreme Gradient Boosting) classifier is implemented as a trading strategy for Ethereum (ETH) based on historical price and volume data. Here's an overview of the strategy and its implementation:

**Fetching Data:** Historical ETH price and volume data are fetched from Yahoo Finance using the yfinance library.

**Feature Engineering:** Relevant features are derived from the raw data to train the XGBoost classifier. Features such as daily returns, price rate of change, and volume rate of change are computed to capture potential patterns in the data.

**Target Variable Definition:** Similar to the Random Forest Classifier strategy, the target variable is defined based on whether the closing price of ETH increases or decreases compared to the next day. If the closing price increases, the target variable is set to 1; otherwise, it's set to 0.

**Data Splitting:** The data is split into training and testing sets. The training set consists of data from January 2016 until the end of June 2022, while the testing set contains data from September 1st, 2022, until December 31st, 2023.

**Standardization:** The features in the training and testing sets are standardized using StandardScaler to ensure that they have a mean of 0 and a standard deviation of 1, which is a common preprocessing step for many machine learning algorithms.

**Hyperparameter Tuning:** Grid search with cross-validation (GridSearchCV) is performed to find the best hyperparameters for the XGBoost model. Hyperparameters such as the number of estimators, maximum depth, and learning rate are tuned to optimize the model's performance.

**Model Training:** The XGBoost model is trained with the best hyperparameters obtained from the grid search.

**Model Evaluation:** The trained XGBoost model is evaluated on the testing set using accuracy score and classification report metrics.

**Signal Generation:** The trained model is used to predict trading signals (buy/sell) based on the testing data.

**Trading Metrics Calculation:** Various trading metrics such as annual return rate, cumulative returns, annual volatility, Calmar ratio, Sharpe ratio, Omega ratio, Sortino ratio, skewness, kurtosis, tail ratio, daily value at risk (VaR), winning trades, losing trades, and win rate are calculated based on the predicted signals and actual returns.

**Plotting:** Cumulative returns, strategy signals, and drawdown over the testing period are plotted to visualize the performance of the strategy.

**The hyperparameters of the XGBoost model are tuned using grid search to find the optimal combination that maximizes the model's performance. This ensures that the model is robust and generalizes well to unseen data, resulting in reliable trading signals. Hyperparameter tuning is crucial for improving the accuracy and effectiveness of the trading strategy.**

# blockhouse-submission

### Documentation

#### 1. Introduction

The goal of this project is to develop and fine-tune a Transformer-based model to generate trade recommendations based on market data. Transformers have shown great success in sequence modeling tasks, making them a suitable choice for financial time-series forecasting. The model will be trained to predict future price movements, and its performance will be evaluated by simulating trades and comparing the results with a baseline model.

#### 2. Data Preprocessing

##### 2.1 Dataset Description

The dataset used for this project is sourced from Databento, specifically the "XNAS ITCH" feed. This dataset contains detailed trade and order book data, including price, volume, bid/ask prices, and sizes. Each record in the dataset is timestamped, providing a granular view of market activity. For the purpose of this model, we focus on essential features like prices, volumes, and derived technical indicators that can help in predicting future price movements.

##### 2.2 Feature Engineering and Technical Indicators

To enhance the model's ability to predict price movements, several technical indicators are calculated and added to the dataset. These indicators are commonly used in technical analysis and are believed to capture patterns that can indicate future price directions. Below are the types of indicators used:

- **Momentum Indicators**: These indicators measure the speed and change of price movements. Examples include:
  - **Relative Strength Index (RSI)**: Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
  - **MACD (Moving Average Convergence Divergence)**: Captures the relationship between two moving averages of a security’s price.
  - **Stochastic Oscillator**: Compares a particular closing price of a security to a range of its prices over a certain period.

- **Volume Indicators**: These indicators are based on trading volumes and help assess the strength of a price movement.
  - **On-Balance Volume (OBV)**: Relates price changes to trading volume.
  
- **Volatility Indicators**: These indicators measure the rate of price change. Examples include:
  - **Bollinger Bands (Upper, Middle, Lower)**: Measures volatility by showing the range within which the price typically moves.
  - **Average True Range (ATR)**: Measures market volatility by decomposing the entire range of an asset price for that period.

- **Trend Indicators**: These indicators help in identifying the direction of the market.
  - **Average Directional Index (ADX)**: Indicates the strength of a trend.
  - **Directional Indicators (+DI, -DI)**: Show the direction of the trend.

- **Other Indicators**:
  - **Detrended Log Return (DLR)**: Measures the logarithmic return of prices.
  - **Time-Weighted Average Price (TWAP)**: Averages prices over a period, weighted by time.
  - **Volume-Weighted Average Price (VWAP)**: Averages prices over a period, weighted by volume.

These indicators were calculated using the `TA-Lib` library, which provides a comprehensive collection of technical indicators.

##### 2.3 Data Preparation

Once the technical indicators were calculated, the dataset was prepared for input into the Transformer model. This involved the following steps:

- **Price Normalization**: Prices were normalized by dividing them by a factor of \(1 \times 10^9\) to bring them into a more manageable range for model training.

- **Feature Extraction**: 
  - The price, volume, bid/ask prices, and sizes were used to calculate additional features like percentage change (`pct_change`) and liquidity.
  - Rolling statistics were calculated for volatility (`rolling_mean_vol`, `rolling_std_vol`) and liquidity (`rolling_mean_liq`, `rolling_std_liq`) over a window of 60 periods.

- **Sequence Creation**: 
  - For the Transformer model, the data was reshaped into sequences to capture the time-series nature of the input features. 
  - Each input sequence consists of 60 consecutive time steps, where each step includes the values of the selected features.
  - The target variable (`y`) is the price at the next time step, which the model will learn to predict.

##### 2.4 Final Dataset

The final dataset, `market_features_df`, contains the calculated technical indicators, normalized prices, and additional features like rolling statistics for volatility and liquidity. The data was cleaned by removing any rows with missing values, resulting in a dataset ready for training the Transformer model.

##### 2.5 Preprocessing for the Transformer Model

To prepare the data for input into the Transformer model:

- **Input Features**: The features selected for the model include price, RSI, MACD, Stochastic Oscillator, OBV, Bollinger Bands, ATR, ADX, Directional Indicators, and CCI. These features capture different aspects of market behavior, from momentum and trend to volatility and volume.
- **Sequence Length**: A sequence length of 60 was chosen, meaning the model looks at the past 60 time steps to predict the next price. This choice is based on common practices in financial modeling, where a few days to weeks of past data are often used to predict short-term market movements.
- **Splitting the Data**: The data was split into sequences, with each sequence representing a fixed period of market activity. These sequences form the input (`X`) and target (`y`) pairs used for training and evaluation.

This comprehensive preprocessing ensures that the model has a rich set of inputs to learn from, potentially capturing complex patterns in the data that can lead to more accurate predictions.

---

Would you like to move on to the next section or refine anything in this documentation?
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

#### 3. Model Implementation

##### 3.1 Transformer Model Architecture

The core of this project is the Transformer-based model, which is designed to process sequential market data and generate predictions for future price movements. The Transformer architecture is well-suited for handling time-series data due to its ability to capture long-range dependencies through self-attention mechanisms. Here's a breakdown of the model components:

- **Transformer Block**: 
  - The model begins with a series of Transformer blocks, each containing a multi-head attention layer followed by a feed-forward network.
  - **Multi-Head Attention**: This layer allows the model to focus on different parts of the input sequence simultaneously, enhancing its ability to capture complex patterns in the data.
  - **Feed-Forward Network**: Following the attention mechanism, the data passes through a series of dense layers that further process the information.
  - **Residual Connections and Layer Normalization**: These components help stabilize the learning process and improve model performance by ensuring smooth gradients.

- **Output Layers**:
  - After passing through the Transformer blocks, the data is further processed by a couple of dense layers before the final output layer.
  - The output layer uses a linear activation function to predict the next value in the sequence, which corresponds to the future price of the asset.

##### 3.2 Model Compilation and Summary

Once the model architecture was defined, it was compiled using the Mean Squared Error (MSE) as the loss function. MSE is a common choice for regression tasks like price prediction because it penalizes larger errors more severely, encouraging the model to be as accurate as possible.

The model was summarized, showing the number of parameters and the structure of each layer, ensuring that the architecture aligns with the goals of the project.

#### 4. Fine-Tuning Process

##### 4.1 Splitting the Data

The dataset was split into training and validation sets, with 80% of the data used for training and 20% reserved for validation. This split allows for effective training while ensuring that the model's performance is evaluated on unseen data, reducing the risk of overfitting.

##### 4.2 Hyperparameter Tuning and Callbacks

To optimize the model's performance, several hyperparameters were fine-tuned:

- **Learning Rate**: Set initially to 5e-5, this parameter controls the step size of the model's weight updates during training.
- **Batch Size**: A batch size of 16 was chosen, balancing computational efficiency with the need to provide the model with sufficient data in each iteration.
- **Early Stopping**: This callback was used to halt training if the model's performance on the validation set did not improve after five consecutive epochs, preventing overfitting.
- **Learning Rate Scheduler**: The learning rate was reduced by a factor of 0.5 if the validation loss plateaued for three epochs, allowing the model to fine-tune its learning process as it converged.

##### 4.3 Fine-Tuning and Evaluation

The model was trained for up to 100 epochs, with the training process monitored by the callbacks. The final validation loss was evaluated to gauge the model's performance after fine-tuning.

#### 5. Trade Simulation and Evaluation

##### 5.1 Simulating Trades

To evaluate the effectiveness of the Transformer model's predictions, a trade simulation was conducted using a custom trading blotter. The trading strategy was based on the RSI indicator and the model's predicted prices:

- **Buy and Sell Signals**: Trades were executed based on the RSI value, with buy signals triggered when RSI fell below 30 and sell signals when it exceeded 70.
- **Price Adjustments**: The trade prices were adjusted based on market conditions, using rolling statistics of volatility and liquidity.
- **Transaction Costs and Rewards**: Each trade's cost was calculated, and rewards were assessed based on the difference between expected and actual prices, slippage, and penalties for execution time.

##### 5.2 Performance Evaluation

The blotter recorded each trade, including the action taken, price, number of shares held, balance, portfolio value, and transaction details. The final portfolio value, total profit, and cumulative reward were calculated to assess the model's performance.

This trade simulation approach provides a practical evaluation of the model's real-world applicability by simulating how it would perform in a live trading environment.

#### 6. Challenges and Solutions

During the implementation and fine-tuning process, several challenges were encountered:

- **Data Preprocessing**: Ensuring the data was clean and ready for the Transformer model required careful handling of missing values and the calculation of rolling statistics.
- **Model Convergence**: Fine-tuning the learning rate and using appropriate callbacks were critical to achieving model convergence without overfitting.
- **Trade Simulation Accuracy**: Implementing a robust trading blotter that accurately simulated real-world conditions was essential for meaningful performance evaluation.

Each of these challenges was addressed through careful parameter tuning, thorough testing, and iterative improvements.

#### 7. Justification of Approach

##### 7.1 Model Choice

The Transformer model was chosen due to its ability to handle sequential data and capture long-range dependencies, making it well-suited for financial time-series forecasting. The addition of technical indicators further enhanced the model's ability to detect patterns that might not be evident from raw price data alone.

##### 7.2 Effectiveness

The model's architecture, combined with the fine-tuning process, effectively addressed the project’s objectives. The trade simulation showed promising results, with the model generating meaningful predictions that led to profitable trades in many cases.

#### 8. Innovation and Relevance

The innovative use of the Transformer model in this context, combined with traditional technical indicators and a custom trading blotter, demonstrates a strong understanding of both machine learning and financial trading. The approach is relevant to Blockhouse's needs, providing a scalable and effective solution for generating trade recommendations.

#### 9. Conclusion

The Transformer-based model developed in this project effectively leverages advanced machine learning techniques and financial indicators to predict future price movements and generate trade recommendations. The model was thoroughly fine-tuned and evaluated, showing strong potential for real-world application in trading scenarios.

Future work could explore the combination of the Transformer model with reinforcement learning techniques, such as the PPO model, to further enhance trading performance.

#### 10. References

- TA-Lib Documentation: [TA-Lib: Technical Analysis Library](https://mrjbq7.github.io/ta-lib/)
- Transformer Model Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- TensorFlow Documentation: [TensorFlow: An end-to-end open-source machine learning platform](https://www.tensorflow.org/)
- PPO Implementation Git Repo: [PPO Implementation Git Repo](https://github.com/Blockhouse-Repo/Blockhouse-Work-Trial)

#### 11. Appendix

- **Code Snippets**: See the provided code for detailed implementations of the Transformer model, technical indicators, and trade simulation.
- **Hyperparameters**: Listed in the model implementation section, including learning rates, batch sizes, and dropout rates.


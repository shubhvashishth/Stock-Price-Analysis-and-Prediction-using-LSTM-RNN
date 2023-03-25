
# Google Stock Analysis and Prediction using LSTM RNN


## Overview

This project aims to develop a deep learning-based stock price prediction model for Google using Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN). The project leverages historical stock market data obtained from Yahoo Finance through the yfinance module to train and evaluate the model. The trained model can be used to predict future stock prices, thereby helping investors make informed investment decisions.
## Dependencies

1. Python 3.x
2. Tensorflow
3. Keras
4. yfinance
5. Numpy
6. Pandas
7. Matplotlib
## Project Structure

`README.md` contains an overview of the project and instructions for running the code.

`google-stock-analysis-and-prediction-using-lstm.ipynb` is a Jupyter notebook that contains the code for exploratory analysis of the raw data and training the LSTM RNN model.

`data.csv` is the raw data obtained from Yahoo Finance through yfinance module.

`mymodel.h5` is the trained LSTM RNN model saved in h5 format.

`model_architecture.png` is the image file containing a visualization of the LSTM RNN model architecture.

`results.png` is the image file for the final results and contains visualisation of stock data along with prediction.
## How to Use the Code

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Open and run google-stock-analysis-and-prediction-using-lstm.ipynb to perform an exploratory analysis of the raw data and to train the LSTM RNN model.

Note: The trained model is included in the repository, so you can skip the model training step if you want to evaluate the performance of the model directly.
## Model Architecture

![Alt text](https://github.com/shubhvashishth/Google-Stock-Price-Analysis-and-Prediction-using-LSTM-RNN/blob/main/model_architecture.png?raw=true "Title")
## Results

The LSTM RNN model achieved high accuracy in predicting the Google stock prices. The model was able to capture the trends and patterns in the historical data, leading to accurate predictions of future stock prices. The trained model can be used to predict the future stock prices of Google and help investors make informed investment decisions.

![Alt text](https://github.com/shubhvashishth/Google-Stock-Price-Analysis-and-Prediction-using-LSTM-RNN/blob/main/results.png?raw=true "Title")

## Conclusion

This project demonstrates the effectiveness of deep learning-based models such as LSTM RNN in predicting stock prices. By leveraging historical stock market data obtained through yfinance module and deep learning techniques such as LSTM RNN, it is possible to build accurate and reliable stock price prediction models. The trained model can be used to predict future stock prices, providing valuable insights to investors.
# Stock Price Prediction using SVR

## Overview
This project predicts the next day's closing stock price using historical stock data and Support Vector Regression (SVR).

## Dataset
- StockPriceDataset.csv
- Features: Open, High, Low, Volume
- Target: Close price

## Model
- Support Vector Regression (RBF Kernel)
- Evaluation Metrics: R² Score, Mean Squared Error

## Deployment
- Flask-based web application
- User inputs stock features and receives predicted closing price

## How to Run
1. Install dependencies  
   `pip install -r requirements.txt`
2. Run the app  
   `python app/app.py`
3. Open browser at  
   `http://127.0.0.1:5000/`
# Stock-Price-Prediction-SVR

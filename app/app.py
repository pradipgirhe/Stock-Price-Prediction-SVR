from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('../models/svr_model.pkl', 'rb'))
scaler_x = pickle.load(open('../models/scaler_x.pkl', 'rb'))
scaler_y = pickle.load(open('../models/scaler_y.pkl', 'rb'))

@app.route('/')
def home():
    df = pd.read_csv('../data/StockPriceDataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    historical_df = df.sort_values('Date').tail(30)
    historical_data = {
        'dates': list(historical_df['Date'].dt.strftime('%Y-%m-%d')),
        'prices': list(historical_df['Close'])
    }
    return render_template('index.html', historical_data=historical_data)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler_x.transform([features])
    prediction_scaled = model.predict(features_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1))
    df = pd.read_csv('../data/StockPriceDataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    historical_df = df.sort_values('Date').tail(30)
    historical_data = {
        'dates': list(historical_df['Date'].dt.strftime('%Y-%m-%d')),
        'prices': list(historical_df['Close'])
    }
    return render_template('index.html', prediction_text='Predicted Closing Price: {:.2f}'.format(prediction[0][0]), historical_data=historical_data)

if __name__ == '__main__':
    app.run(debug=True)

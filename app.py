from flask import Flask, request, jsonify, render_template
import numpy as np
import yfinance as yf
from tensorflow import keras
model = keras.models.load_model('bitcoin_price_lstm_model.h5')
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


app = Flask(__name__)

# Load the trained model & scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch latest Bitcoin data
        btc = yf.Ticker('BTC-USD')
        data = btc.history(period="180d")

        if data.empty:
            return jsonify({'error': 'No Bitcoin data available'}), 400

        # Prepare data for prediction
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)

        # Use last 60 days as input
        seq_length = 60
        X_input = scaled_prices[-seq_length:].reshape(1, seq_length, 1)

        # Predict Bitcoin price
        predicted_price_scaled = model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(predicted_price_scaled)[0][0])

        # Get latest Bitcoin metrics
        latest_price = float(data['Close'].iloc[-1])
        high_price = float(data['High'].iloc[-1])
        low_price = float(data['Low'].iloc[-1])

        # Provide suggestion (Buy, Sell, Hold)
        suggestion = "Hold"
        if predicted_price > latest_price:
            suggestion = "Buy"
        elif predicted_price < latest_price:
            suggestion = "Sell"

        # Generate graph
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Bitcoin Price')
        plt.title("Bitcoin Price Trend (Last 1 Year)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        graph_path = "static/bitcoin_price_graph.png"
        plt.savefig(graph_path)
        plt.close()

        return jsonify({
            'predicted_price': predicted_price,
            'latest_price': latest_price,
            'high_price': high_price,
            'low_price': low_price,
            'suggestion': suggestion,
            'graph': graph_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
